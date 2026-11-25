"""`profile` and `check`: summaries and plausibility tests that print no raw rows.

profile: column names, dtypes, missing fractions, value ranges and sampling rate.
Value ranges (and the share of True for a flag) are given only for the columns that
the tool recognizes as measurements: the numeric and flag columns of the schema
(neuralfuel.schema) and the raw columns that the adapter lists in MEASUREMENTS. Every
other column gives only its type and its share of missing values. Ranges are also
withheld for a recognized column with a name like lat/lon/time/date/reg/tail/flight,
and for numbers that look like epoch times. Text and datetime columns never give
ranges.

check: physical plausibility after the adapter, to catch wrong units and scale factors.
Every test names the column it suspects when it fails. It also reports the share of
samples in flight (CAS >= 100 kt) that are clean, with speed brake, with gear down
and with anti-ice on, overall and by altitude band, so that a wrong decoding of a
configuration flag is visible.
"""

import re

import numpy as np
import pandas as pd

from . import atmos
from .prep import airspeed_and_temperature
from .schema import COLUMNS, validate

SCHEMA_MEASUREMENTS = tuple(c.name for c in COLUMNS if c.kind in ("float", "bool"))
SCHEMA_TEXT = tuple(c.name for c in COLUMNS if c.kind in ("str", "time"))

SENSITIVE = re.compile(
    r"lat|lon|position|pos_|^pos$|time|date|utc|epoch|reg|tail|flight|callsign|icao|"
    r"leg|origin|dest|airport|route|squawk",
    re.IGNORECASE,
)


def _quantiles(v):
    q = np.nanquantile(v, [0, 0.01, 0.5, 0.99, 1])
    return dict(
        zip(["min", "p1", "p50", "p99", "max"], [float(x) for x in q], strict=True)
    )


def profile(df: pd.DataFrame, measurements=()) -> dict:
    """Column summary without identifiers, dates, positions or rows. measurements:
    names of raw columns that are measurements (the MEASUREMENTS of an adapter); with
    the numeric and flag columns of the schema they are the only columns with value
    ranges."""
    known = set(SCHEMA_MEASUREMENTS) | {str(m) for m in measurements}
    cols = []
    for name in df.columns:
        s = df[name]
        r = {
            "name": str(name),
            "dtype": str(s.dtype),
            "missing_fraction": float(s.isna().mean()),
        }
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if str(name) not in known and str(name) not in SCHEMA_TEXT:
            r["ranges"] = "withheld (not a recognized measurement column)"
        elif pd.api.types.is_bool_dtype(s):
            r["true_fraction"] = float(s.mean())
        elif numeric:
            v = s.to_numpy(dtype=float)
            fin = v[np.isfinite(v)]
            if SENSITIVE.search(str(name)) or (
                len(fin) and np.median(np.abs(fin)) > 9e8
            ):
                r["ranges"] = "withheld (possible identifier, time or position)"
            elif len(fin):
                r.update(_quantiles(fin))
                r["distinct_values"] = int(min(len(np.unique(fin[:200000])), 1000))
        else:
            if str(name) in SCHEMA_TEXT:
                r["distinct_values"] = int(s.nunique())
            r["ranges"] = "withheld (text or time column)"
        cols.append(r)
    out = {"rows": len(df), "columns": cols}
    out["sampling"] = sampling(df)
    return out


def sampling(df: pd.DataFrame) -> dict:
    """Time step statistics, if a datetime column and a flight column can be found."""
    tcol = (
        "time"
        if "time" in df
        else next(
            (c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])), None
        )
    )
    fcol = (
        "flight_id"
        if "flight_id" in df
        else next(
            (c for c in df.columns if re.search("flight|leg", str(c), re.IGNORECASE)),
            None,
        )
    )
    if tcol is None or not pd.api.types.is_datetime64_any_dtype(df[tcol]):
        return {"status": "no datetime column found"}
    t = df[tcol]
    keys = [fcol, tcol] if fcol else [tcol]
    d = df[keys].sort_values(keys, kind="stable")
    dt = d[tcol].diff().dt.total_seconds().to_numpy()
    if fcol:
        same = d[fcol].to_numpy()[1:] == d[fcol].to_numpy()[:-1]
        dt = dt[1:][same]
    else:
        dt = dt[1:]
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return {"status": "too few samples"}
    return {
        "time_column": str(tcol),
        "flight_column": str(fcol) if fcol else None,
        "flights": int(df[fcol].nunique()) if fcol else None,
        "median_step_s": float(np.median(dt)),
        "share_steps_1s": float(np.mean(dt == 1.0)),
        "share_steps_above_10s": float(np.mean(dt > 10.0)),
        "duplicate_times": int((dt == 0).sum()),
        "tz_aware": getattr(t.dt, "tz", None) is not None,
    }


def _result(name, column, ok, value, expected, msg, warn_only=False):
    status = "pass" if ok else ("warn" if warn_only else "fail")
    return {
        "test": name,
        "column": column,
        "status": status,
        "value": None if value is None or not np.isfinite(value) else float(value),
        "expected": expected,
        "message": "" if ok else msg,
    }


IN_FLIGHT_CAS_KT = 100.0
MASS_GAP_S, MASS_MIN_S = 10.0, 1800.0  # mass against fuel: gap, shortest segment
FLAG_BANDS_FT = (
    ("below 10000 ft", -1e9, 10000.0),
    ("10000 to 25000 ft", 10000.0, 25000.0),
    ("above 25000 ft", 25000.0, 1e9),
)


def flag_shares(d: pd.DataFrame, cas_kt) -> dict:
    """Share of samples in flight (CAS >= IN_FLIGHT_CAS_KT) with each configuration
    flag, overall and per altitude band. Flags missing from the table are None."""
    fly = np.asarray(cas_kt) >= IN_FLIGHT_CAS_KT
    alt = d.alt_ft.to_numpy(dtype=float)
    names = ("clean", "speedbrake", "gear_down", "anti_ice")
    out = {
        "definition": f"samples with CAS >= {IN_FLIGHT_CAS_KT:g} kt",
        "samples": int(fly.sum()),
    }

    def shares(sel):
        return {
            k: (
                float(d[k].to_numpy(dtype=bool)[sel].mean())
                if k in d and sel.any()
                else None
            )
            for k in names
        }

    out["all"] = shares(fly)
    out["bands"] = {
        name: {
            "samples": int((fly & (alt >= lo) & (alt < hi)).sum()),
            **shares(fly & (alt >= lo) & (alt < hi)),
        }
        for name, lo, hi in FLAG_BANDS_FT
    }
    return out


def check(df: pd.DataFrame) -> dict:
    """Schema validation and plausibility tests. Returns {'ok', 'schema', 'tests',
    'flags'}."""
    problems = validate(df)
    out = {"schema": [p.as_dict() for p in problems], "tests": []}
    if any(p.level == "error" for p in problems):
        out["ok"] = False
        return out
    t = out["tests"]
    d = df.sort_values(["flight_id", "time"], kind="stable").reset_index(drop=True)
    alt = d.alt_ft.to_numpy(dtype=float)
    mach, sat, cas = airspeed_and_temperature(d)
    spd = next(c for c in ("cas_kt", "mach", "tas_kt") if c in d)
    tmp = "tat_k" if "tat_k" in d else "sat_k"
    clean = d.clean.to_numpy(dtype=bool)
    hi = (alt > 25000) & clean
    air = (alt > 10000) & clean

    # sampling rate
    s = sampling(d)
    med = s.get("median_step_s", np.nan)
    t.append(
        _result(
            "sampling rate",
            "time",
            med == 1.0,
            med,
            "median step 1 s",
            "the pipeline expects 1 Hz samples; resample or check the time unit",
        )
    )
    share = s.get("share_steps_1s", np.nan)
    t.append(
        _result(
            "regular 1 s steps",
            "time",
            share >= 0.9,
            share,
            ">= 0.9 of steps",
            "many gaps or irregular steps; check duplicate or unsorted times",
            True,
        )
    )

    # altitude
    if len(alt):
        top = d.groupby("flight_id").alt_ft.max().median()
        t.append(
            _result(
                "cruise altitude",
                "alt_ft",
                15000 <= top <= 47000,
                top,
                "median of flight maxima in [15000, 47000] ft",
                "pressure altitude not in ft (metres? flight level?)",
            )
        )

    # Mach
    if air.any():
        share = np.mean((mach[air] > 0.2) & (mach[air] < 0.92))
        t.append(
            _result(
                "Mach range above 10,000 ft",
                spd,
                share >= 0.99,
                share,
                ">= 0.99 of samples with 0.2 < M < 0.92",
                f"Mach out of range; check the unit of {spd} and of alt_ft",
            )
        )
    if hi.any():
        mc = np.nanmedian(mach[hi])
        t.append(
            _result(
                "cruise Mach",
                spd,
                0.55 <= mc <= 0.88,
                mc,
                "median in [0.55, 0.88]",
                f"cruise Mach implausible; check the unit or scale of {spd}",
            )
        )

        # temperature against ISA
        dev = sat[hi] - atmos.isa_temperature(alt[hi])
        md = np.nanmedian(dev)
        t.append(
            _result(
                "temperature against ISA",
                tmp,
                -25 <= md <= 25,
                md,
                "median SAT - ISA in [-25, 25] K above 25,000 ft",
                "temperature far from ISA; check the scale factor and unit (K) of "
                f"{tmp}",
            )
        )
        sd = np.nanstd(dev)
        t.append(
            _result(
                "temperature spread",
                tmp,
                sd <= 12,
                sd,
                "SD of SAT - ISA <= 12 K",
                f"temperature spread too large; check the scale factor of {tmp}",
                True,
            )
        )

        # fuel flow magnitude
        r = np.nanmedian(d.ff_kgh.to_numpy()[hi] / d.mass_kg.to_numpy()[hi])
        t.append(
            _result(
                "cruise fuel flow per mass",
                "ff_kgh",
                0.012 <= r <= 0.09,
                r,
                "median fuel flow / mass in [0.012, 0.09] per hour",
                "fuel flow or mass in the wrong unit (kg/s, lb/h, t, lb?)",
            )
        )

        # groundspeed against true airspeed
        tas = mach[hi] * atmos.sound_speed(sat[hi]) / atmos.KT
        g = np.nanmedian(d.gs_kt.to_numpy()[hi] / tas)
        t.append(
            _result(
                "groundspeed against TAS",
                "gs_kt",
                0.7 <= g <= 1.4,
                g,
                "median groundspeed / TAS in [0.7, 1.4] above 25,000 ft",
                "groundspeed not in kt (m/s? km/h?)",
            )
        )

    # mass decrease against integrated fuel flow, in continuous segments (same flight,
    # no step above MASS_GAP_S) of at least MASS_MIN_S
    ts = d.groupby("flight_id").time.transform("min")
    sec = (d.time - ts).dt.total_seconds().to_numpy()
    fid = d.flight_id.to_numpy()
    step = np.r_[np.diff(sec), np.inf]
    last = np.r_[fid[1:] != fid[:-1], True] | (step > MASS_GAP_S) | (step < 0)
    seg = np.r_[0, np.cumsum(last)[:-1]]
    dt = np.where(last, 0.0, step)
    fuel = pd.Series(d.ff_kgh.to_numpy() * dt / 3600).groupby(seg).sum()
    dur = pd.Series(sec).groupby(seg).agg(lambda v: v.max() - v.min())
    burn = d.mass_kg.groupby(seg).apply(
        lambda m: np.nanmedian(m.to_numpy()[:60]) - np.nanmedian(m.to_numpy()[-60:])
    )
    ok = (dur >= MASS_MIN_S) & (fuel > 0)
    name = "mass decrease against integrated fuel flow"
    expected = (
        "median (mass decrease / integrated fuel flow) in [0.8, 1.25] over continuous "
        f"segments of at least {MASS_MIN_S / 60:g} min"
    )
    if ok.any():
        ratio = float(np.median(burn[ok] / fuel[ok]))
        t.append(
            _result(
                name,
                "ff_kgh/mass_kg",
                0.8 <= ratio <= 1.25,
                ratio,
                expected,
                "mass and fuel flow do not agree; check the units of ff_kgh (kg/h) "
                "and mass_kg (kg)",
            )
        )
    else:
        t.append(
            _result(
                name,
                "ff_kgh/mass_kg",
                False,
                None,
                expected,
                f"insufficient coverage: no continuous segment (steps up to "
                f"{MASS_GAP_S:g} s) of at least {MASS_MIN_S / 60:g} min",
                True,
            )
        )

    # engine fuel flows
    if "ff1_kgh" in d and "ff2_kgh" in d:
        tot = d.ff_kgh.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.nanmedian(
                np.abs(tot - d.ff1_kgh.to_numpy() - d.ff2_kgh.to_numpy()) / tot
            )
        t.append(
            _result(
                "engine fuel flows add up",
                "ff1_kgh/ff2_kgh",
                e <= 0.02,
                e,
                "median |ff - ff1 - ff2| / ff <= 0.02",
                "engine fuel flows do not add up to the total; check their units",
            )
        )

    # configuration flag
    up = alt > 20000
    if up.any():
        c = float(d.clean.to_numpy()[up].mean())
        t.append(
            _result(
                "clean configuration above 20,000 ft",
                "clean",
                c >= 0.95,
                c,
                ">= 0.95 of samples clean",
                "the clean flag is inverted or wrong",
            )
        )
    # optional configuration flags
    out["flags"] = flag_shares(d, cas)
    if up.any() and "gear_down" in d:
        c = float(d.gear_down.to_numpy(dtype=bool)[up].mean())
        t.append(
            _result(
                "gear down above 20,000 ft",
                "gear_down",
                c <= 0.01,
                c,
                "<= 0.01 of samples",
                "the gear flag is inverted or wrong",
            )
        )
    hi_sb = alt > 30000
    if hi_sb.any() and "speedbrake" in d:
        c = float(d.speedbrake.to_numpy(dtype=bool)[hi_sb].mean())
        t.append(
            _result(
                "speed brake above 30,000 ft",
                "speedbrake",
                c <= 0.02,
                c,
                "<= 0.02 of samples",
                "speed brake flag often on in cruise; check its decoding (inverted, or "
                "roll spoilers counted)",
                True,
            )
        )
    if "anti_ice" in d and len(d):
        c = float(d.anti_ice.to_numpy(dtype=bool).mean())
        t.append(
            _result(
                "anti-ice share",
                "anti_ice",
                c <= 0.5,
                c,
                "<= 0.5 of samples",
                "anti-ice on in most samples; the flag may be inverted",
                True,
            )
        )
    out["ok"] = not any(x["status"] == "fail" for x in t)
    return out
