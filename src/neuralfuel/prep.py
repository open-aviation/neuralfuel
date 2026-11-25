"""Preprocessing: 1 Hz samples in the schema to 10 s blocks and steady 300 s windows.

Steps (see `prepare`):
1. derived quantities per sample: Mach, SAT, CAS, ISA pressure ratio, true airspeed;
2. Savitzky-Golay derivatives (21 s, order 2) of pressure altitude and groundspeed
   inside each segment (same flight, 1 s steps);
3. inertial energy inputs sin_gamma = (geometric vertical speed) / V_a and
   acc_in = (V_g / V_a) (dV_g/dt) / g;
4. sample filters: clean configuration, speed brake retracted and landing gear up and
   locked (if the columns `speedbrake` and `gear_down` exist), engine anti-ice off (only
   if `exclude_anti_ice` is set), CAS >= 180 kt, fuel flow > 0, engine fuel-flow
   imbalance <= 15 %, finite values;
5. 10 s block means; a block is kept only if all its samples pass the filters; the
   block carries `tail` and `anti_ice` (True if anti-ice is on in any of its samples)
   when the samples have them;
6. phase labels and a chronological 80/10/10 split by flight;
7. steady 300 s cruise windows (positive fuel flow in every sample, finite model
   inputs and target); without a window the table is empty with the same columns.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from . import atmos
from .atmos import FT, G0, KT, T0
from .schema import require_valid

FEATURES = ["mach", "ln_delta", "ln_m", "sin_gamma", "acc_in", "ln_theta"]
CONFIG_FLAGS = ("speedbrake", "gear_down", "anti_ice")  # optional schema flags


@dataclass
class PrepConfig:
    block_s: int = 10
    sg_window: int = 21  # samples (1 Hz)
    sg_order: int = 2
    min_cas_kt: float = 180.0
    max_ff_imbalance: float = 0.15
    level_fpm: float = 300.0  # |geometric vertical speed| below this is level
    cruise_alt_ft: float = 25000.0
    split: tuple = (0.8, 0.1)  # train, validation; the rest is test
    win_s: int = 300
    win_min_alt_ft: float = 25000.0
    win_lookahead_s: int = 60
    win_alt_change_ft: float = 100.0  # |alt(t + 60 s) - alt(t)| below this
    win_mach_sd: float = 0.003
    win_alt_range_ft: float = 150.0
    win_roll_deg: float = 5.0
    win_roll_share: float = 0.95
    win_ff_imbalance: float = 0.05
    exclude_anti_ice: bool = False  # also remove samples with engine anti-ice on


def check_config(cfg: PrepConfig) -> None:
    """Raise ValueError for a setting out of range."""
    for k in ("block_s", "sg_window", "sg_order", "win_s", "win_lookahead_s"):
        v = getattr(cfg, k)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"{k} must be an integer of 1 or more, got {v!r}")
    if cfg.sg_window % 2 == 0 or cfg.sg_window <= cfg.sg_order:
        raise ValueError("sg_window must be odd and larger than sg_order")
    a, b = cfg.split
    if not (0 < a and 0 < b and a + b < 1):
        raise ValueError("split: train and validation shares above 0, sum below 1")
    if cfg.level_fpm <= 0:
        raise ValueError("level_fpm must be positive")


# ---------------------------------------------------------------- per sample
def segment_ids(flight_codes, time_s):
    """Segment number per sample: a new segment starts at a new flight or a time step
    other than 1 s. Input must be sorted by flight and time."""
    n = len(time_s)
    brk = np.ones(n, dtype=bool)
    brk[1:] = (flight_codes[1:] != flight_codes[:-1]) | (np.diff(time_s) != 1)
    return np.cumsum(brk) - 1


def sg_derivative(x, seg, window=21, order=2, dt=1.0):
    """Savitzky-Golay first derivative inside each segment; NaN in segments shorter
    than the window."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    start = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    end = np.r_[start[1:], len(x)]
    for a, b in zip(start, end, strict=True):
        if b - a >= window:
            out[a:b] = savgol_filter(x[a:b], window, order, deriv=1, delta=dt)
    return out


def energy_inputs(dhp_ms, gs_ms, dgs_ms2, tas_ms, sat_k, t_isa_k):
    """Inertial energy-rate inputs.

    dhp_ms: pressure-altitude rate (m/s); gs_ms, dgs_ms2: groundspeed and its rate;
    tas_ms: true airspeed. The geometric rate is dhp * SAT / T_ISA.
    Returns (geometric vertical speed m/s, sin_gamma, acc_in).
    """
    dhg = dhp_ms * sat_k / t_isa_k
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_gamma = dhg / tas_ms
        acc_in = (gs_ms / tas_ms) * dgs_ms2 / G0
    return dhg, sin_gamma, acc_in


def airspeed_and_temperature(d: pd.DataFrame):
    """Mach, SAT (K) and CAS (kt) from the columns present (CAS > Mach > TAS;
    TAT > SAT)."""
    alt = d["alt_ft"].to_numpy(dtype=float)
    tat = d["tat_k"].to_numpy(dtype=float) if "tat_k" in d else None
    sat = None if tat is not None else d["sat_k"].to_numpy(dtype=float)
    if "cas_kt" in d:
        mach = atmos.mach_from_cas(d["cas_kt"].to_numpy(dtype=float), alt)
    elif "mach" in d:
        mach = d["mach"].to_numpy(dtype=float)
    elif tat is not None:
        mach = atmos.mach_from_tas_tat(d["tas_kt"].to_numpy(dtype=float), tat)
    else:
        mach = atmos.mach_from_tas(d["tas_kt"].to_numpy(dtype=float), sat)
    if tat is not None:
        sat = atmos.sat_from_tat(tat, mach)
    if "cas_kt" in d:
        cas = d["cas_kt"].to_numpy(dtype=float)
    else:
        cas = atmos.cas_from_mach(mach, alt)
    return mach, sat, cas


def derive(df: pd.DataFrame, cfg: PrepConfig | None = None):
    """Per-sample derived quantities, energy inputs and the filter flag `keep`.

    Returns (samples, filter counts, flight start times). The samples are sorted by
    flight and time; `seg` is the segment number and `time_s` the time since the first
    sample of the flight.
    """
    cfg = cfg or PrepConfig()
    d = df.sort_values(["flight_id", "time"], kind="stable").reset_index(drop=True)
    codes = pd.factorize(d["flight_id"], sort=True)[0]
    t0 = d.groupby("flight_id")["time"].transform("min")
    time_s = (d["time"] - t0).dt.total_seconds().to_numpy()
    alt = d["alt_ft"].to_numpy(dtype=float)
    gs = d["gs_kt"].to_numpy(dtype=float)
    mach, sat, cas = airspeed_and_temperature(d)
    t_isa = atmos.isa_temperature(alt)
    tas_ms = mach * atmos.sound_speed(sat)
    seg = segment_ids(codes, time_s)
    win, order = cfg.sg_window, cfg.sg_order
    dhp = sg_derivative(alt * FT, seg, win, order)
    dgs = sg_derivative(gs * KT, seg, win, order)
    dhg, sin_gamma, acc_in = energy_inputs(dhp, gs * KT, dgs, tas_ms, sat, t_isa)
    out = pd.DataFrame(
        {
            "flight_id": d["flight_id"].to_numpy(),
            "flight_code": codes,
            "seg": seg,
            "time_s": time_s,
            "alt_ft": alt,
            "cas_kt": cas,
            "mach": mach,
            "sat_k": sat,
            "delta": atmos.isa_delta(alt),
            "theta": sat / T0,
            "dT_isa": sat - t_isa,
            "tas_kt": tas_ms / KT,
            "gs_kt": gs,
            "mass_kg": d["mass_kg"].to_numpy(dtype=float),
            "ff_kgh": d["ff_kgh"].to_numpy(dtype=float),
            "dhg": dhg,
            "sin_gamma": sin_gamma,
            "acc_in": acc_in,
            "clean": d["clean"].to_numpy(dtype=bool),
        }
    )
    for c in ("tail", "ff1_kgh", "ff2_kgh", "roll_deg", *CONFIG_FLAGS):
        if c in d:
            out[c] = d[c].to_numpy()
    out["keep"], counts = sample_filter(out, cfg)
    return out, counts, d.groupby("flight_id")["time"].min()


def config_steps(s: pd.DataFrame, cfg: PrepConfig):
    """(name, flag) of the configuration filters: clean, and the optional flags."""
    steps = [("clean configuration", s["clean"].to_numpy(dtype=bool))]
    if "speedbrake" in s:
        steps.append(("speed brake retracted", ~s["speedbrake"].to_numpy(dtype=bool)))
    if "gear_down" in s:
        steps.append(("gear up", ~s["gear_down"].to_numpy(dtype=bool)))
    if cfg.exclude_anti_ice and "anti_ice" in s:
        steps.append(("anti-ice off", ~s["anti_ice"].to_numpy(dtype=bool)))
    return steps


def sample_filter(s: pd.DataFrame, cfg: PrepConfig):
    """Filter flag per sample and the number of samples left after each filter."""
    steps = config_steps(s, cfg)
    steps.append(
        (f"CAS >= {cfg.min_cas_kt:g} kt", s["cas_kt"].to_numpy() >= cfg.min_cas_kt)
    )
    steps.append(("fuel flow > 0", s["ff_kgh"].to_numpy() > 0))
    if "ff1_kgh" in s and "ff2_kgh" in s:
        f1, f2 = s["ff1_kgh"].to_numpy(dtype=float), s["ff2_kgh"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            imb = np.abs(f1 - f2) / (0.5 * (f1 + f2))
        steps.append(
            (
                f"engine imbalance <= {cfg.max_ff_imbalance:g}",
                imb <= cfg.max_ff_imbalance,
            )
        )
    need = [
        "mach",
        "sat_k",
        "delta",
        "tas_kt",
        "gs_kt",
        "mass_kg",
        "ff_kgh",
        "sin_gamma",
        "acc_in",
    ]
    steps.append(
        ("finite values", np.isfinite(s[need].to_numpy(dtype=float)).all(axis=1))
    )
    keep = np.ones(len(s), dtype=bool)
    counts = {"samples": len(s)}
    for name, k in steps:
        keep &= k
        counts[f"after {name}"] = int(keep.sum())
    return keep, counts


# ---------------------------------------------------------------- blocks
BLOCK_MEANS = [
    "time_s",
    "alt_ft",
    "cas_kt",
    "mach",
    "sat_k",
    "delta",
    "theta",
    "dT_isa",
    "tas_kt",
    "gs_kt",
    "mass_kg",
    "ff_kgh",
    "dhg",
    "sin_gamma",
    "acc_in",
]


def add_log_inputs(t: pd.DataFrame) -> pd.DataFrame:
    """Model inputs from (block or window) means: logs of the mean values."""
    t["ln_delta"] = np.log(t["delta"])
    t["ln_m"] = np.log(t["mass_kg"])
    t["ln_theta"] = np.log(t["theta"])
    t["ln_ff"] = np.log(t["ff_kgh"])
    return t


def make_blocks(s: pd.DataFrame, cfg: PrepConfig | None = None) -> pd.DataFrame:
    """Means over consecutive blocks of cfg.block_s samples inside each segment.

    A block is kept only if it is complete in time and all of its samples pass the
    sample filters."""
    cfg = cfg or PrepConfig()
    seg = s["seg"].to_numpy()
    start = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    pos = np.arange(len(s)) - start[seg]
    key = seg.astype(np.int64) * 1_000_000 + pos // cfg.block_s
    g = s[BLOCK_MEANS].groupby(key, sort=True)
    b = g.mean()
    b["n"] = g.size()
    b["n_keep"] = s["keep"].groupby(key, sort=True).sum()
    first = s.groupby(key, sort=True).head(1)
    for c in ("flight_id", "flight_code", "seg", "tail"):
        if c in s:
            b[c] = first[c].to_numpy()
    if "anti_ice" in s:
        b["anti_ice"] = s["anti_ice"].groupby(key, sort=True).max().to_numpy()
    b = b[(b.n == cfg.block_s) & (b.n_keep == cfg.block_s)].reset_index(drop=True)
    b = b[np.isfinite(b[BLOCK_MEANS].to_numpy(dtype=float)).all(axis=1)]
    b = add_log_inputs(b.drop(columns=["n", "n_keep"]).reset_index(drop=True))
    vs_fpm = b["dhg"] / FT * 60.0
    b["phase"] = np.where(
        vs_fpm > cfg.level_fpm,
        "climb",
        np.where(vs_fpm < -cfg.level_fpm, "descent", "level"),
    )
    b["cruise"] = (b["alt_ft"] > cfg.cruise_alt_ft) & (b["phase"] == "level")
    return b


def split_flights(flight_start: pd.Series, flights, cfg: PrepConfig | None = None):
    """Chronological split by flight start time: first 80 % train, next 10 % val."""
    cfg = cfg or PrepConfig()
    start = flight_start[flight_start.index.isin(flights)].sort_values(kind="stable")
    n = len(start)
    n_tr, n_va = int(cfg.split[0] * n), int(cfg.split[1] * n)
    split = pd.Series("test", index=start.index)
    split.iloc[:n_tr] = "train"
    split.iloc[n_tr : n_tr + n_va] = "val"
    return split


# ---------------------------------------------------------------- steady windows
WINDOW_MEANS = [
    "alt_ft",
    "mach",
    "sat_k",
    "delta",
    "theta",
    "dT_isa",
    "tas_kt",
    "gs_kt",
    "mass_kg",
    "ff_kgh",
    "sin_gamma",
    "acc_in",
]


def steady_windows(s: pd.DataFrame, cfg: PrepConfig | None = None) -> pd.DataFrame:
    """Steady cruise windows of cfg.win_s seconds.

    A sample is a candidate if it is above win_min_alt_ft and the altitude 60 s later
    differs by less than win_alt_change_ft. Runs of candidates (same flight, 1 s steps)
    are cut into consecutive windows from the start of the run. A window is kept if the
    Mach SD is below win_mach_sd, the altitude range below win_alt_range_ft, at least
    win_roll_share of the roll values are below win_roll_deg (if roll is present), the
    engine fuel-flow imbalance of the window means is below win_ff_imbalance (if engine
    fuel flows are present), all samples pass the configuration filters (clean, and
    the speed-brake, gear and anti-ice filters as in sample_filter) and the means are
    finite. The window carries `anti_ice` (True if on in any sample) if present.
    """
    cfg = cfg or PrepConfig()
    fc = s["flight_code"].to_numpy()
    ts = s["time_s"].to_numpy()
    alt = s["alt_ft"].to_numpy()
    n, lag, w = len(s), cfg.win_lookahead_s, cfg.win_s
    ahead = np.full(n, np.nan)
    ok = (fc[lag:] == fc[:-lag]) & (ts[lag:] - ts[:-lag] == lag)
    ahead[:-lag] = np.where(ok, alt[lag:] - alt[:-lag], np.nan)
    cond = (alt > cfg.win_min_alt_ft) & (np.abs(ahead) < cfg.win_alt_change_ft)
    brk = np.ones(n, dtype=bool)
    brk[1:] = (fc[1:] != fc[:-1]) | (np.diff(ts) != 1) | ~cond[1:]
    brk |= ~cond
    run = np.cumsum(brk)
    idx = np.flatnonzero(cond)
    runs = (
        pd.DataFrame({"run": run[idx], "i": idx}).groupby("run").i.agg(["min", "count"])
    )
    runs = runs[runs["count"] >= w]
    # without a run the table is empty, with the same columns
    starts = np.concatenate(
        [np.zeros(0, int)]
        + [
            a + w * np.arange(c // w)
            for a, c in zip(runs["min"], runs["count"], strict=True)
        ]
    )
    ii = starts[:, None] + np.arange(w)[None, :]

    def win(col):
        return s[col].to_numpy(dtype=float)[ii]

    m = win("mach")
    a = win("alt_ft")
    keep = (m.std(axis=1, ddof=1) < cfg.win_mach_sd) & (
        a.max(axis=1) - a.min(axis=1) < cfg.win_alt_range_ft
    )
    if "roll_deg" in s:
        keep &= (np.abs(win("roll_deg")) < cfg.win_roll_deg).mean(
            axis=1
        ) >= cfg.win_roll_share
    if "ff1_kgh" in s and "ff2_kgh" in s:
        f1, f2 = win("ff1_kgh").mean(axis=1), win("ff2_kgh").mean(axis=1)
        keep &= np.abs(f1 - f2) / (0.5 * (f1 + f2)) < cfg.win_ff_imbalance
    ok = np.logical_and.reduce([k for _, k in config_steps(s, cfg)])
    keep &= ok[ii].all(axis=1)
    keep &= (win("ff_kgh") > 0).all(axis=1)  # as the sample filter of the blocks
    t = pd.DataFrame({c: win(c).mean(axis=1) for c in WINDOW_MEANS})
    t.insert(0, "time_s", ts[starts])
    t.insert(0, "flight_id", s["flight_id"].to_numpy()[starts])
    if "tail" in s:
        t["tail"] = s["tail"].to_numpy()[starts]
    if "anti_ice" in s:
        t["anti_ice"] = s["anti_ice"].to_numpy(dtype=bool)[ii].any(axis=1)
    keep &= np.isfinite(t[WINDOW_MEANS].to_numpy(dtype=float)).all(axis=1)
    t = add_log_inputs(t[keep].reset_index(drop=True))
    t["wd_t"] = t["mass_kg"] / t["delta"] / 1000.0
    # finite model inputs and target, as for the blocks
    fin = np.isfinite(t[[*FEATURES, "ln_ff"]].to_numpy(dtype=float)).all(axis=1)
    return t[fin].reset_index(drop=True)


# ---------------------------------------------------------------- all steps
def prepare(df: pd.DataFrame, cfg: PrepConfig | None = None):
    """Samples in the schema -> (blocks, windows, summary).

    blocks and windows hold flight identifiers for the local split and the trip-fuel
    test; summary holds counts only.
    """
    cfg = cfg or PrepConfig()
    check_config(cfg)
    require_valid(df)
    s, counts, flight_start = derive(df, cfg)
    blocks = make_blocks(s, cfg)
    split = split_flights(flight_start, blocks["flight_id"].unique(), cfg)
    blocks["split"] = blocks["flight_id"].map(split).to_numpy()
    windows = steady_windows(s, cfg)
    windows["split"] = windows["flight_id"].map(split).to_numpy()
    windows = windows[windows["split"].notna()].reset_index(drop=True)
    blocks = blocks.drop(columns=["flight_code"])
    summary = {
        "config": asdict(cfg),
        "flights": int(s["flight_id"].nunique()),
        "filter_counts": counts,
        "blocks": {
            sp: {
                "blocks": int((blocks.split == sp).sum()),
                "flights": int(blocks.flight_id[blocks.split == sp].nunique()),
                **{
                    ph: int(((blocks.split == sp) & (blocks.phase == ph)).sum())
                    for ph in ("climb", "level", "descent")
                },
                "cruise": int(((blocks.split == sp) & blocks.cruise).sum()),
            }
            for sp in ("train", "val", "test")
        },
        "windows": {
            sp: int((windows.split == sp).sum()) for sp in ("train", "val", "test")
        },
    }
    return blocks, windows, summary
