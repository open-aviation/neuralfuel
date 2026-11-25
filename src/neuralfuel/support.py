"""Valid envelope: where a model has training data.

`data_envelope` describes the training data of a model, `in_envelope` tells for model
inputs whether they lie inside that description. The envelope is a JSON-ready dict
that is stored in the weights file, the run record, the validation report and the
ONNX metadata (entry 'neuralfuel_envelope'), so a tool such as a trajectory optimizer
can read where the model is valid.

Contents:
  phases   for 'all', 'climb', 'level' and 'descent' blocks: the number of blocks and
           the range from the 1 % to the 99 % quantile of Mach, pressure altitude (ft),
           mass (kg), W/delta (t), ISA deviation (K), sin_gamma and acc_in
  cruise   from the steady cruise windows: the number of windows, the largest W/delta
           (t) and the largest lift coefficient (only with the wing area of the
           aircraft settings)
  phase_rule
           the rule that gives the phase of a state: climb above +level_fpm, descent
           below -level_fpm geometric vertical speed, else level; 'cruise' is level
           flight above cruise_alt_ft

`union_envelope` joins envelopes (per phase the lowest minimum and the highest
maximum); a model that continued from shared weights gets the union of the shared
envelope and that of its local data. The union is a box per phase: it can span gaps
between the joined envelopes. Its block and window counts are sums: over rounds of
aggregation they count the training exposures of the data, not unique observations.

A state is inside the envelope if every variable lies inside the range of its phase
and, for level flight above cruise_alt_ft, W/delta and the lift coefficient are not
larger than the cruise maxima. The comparison allows a relative tolerance of BOX_RTOL,
so that a value at a bound stays inside after the log and exp transforms of the model
inputs (for example a constant mass).

The lift coefficient is that of level flight, lift = weight:
CL = 2 m g / (gamma p0 delta M^2 S).
"""

import copy
import math

import numpy as np
import pandas as pd

from . import atmos
from .atmos import FT, G0, GAMMA, P0, T0
from .prep import FEATURES
from .privacy import NUM, STR

ENVELOPE_FORMAT = "neuralfuel-envelope-1"
ENVELOPE_VARS = ("mach", "alt_ft", "mass_kg", "wd_t", "dT_isa_k", "sin_gamma", "acc_in")
PHASES = ("all", "climb", "level", "descent")
BOX_RTOL = 1e-9  # tolerance of the box bounds, relative to max(1, |bound|)
# the keys of an envelope (format of neuralfuel.privacy.allowed)
ENVELOPE = {
    "format": STR,
    "quantiles": NUM,
    "phase_rule": {"level_fpm": NUM, "cruise_alt_ft": NUM},
    "phases": {p: {"blocks": NUM, **dict.fromkeys(ENVELOPE_VARS, NUM)} for p in PHASES},
    "cruise": {"windows": NUM, "max_wd_t": NUM, "max_cl": NUM},
    "wing_area_m2": NUM,
}


def lift_coefficient(mass_kg, mach, delta, wing_area_m2):
    """Lift coefficient of level flight (lift = weight)."""
    m, mach, d = (np.asarray(v, dtype=float) for v in (mass_kg, mach, delta))
    return 2 * m * G0 / (GAMMA * P0 * d * mach**2 * wing_area_m2)


def _states(t: pd.DataFrame) -> dict:
    """Envelope variables of blocks or windows (columns of neuralfuel.prep)."""
    return {
        "mach": t["mach"].to_numpy(dtype=float),
        "alt_ft": t["alt_ft"].to_numpy(dtype=float),
        "mass_kg": t["mass_kg"].to_numpy(dtype=float),
        "wd_t": t["mass_kg"].to_numpy(dtype=float)
        / t["delta"].to_numpy(dtype=float)
        / 1000.0,
        "dT_isa_k": t["dT_isa"].to_numpy(dtype=float),
        "sin_gamma": t["sin_gamma"].to_numpy(dtype=float),
        "acc_in": t["acc_in"].to_numpy(dtype=float),
    }


def data_envelope(
    blocks: pd.DataFrame,
    windows: pd.DataFrame,
    wing_area_m2: float = math.nan,
    quantiles=(0.01, 0.99),
    level_fpm: float = 300.0,
    cruise_alt_ft: float = 25000.0,
) -> dict:
    """Envelope (dict) of the given blocks and steady windows, normally the training
    split. level_fpm and cruise_alt_ft are the phase settings of the preparation."""
    v = _states(blocks)
    ph = blocks["phase"].to_numpy()
    out = {
        "format": ENVELOPE_FORMAT,
        "quantiles": list(quantiles),
        "phase_rule": {"level_fpm": level_fpm, "cruise_alt_ft": cruise_alt_ft},
        "phases": {},
    }
    for p in PHASES:
        sel = np.ones(len(ph), bool) if p == "all" else ph == p
        r = {"blocks": int(sel.sum())}
        for k in ENVELOPE_VARS:
            r[k] = (
                [float(x) for x in np.quantile(v[k][sel], quantiles)]
                if sel.any()
                else None
            )
        out["phases"][p] = r
    c = {"windows": len(windows), "max_wd_t": None, "max_cl": None}
    if len(windows):
        w = _states(windows)
        c["max_wd_t"] = float(w["wd_t"].max())
        if math.isfinite(wing_area_m2):
            cl = lift_coefficient(
                w["mass_kg"], w["mach"], windows["delta"].to_numpy(), wing_area_m2
            )
            c["max_cl"] = float(cl.max())
    out["cruise"] = c
    out["wing_area_m2"] = wing_area_m2 if math.isfinite(wing_area_m2) else None
    return out


def _has_cruise(envelope) -> bool:
    """True if the envelope holds level blocks above the cruise altitude, so that its
    cruise limits apply to some states."""
    lv = envelope["phases"].get("level")
    if not lv or not lv.get("blocks") or lv.get("alt_ft") is None:
        return False
    return lv["alt_ft"][1] > envelope["phase_rule"]["cruise_alt_ft"]


def union_envelope(*envelopes) -> dict | None:
    """Envelope that covers all given envelopes (None entries are skipped): per phase
    the lowest minimum and the highest maximum of every variable and the sum of the
    block counts; for cruise the largest W/delta and lift coefficient and the sum of
    the window counts. A cruise limit None means no limit (in_envelope): if an
    envelope with level blocks above the cruise altitude has no limit, the union has
    none either. All envelopes must have the same format, quantiles, phase rule and
    wing area."""
    envs = [e for e in envelopes if e is not None]
    if not envs:
        return None
    for e in envs[1:]:
        for k in ("format", "quantiles", "phase_rule", "wing_area_m2"):
            if e.get(k) != envs[0].get(k):
                raise ValueError(
                    f"envelopes with different {k.replace('_', ' ')} cannot be joined"
                )
    no_limit = {
        k: any(_has_cruise(e) and e["cruise"].get(k) is None for e in envs)
        for k in ("max_wd_t", "max_cl")
    }
    out = copy.deepcopy(envs[0])
    for e in envs[1:]:
        for p in PHASES:
            a, b = out["phases"].get(p), e["phases"].get(p)
            if b is None:
                continue
            if a is None:
                out["phases"][p] = copy.deepcopy(b)
                continue
            a["blocks"] = int(a.get("blocks", 0)) + int(b.get("blocks", 0))
            for k in ENVELOPE_VARS:
                if b.get(k) is None:
                    continue
                if a.get(k) is None:
                    a[k] = list(b[k])
                else:
                    a[k] = [min(a[k][0], b[k][0]), max(a[k][1], b[k][1])]
        ca, cb = out["cruise"], e["cruise"]
        ca["windows"] = int(ca.get("windows", 0)) + int(cb.get("windows", 0))
        for k in ("max_wd_t", "max_cl"):
            v = [x for x in (ca.get(k), cb.get(k)) if x is not None]
            ca[k] = max(v) if v else None
    for k, free in no_limit.items():
        if free:
            out["cruise"][k] = None
    return out


def input_states(x) -> dict:
    """Envelope variables and the phase of model inputs x: an (n, 6) array in the order
    of FEATURES, or a DataFrame with these columns."""
    if isinstance(x, pd.DataFrame):
        x = x[FEATURES].to_numpy(dtype=float)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    mach, lnd, lnm, sg, acc, lnth = x.T
    delta, sat = np.exp(lnd), T0 * np.exp(lnth)
    alt = atmos.alt_from_delta(delta)
    mass = np.exp(lnm)
    vs_fpm = sg * mach * atmos.sound_speed(sat) / FT * 60.0
    return {
        "mach": mach,
        "alt_ft": alt,
        "mass_kg": mass,
        "wd_t": mass / delta / 1000.0,
        "dT_isa_k": sat - atmos.isa_temperature(alt),
        "sin_gamma": sg,
        "acc_in": acc,
        "delta": delta,
        "vs_fpm": vs_fpm,
    }


def phase_of(vs_fpm, level_fpm=300.0):
    """'climb', 'level' or 'descent' from the geometric vertical speed (ft/min)."""
    vs = np.asarray(vs_fpm, dtype=float)
    return np.where(
        vs > level_fpm, "climb", np.where(vs < -level_fpm, "descent", "level")
    )


def in_envelope(envelope: dict, x) -> np.ndarray:
    """True for the model inputs x (see input_states) that lie inside the envelope."""
    if envelope is None:
        raise ValueError("no envelope: the model file has no valid envelope")
    s = input_states(x)
    rule = envelope["phase_rule"]
    ph = phase_of(s["vs_fpm"], rule["level_fpm"])
    ok = np.zeros(len(ph), bool)
    for p in ("climb", "level", "descent"):
        box = envelope["phases"].get(p)
        sel = ph == p
        if box is None or box["blocks"] == 0 or not sel.any():
            continue
        inside = np.ones(int(sel.sum()), bool)
        for k in ENVELOPE_VARS:
            lo, hi = box[k]
            tol = BOX_RTOL * max(1.0, abs(lo), abs(hi))
            inside &= (s[k][sel] >= lo - tol) & (s[k][sel] <= hi + tol)
        ok[sel] = inside
    cr = envelope["cruise"]
    cruise = (ph == "level") & (s["alt_ft"] > rule["cruise_alt_ft"])
    if cr["max_wd_t"] is not None:
        ok &= ~cruise | (s["wd_t"] <= cr["max_wd_t"] * (1 + BOX_RTOL))
    if cr["max_cl"] is not None and envelope.get("wing_area_m2"):
        cl = lift_coefficient(
            s["mass_kg"], s["mach"], s["delta"], envelope["wing_area_m2"]
        )
        ok &= ~cruise | (cl <= cr["max_cl"] * (1 + BOX_RTOL))
    return ok


def random_inputs(envelope: dict, n: int, seed: int = 0, phase: str = "all"):
    """n model inputs (n, 6), uniform in the ranges of `phase` and inside the envelope
    (rejection sampling)."""
    box = envelope["phases"][phase]
    rng = np.random.default_rng(seed)
    keys = ("mach", "alt_ft", "mass_kg", "dT_isa_k", "sin_gamma", "acc_in")
    lo = np.array([box[k][0] for k in keys])
    hi = np.array([box[k][1] for k in keys])
    out, have = [], 0
    for _ in range(1000):
        m, alt, w, dt, sg, acc = (lo + rng.random((4 * n, 6)) * (hi - lo)).T
        sat = atmos.isa_temperature(alt) + dt
        x = np.column_stack(
            [m, np.log(atmos.isa_delta(alt)), np.log(w), sg, acc, np.log(sat / T0)]
        )
        x = x[in_envelope(envelope, x)]
        out.append(x)
        have += len(x)
        if have >= n:
            return np.concatenate(out)[:n]
    raise ValueError(f"could not draw {n} states inside the envelope")
