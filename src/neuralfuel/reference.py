"""Comparison of a model with reference fuel models, for the review of contributions.

A reference is a plain callable

    ref(mach, alt_ft, mass_kg, dT_isa_k, vs_fpm) -> fuel flow in kg/h (all engines)

with numpy arrays of the same shape (pressure altitude, ISA deviation of the static
temperature at that pressure altitude, geometric vertical speed; still air, no
acceleration). Three are included:

  openap_reference(type)   OpenAP (optional dependency group 'reference')
  bada_reference(name)     BADA 4 through the OpenAP add-on; needs a licensed local
                           BADA 4 table folder, given by the environment variable
                           NEURALFUEL_BADA or by `bada_path` in the [reference] table
                           of local.toml. No BADA data ships with the package. The
                           comparison then reports ratios only, never an absolute BADA
                           value.
  model_reference(model)   another neuralfuel model, for example the previous shared
                           model

OpenAP and its BADA 4 add-on take a geometric altitude and a temperature shift dT of
the whole atmosphere profile: at dT not 0 a pressure altitude is not their altitude.
`openap_conditions` gives the altitude and the shift at which their atmosphere has
the pressure and the static temperature of the model input, and the true airspeed
that gives the same Mach. OpenAP limits the shift to -25 to +15 K. Where a larger
shift is needed (warm days below the tropopause), the pressure, the Mach and the
flight-path angle stay the same and the temperature is the nearest that OpenAP can
represent. The OpenAP fuel flow depends on the atmosphere only through the drag,
which depends on pressure and Mach, so it is not affected; BADA 4 also uses theta.

`compare` evaluates both on a fixed grid of states inside the valid envelope of the
model (`compare_states`): steady cruise (Mach x altitude x mass), climb and descent at
the middle vertical speed of their phase. It reports the fuel-flow ratio model /
reference per phase (median, 5 % and 95 % quantiles), the cruise Mach sensitivity
s = d ln F / dM of both (the reference by a central difference of +-0.005 Mach) with
the median of the pointwise ratios model / reference, and the Mach of maximum
specific air range M / F of both at a few altitudes and masses. Each best-range Mach
has three flags: at the upper bound of the search (the MMO), at the lower bound of
the search (MMO - SR_SPAN), and outside the valid envelope of the model (the state of
level flight at that Mach, altitude, mass and ISA deviation); a flagged Mach is an
extrapolation of the model or a limit of the search, not an optimum inside the
envelope.
"""

import copy
import os
import tomllib
from pathlib import Path

import numpy as np
import torch

from . import atmos
from .atmos import FT, KT, T0
from .model import FuelFlowNet

DMACH = 0.005  # half step of the central difference of the reference sensitivity
SR_SPAN, SR_STEP = 0.22, 0.002  # search of the best-range Mach below the MMO


# ---------------------------------------------------------------- references
OPENAP_DT = (-25.0, 15.0)  # range of the temperature shift in the OpenAP atmosphere


def openap_conditions(aero, mach, alt_ft, dT_isa_k, vs_fpm, iters=60):
    """(alt_ft, dT, tas_kt, vs_fpm) for an OpenAP-style model with the atmosphere
    `aero` (openap.aero.Aero) such that it sees the pressure and the static
    temperature of the model input (pressure altitude alt_ft, ISA deviation dT_isa_k),
    the same Mach and the same flight-path angle. The shift is limited to OPENAP_DT;
    the pressure is matched in every case."""
    m, a, t, v = (
        np.asarray(x, dtype=float)
        for x in np.broadcast_arrays(mach, alt_ft, dT_isa_k, vs_fpm)
    )
    p = atmos.P0 * atmos.isa_delta(a)
    sat = atmos.isa_temperature(a) + t

    def height(dT):  # geometric height (m) with the pressure p at the shift dT
        lo, hi = np.full(p.shape, -3000.0), np.full(p.shape, 30000.0)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            high = aero.pressure(mid, dT=dT) > p
            lo, hi = np.where(high, mid, lo), np.where(high, hi, mid)
        return 0.5 * (lo + hi)

    # the temperature at the pressure p increases with the shift: bisection
    lo, hi = np.full(p.shape, OPENAP_DT[0]), np.full(p.shape, OPENAP_DT[1])
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        warm = aero.temperature(height(mid), dT=mid) > sat
        lo, hi = np.where(warm, lo, mid), np.where(warm, mid, hi)
    dT = 0.5 * (lo + hi)
    h = height(dT)
    tas_ms = m * aero.vsound(h, dT=dT)
    sg = v * FT / 60.0 / (m * atmos.sound_speed(sat))
    # OpenAP takes the path angle as atan(vs / tas); the model input is sin_gamma
    vs = np.tan(np.arcsin(sg)) * tas_ms / FT * 60.0
    return h / FT, dT, tas_ms / KT, vs


def _enroute(ff, mach, alt_ft, mass_kg, dT_isa_k, vs_fpm):
    """Fuel flow (kg/h) of an OpenAP-style FuelFlow object at the model state."""
    h, dT, tas, vs = openap_conditions(ff.aero, mach, alt_ft, dT_isa_k, vs_fpm)
    v = ff.enroute(
        np.asarray(mass_kg, dtype=float) * np.ones_like(h), tas, h, vs, dT=dT
    )
    return 3600.0 * np.asarray(v, dtype=float).reshape(np.shape(mach))


def openap_reference(ac_type: str, engine=None):
    """OpenAP fuel flow of `ac_type` (for example 'A320')."""
    try:
        from openap import FuelFlow
    except ImportError as e:
        raise ImportError(
            "OpenAP is not installed; run 'uv sync --group reference'"
        ) from e
    ff = FuelFlow(ac_type.upper(), eng=engine)

    def ref(mach, alt_ft, mass_kg, dT_isa_k, vs_fpm):
        return _enroute(ff, mach, alt_ft, mass_kg, dT_isa_k, vs_fpm)

    return ref


def bada_path():
    """Path of the local BADA 4 table folder (NEURALFUEL_BADA or local.toml)."""
    p = os.environ.get("NEURALFUEL_BADA")
    if not p:
        cfg = Path(os.environ.get("NEURALFUEL_CONFIG", "local.toml"))
        if cfg.exists():
            with open(cfg, "rb") as fh:
                p = tomllib.load(fh).get("reference", {}).get("bada_path")
    if not p or not Path(p).exists():
        raise FileNotFoundError(
            "no BADA 4 folder: set NEURALFUEL_BADA or [reference] bada_path in "
            "local.toml"
        )
    return p


def bada_reference(name: str, path=None):
    """BADA 4 fuel flow of the BADA model `name`, read from the local folder."""
    try:
        from openap.addon import bada4
    except ImportError as e:
        raise ImportError(
            "OpenAP is not installed; run 'uv sync --group reference'"
        ) from e
    ff = bada4.FuelFlow(name, path or bada_path())

    def ref(mach, alt_ft, mass_kg, dT_isa_k, vs_fpm):
        return _enroute(ff, mach, alt_ft, mass_kg, dT_isa_k, vs_fpm)

    return ref


def model_reference(model: FuelFlowNet):
    """Another neuralfuel model as a reference."""
    m = copy.deepcopy(model).cpu().double().eval()

    def ref(mach, alt_ft, mass_kg, dT_isa_k, vs_fpm):
        x = model_inputs(mach, alt_ft, mass_kg, dT_isa_k, vs_fpm)
        with torch.no_grad():
            f = m(torch.as_tensor(x)).numpy()
        return np.exp(f).reshape(np.shape(mach))

    return ref


# ---------------------------------------------------------------- states
def model_inputs(mach, alt_ft, mass_kg, dT_isa_k, vs_fpm):
    """(n, 6) model inputs of still-air flight without acceleration."""
    m, a, w, t, v = (
        np.asarray(x, dtype=float).ravel()
        for x in np.broadcast_arrays(mach, alt_ft, mass_kg, dT_isa_k, vs_fpm)
    )
    sat = atmos.isa_temperature(a) + t
    sg = v * FT / 60.0 / (m * atmos.sound_speed(sat))
    return np.column_stack(
        [
            m,
            np.log(atmos.isa_delta(a)),
            np.log(w),
            sg,
            np.zeros(len(m)),
            np.log(sat / T0),
        ]
    )


def _mid(r):
    return 0.5 * (r[0] + r[1])


def compare_states(envelope: dict, ac) -> dict:
    """Grid states per phase ('cruise', 'climb', 'descent') inside the envelope: dict
    of arrays mach, alt_ft, mass_kg, dT_isa_k, vs_fpm."""
    from .support import in_envelope

    rule = envelope["phase_rule"]
    out = {}
    for phase, box_name in (
        ("cruise", "level"),
        ("climb", "climb"),
        ("descent", "descent"),
    ):
        box = envelope["phases"].get(box_name)
        if box is None or not box.get("blocks"):
            continue
        m_lo, m_hi = box["mach"][0], min(box["mach"][1], ac.mmo)
        a_lo, a_hi = box["alt_ft"]
        if phase == "cruise":
            a_lo = max(a_lo, rule["cruise_alt_ft"] + 1.0)
            mach = np.arange(np.ceil(m_lo * 100) / 100, m_hi + 1e-9, 0.01)
            alt = np.arange(np.ceil(a_lo / 1000) * 1000, a_hi + 1e-6, 1000.0)
            sg = float(np.clip(0.0, *box["sin_gamma"]))  # level: 0 if inside
        else:
            mach = np.arange(np.ceil(m_lo * 50) / 50, m_hi + 1e-9, 0.02)
            alt = np.arange(np.ceil(a_lo / 2000) * 2000, a_hi + 1e-6, 2000.0)
            sg = _mid(box["sin_gamma"])
        mass = np.linspace(box["mass_kg"][0], box["mass_kg"][1], 5)
        dT = _mid(box["dT_isa_k"])
        M, A, W = (v.ravel() for v in np.meshgrid(mach, alt, mass, indexing="ij"))
        if len(M) == 0:
            continue
        sat = atmos.isa_temperature(A) + dT
        vs = sg * M * atmos.sound_speed(sat) / FT * 60.0
        st = {"mach": M, "alt_ft": A, "mass_kg": W, "dT_isa_k": np.full(len(M), dT)}
        st["vs_fpm"] = vs
        x = model_inputs(**st)
        ok = in_envelope(envelope, x)
        if ok.any():
            out[phase] = {k: v[ok] for k, v in st.items()}
    return out


def _model_ff(m64, st):
    x = model_inputs(**st)
    with torch.no_grad():
        return np.exp(m64(torch.as_tensor(x)).numpy())


def _model_s(m64, st):
    x = torch.tensor(model_inputs(**st), requires_grad=True)
    return torch.autograd.grad(m64(x).sum(), x)[0][:, 0].numpy()


def _quant(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0, "median": None, "p05": None, "p95": None}
    q = np.quantile(v, [0.5, 0.05, 0.95])
    return {"n": len(v), "median": float(q[0]), "p05": float(q[1]), "p95": float(q[2])}


def best_range_mach(fn, ac, alt_ft, mass_kg, dT):
    """(Mach of maximum M / F, True if it is the upper bound of the search, True if it
    is the lower bound of the search)."""
    mg = np.arange(ac.mmo - SR_SPAN, ac.mmo + 1e-9, SR_STEP)
    f = fn(
        mg,
        np.full(len(mg), alt_ft),
        np.full(len(mg), mass_kg),
        np.full(len(mg), dT),
        np.zeros(len(mg)),
    )
    i = int(np.argmax(mg / f))
    return float(mg[i]), i == len(mg) - 1, i == 0


def compare(model: FuelFlowNet, ref, ratios_only=False) -> dict:
    """Comparison of model with the reference callable `ref` (see the module
    docstring). ratios_only=True leaves out every absolute value of the reference
    (for a licensed reference such as BADA): the sensitivities and the best-range
    Mach are then given as ratios model / reference."""
    from .support import in_envelope

    if model.envelope is None or model.aircraft is None:
        raise ValueError("the model needs a valid envelope and aircraft settings")
    ac = model.aircraft
    m64 = copy.deepcopy(model).cpu().double().eval()
    states = compare_states(model.envelope, ac)
    out = {"ratios_only": ratios_only, "phases": {}}
    for phase, st in states.items():
        out["phases"][phase] = _quant(_model_ff(m64, st) / ref(**st))
    cr = states.get("cruise")
    if cr is not None:
        s_m = _model_s(m64, cr)
        up = {**cr, "mach": cr["mach"] + DMACH}
        dn = {**cr, "mach": cr["mach"] - DMACH}
        s_r = (np.log(ref(**up)) - np.log(ref(**dn))) / (2 * DMACH)
        ms = {
            "states": len(s_m),
            "ratio_model_over_reference": _quant(s_m / s_r),
            "ratio": "quantiles of the pointwise ratios s_model / s_reference",
        }
        if not ratios_only:
            ms["model"], ms["reference"] = _quant(s_m), _quant(s_r)
        out["cruise_mach_sensitivity"] = ms

        def fm(mach, alt_ft, mass_kg, dT_isa_k, vs_fpm):
            st = {"mach": mach, "alt_ft": alt_ft, "mass_kg": mass_kg}
            return _model_ff(m64, {**st, "dT_isa_k": dT_isa_k, "vs_fpm": vs_fpm})

        alts = np.unique(np.round(np.quantile(cr["alt_ft"], [0.1, 0.5, 0.9]), -3))
        masses = np.unique(np.round(np.quantile(cr["mass_kg"], [0.1, 0.5, 0.9]), -2))
        dT = float(cr["dT_isa_k"][0])
        rows = []
        for a in alts:
            for w in masses:
                bm, bm_ub, bm_lb = best_range_mach(fm, ac, a, w, dT)
                br, br_ub, br_lb = best_range_mach(ref, ac, a, w, dT)
                out_m, out_r = ~in_envelope(
                    model.envelope, model_inputs([bm, br], a, w, dT, 0.0)
                )
                r = {
                    "alt_ft": float(a),
                    "mass_kg": float(w),
                    "ratio": bm / br,
                    "model_at_upper_bound": bm_ub,
                    "reference_at_upper_bound": br_ub,
                    "model_at_lower_bound": bm_lb,
                    "reference_at_lower_bound": br_lb,
                    "model_outside_envelope": bool(out_m),
                    "reference_outside_envelope": bool(out_r),
                }
                if not ratios_only:
                    r["model"], r["reference"] = bm, br
                rows.append(r)
        out["best_range_mach"] = {"dT_isa_k": dT, "rows": rows}
    return out
