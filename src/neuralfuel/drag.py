"""Drag slope of the aerox drag model for the drag-rise constraint P7.

In steady level flight F = TSFC x D. At a fixed pressure altitude and mass (fixed
W/delta), the lift coefficient is CL = 2 m g / (gamma p0 delta M^2 S), so CL varies as
1 / M^2, and

  g(M, CL, delta) = d ln D / dM = d/dM [ln CD(M, CL(M), Re(M, delta)) + 2 ln M]

is the Mach slope of the drag. TSFC does not decrease with Mach, so
d ln F / dM >= g; P7 (neuralfuel.losses) penalizes the violation. g comes from the
full aerox drag model (aerox.models.drag.build_drag_model, with wave drag) at the ISA
temperature; it is a central difference of ln D with step DMACH. `Unavailable` tells
that a type has no such drag model.

The training does not call aerox at every step: `build_table` evaluates g once on a
regular grid in Mach, CL and ln delta, and `lookup` interpolates it (trilinear, torch
or numpy; a value outside the grid is clamped to its edge). The grid and the aerox
version are stored in the model file (`table_spec`), with the SHA-256 of the table
values, so that the validation rebuilds the same table and can tell whether aerox
gives other values.

CL is that of the wing area of the aircraft settings (aircraft.wing_area_m2). aerox
refers its drag coefficient to its own wing area; the table converts CL to it.
"""

import hashlib
import importlib.metadata
import math

import numpy as np
import torch

from . import atmos

DMACH = 1e-4  # step of the central difference of ln D in Mach
# grid steps of the table
TABLE_DMACH = 0.005
TABLE_DCL = 0.01
TABLE_CL_MIN = 0.05
TABLE_N_DELTA = 12


def aerox_version() -> str:
    try:
        return importlib.metadata.version("aerox")
    except importlib.metadata.PackageNotFoundError:
        return ""


class Unavailable(ValueError):
    """P7 is not available for the aircraft type (the message gives the reason)."""


def aerox_drag_model(ac_type: str):
    """(aerox drag model with wave drag, aerox wing area in m²) of a type; Unavailable
    if aerox has no such type, no wave-drag model or no wing area for it."""
    from aerox import get_available_aircraft, load_aircraft
    from aerox.models.drag import build_drag_model

    code = ac_type.upper()
    if code not in get_available_aircraft():
        raise Unavailable(f"aerox has no type '{ac_type}'")
    rec = load_aircraft(code)
    try:
        model = build_drag_model(rec)
    except Exception as e:  # aerox raises several kinds for missing geometry
        raise Unavailable(f"aerox has no drag model for '{ac_type}' ({e})") from e
    if not hasattr(model, "wave_drag"):
        raise Unavailable(f"the aerox drag model of '{ac_type}' has no wave drag")
    wing = getattr(rec, "wing", None)
    area = getattr(wing, "area", None)
    if area is None or not math.isfinite(float(area)):
        raise Unavailable(f"aerox has no wing area for '{ac_type}'")
    return model, float(area)


def table_spec(ac, mach_min: float, cl_max: float) -> dict:
    """Grid of the table: Mach from mach_min to the MMO, CL from TABLE_CL_MIN to
    cl_max, ln delta over the altitude range of the type ([low, high, points])."""
    n_m = max(2, math.ceil((ac.mmo - mach_min) / TABLE_DMACH - 1e-9) + 1)
    n_cl = max(2, math.ceil((cl_max - TABLE_CL_MIN) / TABLE_DCL - 1e-9) + 1)
    lnd = [float(np.log(atmos.isa_delta(a))) for a in (ac.alt_ft[1], ac.alt_ft[0])]
    return {
        "type": ac.type,
        "wing_area_m2": float(ac.wing_area_m2),
        "mach": [float(mach_min), float(ac.mmo), n_m],
        "cl": [TABLE_CL_MIN, float(cl_max), n_cl],
        "ln_delta": [lnd[0], lnd[1], TABLE_N_DELTA],
        "dmach": DMACH,
    }


def axes(spec):
    return [np.linspace(lo, hi, int(n)) for lo, hi, n in (spec[k] for k in AXES)]


AXES = ("mach", "cl", "ln_delta")


def drag_slope(model, aerox_area, wing_area, mach, cl, ln_delta, dmach=DMACH):
    """g = d ln D / dM at fixed W/delta and pressure altitude (numpy, broadcast)."""
    mach, cl, ln_delta = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (mach, cl, ln_delta))
    )
    h_m = atmos.alt_from_delta(np.exp(ln_delta)) * atmos.FT
    cl_aerox = cl * wing_area / aerox_area

    def ln_d(m):
        cd = model.cd(cl=cl_aerox * (mach / m) ** 2, mach=m, h=h_m)
        return np.log(cd) + 2 * np.log(m)

    return (ln_d(mach + dmach) - ln_d(mach - dmach)) / (2 * dmach)


def build_table(spec) -> np.ndarray:
    """g on the grid of spec, shape (n_mach, n_cl, n_delta), float64."""
    model, area = aerox_drag_model(spec["type"])
    m, c, d = np.meshgrid(*axes(spec), indexing="ij")
    return drag_slope(model, area, spec["wing_area_m2"], m, c, d, spec["dmach"])


def table_hash(table) -> str:
    """SHA-256 of the table values (float64, C order, rounded to 1e-9)."""
    a = np.ascontiguousarray(np.round(np.asarray(table, dtype=np.float64), 9))
    return hashlib.sha256(a.tobytes()).hexdigest()


def lookup(table, spec, mach, cl, ln_delta):
    """Trilinear interpolation of the table at the states (torch tensors or numpy
    arrays, same shape); values outside the grid are clamped to its edges."""
    is_t = torch.is_tensor(mach)
    if is_t and not torch.is_tensor(table):
        table = torch.as_tensor(table, dtype=mach.dtype, device=mach.device)
    idx, frac = [], []
    for v, k in zip((mach, cl, ln_delta), AXES, strict=True):
        lo, hi, n = spec[k]
        n = int(n)
        u = (v - lo) / (hi - lo) * (n - 1)
        if is_t:
            u = u.clamp(0, n - 1)
            i0 = torch.floor(u).clamp(max=n - 2)
            idx.append(i0.long())
        else:
            u = np.clip(u, 0, n - 1)
            i0 = np.minimum(np.floor(u), n - 2)
            idx.append(i0.astype(int))
        frac.append(u - i0)
    i, j, k = idx
    a, b, c = frac
    out = 0
    for di in (0, 1):
        wa = a if di else 1 - a
        for dj in (0, 1):
            wb = b if dj else 1 - b
            for dk in (0, 1):
                wc = c if dk else 1 - c
                out = out + wa * wb * wc * table[i + di, j + dj, k + dk]
    return out


def level_cl(mach, ln_delta, ln_m, wing_area_m2):
    """Lift coefficient of level flight (torch or numpy)."""
    lib = torch if torch.is_tensor(mach) else np
    return (
        2
        * lib.exp(ln_m)
        * atmos.G0
        / (atmos.GAMMA * atmos.P0 * lib.exp(ln_delta) * mach**2 * wing_area_m2)
    )
