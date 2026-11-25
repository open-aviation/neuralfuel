"""Validation suite. Computes one JSON-ready report from local blocks and windows.

The report holds aggregate numbers only: no flight identifiers, dates, positions or
aircraft labels. The grids of the Mach and temperature tests come from the steady
windows of the validated data and from the aircraft settings (see ValidationConfig),
so the suite does not depend on one aircraft type.

REPORT is the format of the report (the keys that `validate` writes, for
neuralfuel.privacy.allowed).
"""

import copy
import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
import torch

from . import __version__, atmos
from . import drag as dragmod
from .aircraft import Aircraft, model_name, size_of
from .losses import (
    CONSTRAINT_SETTINGS,
    DRAG_RISE,
    LossConfig,
    constraint_settings,
    envelope,
    model_loss_config,
    p6_band,
)
from .model import FuelFlowNet, state_hash
from .prep import FEATURES, PrepConfig, derive
from .privacy import NUM, NUM_KEY, STR, clean_json
from .support import ENVELOPE, data_envelope, in_envelope

REPORT_FORMAT = "neuralfuel-validation-2"
SPLITS = ("train", "val", "test")
# drag-rise test (P7, only for a model with P7 settings): grid points in Mach (from
# the lower Mach of P7 to the MMO), altitude, mass and ISA deviation, and the
# tolerance of a violation (f_M < g - DRAG_RISE_TOL)
DRAG_RISE_GRID = (13, 20, 11, 8)
DRAG_RISE_TOL = 0.05


@dataclass
class ValidationConfig:
    # Mach sensitivity cells: W/delta x Mach, from the steady windows. The W/delta
    # values are the wd_quantiles of the window W/delta; a cell holds the windows
    # within cell_dwd_rel x W/delta and cell_dmach of its centre. The Mach centres are
    # the multiples of mach_step from the mach_quantiles of the window Mach, up to the
    # MMO. A cell is supported if it holds at least cell_min_windows windows.
    wd_quantiles: tuple = (0.10, 0.35, 0.65, 0.90)
    cell_dwd_rel: float = 0.04
    mach_quantiles: tuple = (0.01, 0.99)
    mach_step: float = 0.01
    cell_dmach: float = 0.01
    cell_min_windows: int = 30
    # Mach of maximum specific air range: search grid from MMO - sr_mach_span to the
    # MMO of the aircraft type in steps of sr_mach_step
    sr_mach_span: float = 0.22
    sr_mach_step: float = 0.001
    # temperature test: Mach of maximum specific air range at the median pressure
    # altitude and mass of the steady windows for these ISA deviations (K)
    temp_dT_isa: tuple = (-20.0, -10.0, 0.0, 10.0, 20.0)
    # constraint grid: cells per dimension (Mach, altitude, mass, energy rate, ISA dev.)
    grid_n: tuple = (24, 20, 11, 17, 8)
    grid_min_count: int = 20  # training blocks per 4D cell for "inside the support"
    tol_mach_convexity: float = 0.1
    tol_mass_monotone: float = 0.01
    tol_energy_monotone: float = 0.05
    # speed-change pairs of steady windows of one flight
    pair_max_dalt_ft: float = 200.0
    pair_min_dmach: float = 0.015
    # 5-minute runs of cruise blocks for the smoothness test
    cruise_run_blocks: int = 30
    # blocks inside a steady window (window length win_s of the preparation, 10 s
    # blocks): the window is used if it holds at least window_min_blocks blocks
    window_s: float = 300.0
    block_s: float = 10.0
    window_min_blocks: int = 25
    # input-noise test: Gaussian noise SD of Mach, pressure altitude (ft), mass
    # (relative), energy rate (split as independent noise of SD noise_energy / sqrt(2)
    # on sin_gamma and on acc_in, so that their sum has SD noise_energy) and SAT (K)
    noise_mach: float = 0.002
    noise_alt_ft: float = 25.0
    noise_mass_rel: float = 0.005
    noise_energy: float = 5e-4
    noise_sat_k: float = 0.5
    noise_seed: int = 0
    # surveillance-input test (Mode S): resolution of Mach, altitude (ft) and
    # groundspeed (kt); temperature unchanged
    mode_s_mach: float = 0.004
    mode_s_alt_ft: float = 25.0
    mode_s_gs_kt: float = 1.0


# ---------------------------------------------------------------- helpers
def acc_metrics(pred, obs):
    """MAPE, relative RMSE and mean bias (%) of fuel flow."""
    e = np.asarray(pred) / np.asarray(obs) - 1.0
    if len(e) == 0:
        return {"n": 0, "MAPE_pct": None, "rRMSE_pct": None, "bias_pct": None}
    return {
        "n": len(e),
        "MAPE_pct": 100 * np.abs(e).mean(),
        "rRMSE_pct": 100 * np.sqrt((e**2).mean()),
        "bias_pct": 100 * e.mean(),
    }


def predict_ln(model, df):
    x = torch.tensor(df[FEATURES].to_numpy(dtype=float), dtype=torch.float64)
    with torch.no_grad():
        return model(x).numpy()


def state(mach, alt_ft, mass_kg, eps, sat_k):
    """(n, 5) float64 states (M, ln delta, ln m, eps, ln theta) from arrays."""
    m, a, w, e, t = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (mach, alt_ft, mass_kg, eps, sat_k)]
    )
    p = np.stack(
        [m, np.log(atmos.isa_delta(a)), np.log(w), e, np.log(t / atmos.T0)], -1
    )
    return torch.tensor(p.reshape(-1, 5), dtype=torch.float64)


def derivatives(model, p, chunk=100000):
    """f = ln F, gradient (n, 5) and f_MM at states p."""
    fs, gs, hs = [], [], []
    for i in range(0, len(p), chunk):
        q = p[i : i + chunk].clone().requires_grad_(True)
        f = model.ln_ff_state(q)
        g = torch.autograd.grad(f.sum(), q, create_graph=True)[0]
        fmm = torch.autograd.grad(g[:, 0].sum(), q)[0][:, 0]
        fs.append(f.detach())
        gs.append(g.detach())
        hs.append(fmm.detach())
    return torch.cat(fs).numpy(), torch.cat(gs).numpy(), torch.cat(hs).numpy()


def hessian(model, p, chunk=50000):
    """f = ln F, gradient (n, 5) and Hessian (n, 5, 5) at states p."""
    fs, gs, hs = [], [], []
    for i in range(0, len(p), chunk):
        q = p[i : i + chunk].clone().requires_grad_(True)
        f = model.ln_ff_state(q)
        g = torch.autograd.grad(f.sum(), q, create_graph=True)[0]
        rows = []
        for k in range(q.shape[1]):
            gk = g[:, k]
            if not gk.requires_grad:  # a linear input (ln theta of the fixed rule)
                rows.append(torch.zeros_like(q))
                continue
            r = torch.autograd.grad(gk.sum(), q, retain_graph=True, allow_unused=True)
            rows.append(torch.zeros_like(q) if r[0] is None else r[0])
        fs.append(f.detach())
        gs.append(g.detach())
        hs.append(torch.stack(rows, 1).detach())
    return torch.cat(fs).numpy(), torch.cat(gs).numpy(), torch.cat(hs).numpy()


# ---------------------------------------------------------------- sections
def accuracy(pred, te):
    sub = {
        "all": np.ones(len(te), bool),
        "climb": (te.phase == "climb").to_numpy(),
        "level": (te.phase == "level").to_numpy(),
        "descent": (te.phase == "descent").to_numpy(),
        "cruise": te.cruise.to_numpy(dtype=bool),
    }
    return {k: acc_metrics(pred[s], te.ff_kgh.to_numpy()[s]) for k, s in sub.items()}


def trip_fuel(pred, te):
    """Fuel summed over the kept blocks of each flight, error in %."""
    d = pd.DataFrame(
        {"f": te.flight_id.to_numpy(), "p": pred, "o": te.ff_kgh.to_numpy()}
    )
    g = d.groupby("f")[["p", "o"]].sum()
    e = 100 * (g.p / g.o - 1)
    return {
        "flights": len(g),
        "median_abs_error_pct": e.abs().median(),
        "mean_bias_pct": e.mean(),
        "median_bias_pct": e.median(),
    }


def cell_grid(windows, cfg: ValidationConfig, ac: Aircraft):
    """(W/delta values in t, Mach centres) of the sensitivity cells from the windows."""
    if len(windows) == 0:
        return np.array([]), np.array([])
    wd = np.round(np.quantile(windows.wd_t.to_numpy(), cfg.wd_quantiles), 1)
    lo, hi = np.quantile(windows.mach.to_numpy(), cfg.mach_quantiles)
    st = cfg.mach_step
    mach = np.arange(np.floor(lo / st) * st, np.ceil(hi / st) * st + st / 2, st)
    mach = np.round(mach, 6)
    return wd, mach[mach <= ac.mmo + 1e-9]


def sensitivity_cells(windows, cfg: ValidationConfig, ac: Aircraft):
    """Representative conditions per (W/delta, Mach) cell from the steady windows."""
    rows = []
    wd_grid, mach_cells = cell_grid(windows, cfg, ac)
    wd_w, m_w = windows.wd_t.to_numpy(), windows.mach.to_numpy()
    for q, wd in zip(cfg.wd_quantiles, wd_grid, strict=False):
        for m in mach_cells:
            sel = (np.abs(m_w - m) <= cfg.cell_dmach + 1e-9) & (
                np.abs(wd_w - wd) <= cfg.cell_dwd_rel * wd
            )
            n = int(sel.sum())
            r = {
                "wd_quantile": q,
                "wd_t": wd,
                "mach": m,
                "n_windows": n,
                "supported": n >= cfg.cell_min_windows,
            }
            if r["supported"]:
                r.update(
                    alt_ft=windows.alt_ft.to_numpy()[sel].mean(),
                    mass_kg=windows.mass_kg.to_numpy()[sel].mean(),
                    sat_k=windows.sat_k.to_numpy()[sel].mean(),
                )
            rows.append(r)
    cols = ["wd_quantile", "wd_t", "mach", "n_windows", "supported"]
    return pd.DataFrame(rows, columns=cols if not rows else None)


def mach_sensitivity(model, cells):
    """s = d ln F / dM at eps = 0 in the supported cells."""
    out = cells.copy()
    out["s"] = np.nan
    sup = out.supported.to_numpy(dtype=bool)
    if sup.any():
        c = out[sup]
        p = state(c.mach, c.alt_ft, c.mass_kg, 0.0, c.sat_k)
        out.loc[sup, "s"] = derivatives(model, p)[1][:, 0]
    out["one_over_mach"] = 1 / out.mach
    s = out.s[sup]
    summary = {
        "supported_cells": int(sup.sum()),
        "mean_s_supported": s.mean() if len(s) else None,
        "cells_s_above_1_over_mach": int((s > out.one_over_mach[sup]).sum()),
        "mean_s_per_wd_t": {
            f"{wd:g}": out.s[sup & (out.wd_t == wd).to_numpy()].mean()
            for wd in out.wd_t.unique()
        },
        "cells": out.to_dict(orient="records"),
    }
    return summary


def sr_maximum(model, mg, alt_ft, mass_kg, sat_k):
    """Mach of maximum M / F on the Mach grid mg at eps = 0 (specific air range at
    fixed temperature). Returns (Mach, at the upper bound, number of local maxima)."""
    with torch.no_grad():
        f = model.ln_ff_state(state(mg, alt_ft, mass_kg, 0.0, sat_k)).numpy()
    y = np.log(mg) - f
    i = int(np.argmax(y))
    inner = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    n_max = int(inner.sum()) + int(y[0] > y[1]) + int(y[-1] > y[-2])
    return mg[i], i == len(mg) - 1, n_max


def mach_grid(cfg: ValidationConfig, ac: Aircraft):
    """Search grid of the Mach of maximum specific air range: MMO - sr_mach_span to
    the MMO."""
    lo, hi, step = ac.mmo - cfg.sr_mach_span, ac.mmo, cfg.sr_mach_step
    return np.round(np.arange(lo, hi + step / 2, step), 6)


def max_specific_range(model, cells, cfg: ValidationConfig, ac: Aircraft):
    """Mach of maximum M / F (specific air range at fixed temperature) on a fine grid,
    at the conditions of the best-supported cell of each W/delta; mass gives the
    nominal W/delta."""
    mg = mach_grid(cfg, ac)
    rows = []
    if len(cells) == 0:
        return rows
    for wd, g in cells[cells.supported].groupby("wd_t"):
        q = g.loc[g.n_windows.idxmax()]
        mass = wd * 1000 * float(atmos.isa_delta(q.alt_ft))
        m, at_ub, n_max = sr_maximum(model, mg, q.alt_ft, mass, q.sat_k)
        rows.append(
            {
                "wd_quantile": q.wd_quantile,
                "wd_t": wd,
                "alt_ft": q.alt_ft,
                "sat_k": q.sat_k,
                "mass_kg": mass,
                "mach_max_sr": m,
                "at_upper_bound": at_ub,
                "n_local_max": n_max,
            }
        )
    return rows


def temperature_point(windows, blocks):
    """(pressure altitude, mass, source) of the temperature test: the medians of the
    steady windows, or of the cruise blocks if there are no windows."""
    if len(windows):
        t, src = windows, "median of the steady windows"
    else:
        t, src = blocks[blocks.cruise.astype(bool)], "median of the cruise blocks"
    if len(t) == 0:
        return None, None, "no cruise data"
    return float(t.alt_ft.median()), float(t.mass_kg.median()), src


def temperature_test(model, cfg: ValidationConfig, ac: Aircraft, alt_ft, mass_kg):
    """Mach of maximum specific air range at alt_ft and mass_kg for the ISA deviations
    temp_dT_isa, and the shift between the extremes (Mach at the largest minus Mach at
    the smallest deviation)."""
    mg = mach_grid(cfg, ac)
    t_isa = float(atmos.isa_temperature(alt_ft))
    rows = []
    for dt in sorted(cfg.temp_dT_isa):
        m, at_ub, n_max = sr_maximum(model, mg, alt_ft, mass_kg, t_isa + dt)
        rows.append(
            {
                "dT_isa_k": dt,
                "sat_k": t_isa + dt,
                "mach_max_sr": m,
                "at_upper_bound": at_ub,
                "n_local_max": n_max,
            }
        )
    return {
        "alt_ft": alt_ft,
        "mass_kg": mass_kg,
        "rows": rows,
        "shift_mach_max_sr": rows[-1]["mach_max_sr"] - rows[0]["mach_max_sr"],
    }


def constraint_grid(train_blocks, cfg: ValidationConfig, env):
    """Regular grid in the envelope (CAS limits applied) and a flag for the cells with
    training data support (4D histogram of Mach, altitude, mass and energy rate, and
    ISA deviation inside its 1 to 99 % range)."""
    keys = ["mach", "alt_ft", "mass_kg", "eps", "dT_isa"]
    centers, edges = [], []
    for k, n in zip(keys, cfg.grid_n, strict=False):
        lo, hi = getattr(env, k)
        e = np.linspace(lo, hi, n + 1)
        centers.append(0.5 * (e[:-1] + e[1:]))
        edges.append(e)
    tb = train_blocks
    eps = (tb.sin_gamma + tb.acc_in).to_numpy()
    cnt, _ = np.histogramdd(
        np.column_stack([tb.mach, tb.alt_ft, tb.mass_kg, eps]), bins=edges[:4]
    )
    ii = [
        a.ravel()
        for a in np.meshgrid(*[np.arange(n) for n in cfg.grid_n], indexing="ij")
    ]
    m, alt, w, e, dt = [c[i] for c, i in zip(centers, ii, strict=False)]
    if len(tb):
        t_lo, t_hi = np.quantile(tb.dT_isa.to_numpy(), [0.01, 0.99])
    else:
        t_lo, t_hi = np.inf, -np.inf
    inside = (
        (cnt[ii[0], ii[1], ii[2], ii[3]] >= cfg.grid_min_count)
        & (dt >= t_lo)
        & (dt <= t_hi)
    )
    cas = atmos.cas_from_mach(m, alt)
    ok = (cas >= env.cas_kt[0]) & (cas <= env.cas_kt[1])
    sat = atmos.isa_temperature(alt) + dt
    p = state(m[ok], alt[ok], w[ok], e[ok], sat[ok])
    return p, inside[ok]


def constraints(
    model, p, inside, g, h, cfg: ValidationConfig, lc: LossConfig, ac, drag=None
):
    """Violation rates of P1 to P3 on the grid states p (gradient g and Hessian h of
    ln F there; P1 and P2 only at powered states), of P6 at the MMO and, with the
    stored P7 settings `drag`, of P7."""
    env = envelope(ac, lc)
    fmm = h[:, 0, 0]
    powered = p[:, 3].numpy() >= lc.eps_powered - 1e-9
    res = {
        "mach_convexity": (fmm + g[:, 0] ** 2, cfg.tol_mach_convexity, powered),
        "mass_monotone": (g[:, 2], cfg.tol_mass_monotone, powered),
        "energy_monotone": (g[:, 3], cfg.tol_energy_monotone, np.ones(len(p), bool)),
    }
    out = {"grid_points": len(p), "inside_support": int(inside.sum())}
    for reg, sel in [
        ("inside", inside),
        ("outside", ~inside),
        ("all", np.ones(len(p), bool)),
    ]:
        out[reg] = {
            f"{k}_violation_pct": 100 * (r[sel & use] < -tol).mean()
            if (sel & use).any()
            else None
            for k, (r, tol, use) in res.items()
        }
    # range boundary: M f_M >= p6_min at M = MMO, eps = 0
    a = np.linspace(*p6_band(ac, lc), 10)
    w = np.linspace(*env.mass_kg, 11)
    t = np.linspace(*env.dT_isa, 8)
    A, W, T = [x.ravel() for x in np.meshgrid(a, w, t, indexing="ij")]
    q = state(ac.mmo, A, W, 0.0, atmos.isa_temperature(A) + T)
    gm = derivatives(model, q)[1][:, 0]
    out["range_boundary"] = {
        "points": len(q),
        "mach": ac.mmo,
        "violation_pct": 100 * (ac.mmo * gm < lc.p6_min).mean(),
        "min_mach_times_dlnF_dM": (ac.mmo * gm).min(),
    }
    if drag:
        out["drag_rise"] = drag_rise_test(model, lc, ac, drag)
    return out


def drag_rise_states(lc: LossConfig, ac: Aircraft, drag):
    """Regular grid of P7 (level flight, eps = 0): Mach from the lower Mach of P7 to
    the MMO, the altitude, mass and ISA-deviation ranges of the envelope, CAS limits
    and the CL bound of P7 applied. Returns (states, CL)."""
    env = envelope(ac, lc)
    nm, na, nw, nt = DRAG_RISE_GRID
    m, a, w, t = (
        x.ravel()
        for x in np.meshgrid(
            np.linspace(drag["mach_min"], drag["mach_max"], nm),
            np.linspace(*env.alt_ft, na),
            np.linspace(*env.mass_kg, nw),
            np.linspace(*env.dT_isa, nt),
            indexing="ij",
        )
    )
    cas = atmos.cas_from_mach(m, a)
    cl = dragmod.level_cl(m, np.log(atmos.isa_delta(a)), np.log(w), ac.wing_area_m2)
    ok = (cas >= env.cas_kt[0]) & (cas <= env.cas_kt[1]) & (cl <= drag["cl_max"])
    sat = atmos.isa_temperature(a[ok]) + t[ok]
    return state(m[ok], a[ok], w[ok], 0.0, sat), cl[ok]


def stored_drag_table(drag, name="model"):
    """Rebuild the stored drag-slope table of P7 with the installed aerox. Returns
    (table, whether it has the stored hash). ValueError if no hash is stored."""
    if not drag.get("table_sha256"):
        raise ValueError(
            f"the drag-slope table of P7 of the {name} cannot be checked: its P7 "
            "settings (constraints.drag_rise) have no table_sha256"
        )
    table = dragmod.build_table(drag["table"])
    return table, dragmod.table_hash(table) == drag["table_sha256"]


def drag_rise_test(model, lc: LossConfig, ac: Aircraft, drag):
    """P7 on its grid: share of states with f_M < g - DRAG_RISE_TOL, the smallest
    margin f_M - g, and whether the rebuilt aerox table has the stored hash."""
    q, cl = drag_rise_states(lc, ac, drag)
    table, match = stored_drag_table(drag)
    g = dragmod.lookup(table, drag["table"], q[:, 0].numpy(), cl, q[:, 1].numpy())
    margin = derivatives(model, q)[1][:, 0] - g
    return {
        "points": len(q),
        "mach": [drag["mach_min"], drag["mach_max"]],
        "cl_max": drag["cl_max"],
        "tolerance": DRAG_RISE_TOL,
        "violation_pct": 100 * (margin < -DRAG_RISE_TOL).mean(),
        "min_margin": margin.min(),
        "table_sha256_match": match,
        "aerox_version": dragmod.aerox_version(),
    }


STATE_NAMES = ["mach", "ln_delta", "ln_m", "eps", "ln_theta"]


def state_sd(train_blocks):
    """SD of the state coordinates (M, ln delta, ln m, eps, ln theta) of the blocks."""
    tb = train_blocks
    v = np.column_stack(
        [
            tb.mach,
            tb.ln_delta,
            tb.ln_m,
            tb.sin_gamma + tb.acc_in,
            tb.ln_theta,
        ]
    ).astype(float)
    sd = v.std(0) if len(v) > 1 else np.ones(5)
    sd[~(sd > 0)] = 1.0
    return sd


def roughness(h, inside, sd):
    """Mean Frobenius norm of the Hessian of ln F in standardized state coordinates
    (each coordinate divided by its SD in the training blocks) on the constraint grid,
    inside and outside the support of the training data."""
    hz = h * sd[None, :, None] * sd[None, None, :]
    fro = np.sqrt((hz**2).sum((1, 2)))
    return {
        "coordinates": STATE_NAMES,
        "sd": sd.tolist(),
        "points": len(fro),
        "inside_support": int(inside.sum()),
        "mean_hessian_frobenius_inside": fro[inside].mean() if inside.any() else None,
        "mean_hessian_frobenius_outside": fro[~inside].mean()
        if (~inside).any()
        else None,
        "mean_hessian_frobenius_all": fro.mean() if len(fro) else None,
    }


def window_pairs(w, cfg: ValidationConfig):
    """Pairs of steady windows of one flight within pair_max_dalt_ft and with a Mach
    difference of at least pair_min_dmach; j has the higher Mach."""
    ii, jj = [], []
    alt, m = w.alt_ft.to_numpy(), w.mach.to_numpy()
    for _, g in w.groupby("flight_id", sort=False):
        idx = g.index.to_numpy()
        if len(idx) < 2:
            continue
        a, b = np.triu_indices(len(idx), 1)
        i, j = idx[a], idx[b]
        keep = (np.abs(alt[i] - alt[j]) <= cfg.pair_max_dalt_ft) & (
            np.abs(m[i] - m[j]) >= cfg.pair_min_dmach
        )
        i, j = i[keep], j[keep]
        swap = m[i] > m[j]
        ii.append(np.where(swap, j, i))
        jj.append(np.where(swap, i, j))
    if not ii:
        return np.array([], int), np.array([], int)
    return np.concatenate(ii), np.concatenate(jj)


def pair_test(model, w, cfg: ValidationConfig):
    """Speed-change pair test on the windows w: predicted against measured change of
    ln fuel flow between the windows of each pair."""
    w = w.reset_index(drop=True)
    i, j = window_pairs(w, cfg)
    n_fl = len(pd.unique(w.flight_id.to_numpy()[i]))
    if len(i) < 2:
        return {"pairs": len(i), "flights": n_fl}
    ln_p = predict_ln(model, w)
    d = ln_p[j] - ln_p[i]
    meas = w.ln_ff.to_numpy()[j] - w.ln_ff.to_numpy()[i]
    dm = w.mach.to_numpy()[j] - w.mach.to_numpy()[i]
    return {
        "pairs": len(i),
        "flights": n_fl,
        "median_abs_dmach": np.median(np.abs(dm)),
        "slope_through_origin": (d * meas).sum() / (meas * meas).sum(),
        "correlation": np.corrcoef(d, meas)[0, 1],
        "rmse_of_difference_pct": 100 * np.sqrt(((d - meas) ** 2).mean()),
        "median_predicted_dlnF_dM": np.median(d / dm),
        "median_measured_dlnF_dM": np.median(meas / dm),
    }


PAIR_SETS = {
    "test": (("test",), "test split"),
    "test_and_val": (("test", "val"), "test and validation splits"),
}


def pair_tests(model, windows, cfg: ValidationConfig):
    """Pair test on the windows of the test split and of the test and validation
    splits; every result names the splits of its flights."""
    out = {}
    for key, (splits, text) in PAIR_SETS.items():
        w = windows[windows.split.isin(splits)]
        out[key] = {
            "flights_used": text,
            "splits": list(splits),
            **pair_test(model, w, cfg),
        }
    return out


def smoothness(pred, te, cfg: ValidationConfig):
    """RMS first difference between consecutive 10 s blocks (kg/h), predicted and
    measured, per phase, and the median ratio of the predicted to measured SD in
    5-minute runs of cruise blocks."""
    order = np.lexsort((te.time_s.to_numpy(), te.seg.to_numpy()))
    te = te.iloc[order].reset_index(drop=True)
    pred = pred[order]
    seg, ts, ph = te.seg.to_numpy(), te.time_s.to_numpy(), te.phase.to_numpy()
    obs = te.ff_kgh.to_numpy()
    cons = (seg[1:] == seg[:-1]) & (np.abs(ts[1:] - ts[:-1] - cfg.block_s) < 1e-6)
    out = {}
    for k in ["all", "climb", "level", "descent"]:
        m = cons if k == "all" else cons & (ph[1:] == k) & (ph[:-1] == k)
        r = {"pairs": int(m.sum())}
        if m.any():
            r["rms_diff_pred_kgh"] = np.sqrt((np.diff(pred)[m] ** 2).mean())
            r["rms_diff_meas_kgh"] = np.sqrt((np.diff(obs)[m] ** 2).mean())
            r["ratio_pred_to_meas"] = r["rms_diff_pred_kgh"] / r["rms_diff_meas_kgh"]
        out[k] = r
    cr = te.cruise.to_numpy(dtype=bool)
    new = np.ones(len(te), bool)
    new[1:] = ~(cons & cr[1:] & cr[:-1])
    run = np.cumsum(new)
    idx = np.flatnonzero(cr)
    n = cfg.cruise_run_blocks
    if len(idx):
        pos = pd.Series(idx).groupby(run[idx]).cumcount().to_numpy()
        size = pd.Series(idx).groupby(run[idx]).transform("size").to_numpy()
        win = idx[pos < (size // n) * n].reshape(-1, n)
    else:
        win = np.zeros((0, n), int)
    out["cruise_runs"] = len(win)
    if len(win):
        sd_p, sd_o = pred[win].std(1, ddof=1), obs[win].std(1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["cruise_sd_ratio_median"] = np.nanmedian(sd_p / sd_o)
    return out


def steady_window_sd(pred, te, wt, cfg: ValidationConfig):
    """SD of the predicted and measured fuel flow of the blocks inside each steady
    window (blocks of the same flight that lie completely inside the window), for the
    windows with at least window_min_blocks blocks; median of the ratio."""
    half = (cfg.block_s - 1) / 2  # block time is the mean of its 1 Hz sample times
    t_b = te.time_s.to_numpy()
    obs = te.ff_kgh.to_numpy()
    pos = te.groupby("flight_id", sort=False).indices
    sd_p, sd_o = [], []
    for f, g in wt.groupby("flight_id", sort=False):
        if f not in pos:
            continue
        idx = pos[f][np.argsort(t_b[pos[f]], kind="stable")]
        tf = t_b[idx]
        for t0 in g.time_s.to_numpy():
            a = np.searchsorted(tf, t0 + half - 1e-6, "left")
            b = np.searchsorted(tf, t0 + cfg.window_s - 1 - half + 1e-6, "right")
            if b - a >= cfg.window_min_blocks:
                k = idx[a:b]
                sd_p.append(pred[k].std(ddof=1))
                sd_o.append(obs[k].std(ddof=1))
    sd_p, sd_o = np.array(sd_p), np.array(sd_o)
    out = {"windows": len(sd_p)}
    if len(sd_p):
        with np.errstate(divide="ignore", invalid="ignore"):
            out["sd_ratio_median"] = np.nanmedian(sd_p / sd_o)
        out["sd_pred_median_kgh"] = np.median(sd_p)
        out["sd_meas_median_kgh"] = np.median(sd_o)
    return out


NOISE_INPUTS = ("mach", "alt_ft", "mass", "energy", "sat_k")


def noise_sensitivity(model, te, cfg: ValidationConfig):
    """RMS relative change (%) of the predicted fuel flow for Gaussian noise on one
    input at a time and on all inputs together (noise SD in ValidationConfig)."""
    x0 = te[FEATURES].to_numpy(dtype=float)
    n = len(x0)
    rng = np.random.default_rng(cfg.noise_seed)
    z = rng.standard_normal((n, 6))
    alt = te.alt_ft.to_numpy(dtype=float)
    sat = te.sat_k.to_numpy(dtype=float)
    dx = {k: np.zeros_like(x0) for k in NOISE_INPUTS}
    dx["mach"][:, 0] = cfg.noise_mach * z[:, 0]
    dx["alt_ft"][:, 1] = np.log(
        atmos.isa_delta(alt + cfg.noise_alt_ft * z[:, 1])
    ) - np.log(atmos.isa_delta(alt))
    dx["mass"][:, 2] = np.log1p(cfg.noise_mass_rel * z[:, 2])
    dx["energy"][:, 3] = cfg.noise_energy / math.sqrt(2) * z[:, 3]
    dx["energy"][:, 4] = cfg.noise_energy / math.sqrt(2) * z[:, 4]
    dx["sat_k"][:, 5] = np.log1p(cfg.noise_sat_k * z[:, 5] / sat)

    def f(x):
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float64)).numpy()

    f0 = f(x0)
    out = {
        "blocks": n,
        "noise_sd": {
            "mach": cfg.noise_mach,
            "alt_ft": cfg.noise_alt_ft,
            "mass_rel": cfg.noise_mass_rel,
            "energy_rate": cfg.noise_energy,
            "energy_split": "independent SD/sqrt(2) on sin_gamma and on acc_in",
            "sat_k": cfg.noise_sat_k,
        },
    }
    for k, d in {**dx, "joint": sum(dx.values())}.items():
        out[f"rms_rel_change_pct_{k}"] = 100 * np.sqrt(
            (np.expm1(f(x0 + d) - f0) ** 2).mean()
        )
    return out


def mode_s_blocks(te, samples, cfg: ValidationConfig, prep_config=None):
    """Block inputs rebuilt from 1 Hz samples at Mode S quality: Mach rounded to
    mode_s_mach, pressure altitude to mode_s_alt_ft, groundspeed to mode_s_gs_kt,
    temperature unchanged; derivatives by the Savitzky-Golay filter of
    inference.surveillance_inputs; block means as in neuralfuel.prep. Returns the
    inputs aligned row by row to te (NaN where a block has no rebuilt inputs)."""
    from .inference import surveillance_inputs

    flights = te.flight_id.unique()
    samples = samples[samples.flight_id.isin(flights)]
    if samples.empty:
        raise ValueError(
            "the samples have no flight of the evaluated blocks: give the data and "
            "the adapter that the prepared blocks come from"
        )
    s, _, _ = derive(samples, prep_config)
    mach = np.round(s.mach.to_numpy() / cfg.mode_s_mach) * cfg.mode_s_mach
    alt = np.round(s.alt_ft.to_numpy() / cfg.mode_s_alt_ft) * cfg.mode_s_alt_ft
    gs = np.round(s.gs_kt.to_numpy() / cfg.mode_s_gs_kt) * cfg.mode_s_gs_kt
    sat = s.sat_k.to_numpy()
    x = surveillance_inputs(
        s.time_s, alt, gs, s.mass_kg, sat, mach=mach, track_id=s.flight_id
    )
    seg = s.seg.to_numpy()
    start = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    first = np.repeat(start, np.diff(np.r_[start, len(seg)]))
    key = seg.astype(np.int64) * 1_000_000 + (np.arange(len(s)) - first) // int(
        cfg.block_s
    )
    d = pd.DataFrame(
        {
            "time_s": s.time_s.to_numpy(),
            "mach": mach,
            "delta": atmos.isa_delta(alt),
            "theta": sat / atmos.T0,
            "mass_kg": s.mass_kg.to_numpy(),
            "sin_gamma": x.sin_gamma.to_numpy(),
            "acc_in": x.acc_in.to_numpy(),
        }
    )
    b = d.groupby(key, sort=True).mean()
    b["flight_id"] = s.flight_id.groupby(key, sort=True).first().to_numpy()
    b["n"] = d.groupby(key, sort=True).size()
    b = b[b.n == int(cfg.block_s)]
    b["t_key"] = np.round(b.time_s.to_numpy(), 3)
    k = pd.DataFrame(
        {"flight_id": te.flight_id.to_numpy(), "t_key": np.round(te.time_s, 3)}
    )
    m = k.merge(b, on=["flight_id", "t_key"], how="left", validate="1:1")
    return pd.DataFrame(
        {
            "mach": m.mach.to_numpy(),
            "ln_delta": np.log(m.delta.to_numpy()),
            "ln_m": np.log(m.mass_kg.to_numpy()),
            "sin_gamma": m.sin_gamma.to_numpy(),
            "acc_in": m.acc_in.to_numpy(),
            "ln_theta": np.log(m.theta.to_numpy()),
        }
    )


def surveillance_test(model, te, samples, cfg: ValidationConfig, prep_config=None):
    """Accuracy with Mode S quality inputs (mode_s_blocks) against the accuracy with
    the recorded inputs, on the same blocks."""
    xs = mode_s_blocks(te, samples, cfg, prep_config)
    ok = np.isfinite(xs.to_numpy()).all(axis=1)
    t = te[ok].reset_index(drop=True)
    p_ref = np.exp(predict_ln(model, t))
    p_s = np.exp(predict_ln(model, xs[ok].reset_index(drop=True)))
    return {
        "variant": (
            f"Mode S at 1 Hz: Mach rounded to {cfg.mode_s_mach:g}, pressure altitude "
            f"to {cfg.mode_s_alt_ft:g} ft, groundspeed to {cfg.mode_s_gs_kt:g} kt, "
            "temperature unchanged; the schema has no wind, so airspeed from "
            "groundspeed and wind is not tested"
        ),
        "blocks": int(ok.sum()),
        "blocks_without_inputs": int((~ok).sum()),
        "recorded_inputs": accuracy(p_ref, t),
        "surveillance_inputs": accuracy(p_s, t),
    }


def envelope_section(model, te, tr, tr_windows, ac, prep_config=None):
    """The valid envelope of the model (from the file, or from the training split of
    the validated data) and the share of the evaluated blocks outside it."""
    env, src = model.envelope, "model file"
    if env is None:
        pc = prep_config or PrepConfig()
        rule = {"level_fpm": pc.level_fpm, "cruise_alt_ft": pc.cruise_alt_ft}
        env = data_envelope(tr, tr_windows, ac.wing_area_m2, **rule)
        src = "training split of the validated data"
    inside = in_envelope(env, te[FEATURES])
    out = {"source": src, "evaluated_outside_pct": {}}
    for p in ("all", "climb", "level", "descent"):
        sel = np.ones(len(te), bool) if p == "all" else (te.phase == p).to_numpy()
        out["evaluated_outside_pct"][p] = (
            100 * (~inside[sel]).mean() if sel.any() else None
        )
    out["description"] = env
    return out


COVERAGE_VARS = {
    "mach": ("mach", np.round(np.arange(0.2, 0.901, 0.05), 3)),
    "alt_ft": ("alt_ft", np.arange(0, 45001, 5000)),
    "mass_kg": ("mass_kg", np.arange(40000, 90001, 5000)),
    "dT_isa_k": ("dT_isa", np.arange(-30, 31, 5)),
    "energy_rate": (None, np.round(np.arange(-0.1, 0.101, 0.02), 3)),
}


def coverage(blocks, windows):
    """Ranges, quantiles and histogram counts of the main variables."""
    out = {
        "blocks": len(blocks),
        "flights": int(blocks.flight_id.nunique()),
        "windows": len(windows),
        "blocks_per_phase": {
            p: int((blocks.phase == p).sum()) for p in ("climb", "level", "descent")
        },
        "cruise_blocks": int(blocks.cruise.sum()),
        "variables": {},
    }
    for name, (col, bins) in COVERAGE_VARS.items():
        v = (blocks.sin_gamma + blocks.acc_in) if col is None else blocks[col]
        v = v.to_numpy(dtype=float)
        q = np.quantile(v, [0, 0.01, 0.5, 0.99, 1]) if len(v) else [np.nan] * 5
        cnt, _ = np.histogram(v, bins=bins)
        out["variables"][name] = {
            "min": q[0],
            "p1": q[1],
            "p50": q[2],
            "p99": q[3],
            "max": q[4],
            "bin_edges": bins.tolist(),
            "counts": cnt.tolist(),
            "below_first_edge": int((v < bins[0]).sum()),
            "above_last_edge": int((v > bins[-1]).sum()),
        }
    return out


# ---------------------------------------------------------------- report
def validate(
    model: FuelFlowNet,
    blocks: pd.DataFrame,
    windows: pd.DataFrame,
    splits=("test",),
    cfg: ValidationConfig | None = None,
    loss_cfg: LossConfig | None = None,
    aircraft: Aircraft | None = None,
    samples: pd.DataFrame | None = None,
    prep_config: PrepConfig | None = None,
):
    """Validation report (dict) on the blocks and windows of the given splits
    (names in SPLITS; ValueError for another name).

    The aircraft settings (envelope of the constraint tests, MMO) come from the model;
    `aircraft` is needed only for a model without them (a network made in code
    without aircraft settings). The constraint settings (loss_cfg) come from the model
    too (FuelFlowNet.constraints; the defaults of LossConfig for a model without
    them).

    prep_config: the PrepConfig of the preparation that made the blocks (the 'config'
    of prep.json). It sets block_s and window_s of cfg, the phase settings of an
    envelope made from the data and the preparation of the surveillance test. With
    None, the defaults of PrepConfig are used.

    Accuracy, trip fuel, steady-window accuracy, smoothness and the noise test use the
    given splits. The pair test always reports the test split and the test and
    validation splits. The Mach sensitivity cells and the temperature-test point use
    the steady windows of all splits (they only set representative flight
    conditions). The support of the constraint grid and the roughness scale use the
    'train' blocks if there are any, else all blocks.

    samples: the 1 Hz samples in the schema (adapter output) of the data; if given,
    the optional surveillance-input test runs on the evaluated blocks.
    """
    loss_cfg = loss_cfg or model_loss_config(model)
    drag = (model.constraints or {}).get("drag_rise")
    cfg = cfg or ValidationConfig()
    if prep_config is not None:
        cfg = replace(
            cfg, block_s=float(prep_config.block_s), window_s=float(prep_config.win_s)
        )
    splits = list(splits)
    bad = [s for s in splits if s not in SPLITS]
    if bad or not splits:
        raise ValueError(f"unknown split names {bad}; known: {', '.join(SPLITS)}")
    ac = model.aircraft or aircraft
    if ac is None:
        raise ValueError(
            "the model has no aircraft settings; give them as aircraft=..."
        )
    ac.check()
    m64 = copy.deepcopy(model).cpu().double().eval()
    te = blocks[blocks.split.isin(splits)].reset_index(drop=True)
    wt = windows[windows.split.isin(splits)].reset_index(drop=True)
    if len(te) == 0:
        raise ValueError(f"no blocks in the splits {splits}")
    tr = blocks[blocks.split == "train"]
    if len(tr) == 0:
        tr = blocks
    tr_w = (
        windows[windows.split == "train"]
        if (blocks.split == "train").any()
        else windows
    )
    pred = np.exp(predict_ln(m64, te))
    cells = sensitivity_cells(windows, cfg, ac)
    t_alt, t_mass, t_src = temperature_point(windows, blocks)
    p, inside = constraint_grid(tr, cfg, envelope(ac, loss_cfg))
    _, g, h = hessian(m64, p)
    temp = (
        {"point": t_src, **temperature_test(m64, cfg, ac, t_alt, t_mass)}
        if t_alt is not None
        else {"point": t_src, "rows": []}
    )
    report = {
        "format": REPORT_FORMAT,
        "neuralfuel_version": __version__,
        "model": {
            "weights_sha256": state_hash(model),
            "n_params": sum(p.numel() for p in model.parameters()),
            "features": list(FEATURES),
            "temperature": model.temperature,
            "theta_exp": model.exponent(),
            "size": size_of(model.hidden, model.layers),
            "aircraft": ac.as_dict(),
            "aircraft_from_file": model.aircraft is not None,
            "name": model_name(ac.type, ac.engine, model.size)
            if model.size != "custom"
            else None,
        },
        "evaluated_splits": splits,
        "accuracy": accuracy(pred, te),
        "trip_fuel": trip_fuel(pred, te),
        "steady_windows": acc_metrics(np.exp(predict_ln(m64, wt)), wt.ff_kgh.to_numpy())
        if len(wt)
        else {"n": 0},
        "mach_sensitivity": mach_sensitivity(m64, cells),
        "max_specific_range": max_specific_range(m64, cells, cfg, ac),
        "temperature_test": temp,
        "constraints": constraints(m64, p, inside, g, h, cfg, loss_cfg, ac, drag),
        "roughness": roughness(h, inside, state_sd(tr)),
        "pair_test": pair_tests(m64, windows, cfg),
        "smoothness": {
            **smoothness(pred, te, cfg),
            "steady_windows": steady_window_sd(pred, te, wt, cfg),
        },
        "noise_sensitivity": noise_sensitivity(m64, te, cfg),
        "envelope": envelope_section(model, te, tr, tr_w, ac, prep_config),
        "coverage": {
            "evaluated": coverage(te, wt),
            "train": coverage(tr, windows[windows.split == "train"]),
        },
        "config": asdict(cfg),
        "constraint_settings": constraint_settings(loss_cfg)
        | ({"drag_rise": drag} if drag else {}),
        "prep": asdict(prep_config) if prep_config is not None else None,
    }
    if samples is not None:
        report["surveillance"] = surveillance_test(m64, te, samples, cfg, prep_config)
    return clean_json(report)


def _fields(cls):
    """Format (neuralfuel.privacy) of the fields of a settings dataclass."""
    import dataclasses

    return {
        f.name: STR if f.type in (str, "str") else NUM for f in dataclasses.fields(cls)
    }


_ACC = dict.fromkeys(("n", "MAPE_pct", "rRMSE_pct", "bias_pct"), NUM)
_PHASES = ("all", "climb", "level", "descent")
_COVERAGE = {
    "blocks": NUM,
    "flights": NUM,
    "windows": NUM,
    "blocks_per_phase": dict.fromkeys(("climb", "level", "descent"), NUM),
    "cruise_blocks": NUM,
    "variables": {
        k: dict.fromkeys(
            (
                "min",
                "p1",
                "p50",
                "p99",
                "max",
                "bin_edges",
                "counts",
                "below_first_edge",
                "above_last_edge",
            ),
            NUM,
        )
        for k in COVERAGE_VARS
    },
}
_PAIR = {
    "flights_used": STR,
    "splits": STR,
    **dict.fromkeys(
        (
            "pairs",
            "flights",
            "median_abs_dmach",
            "slope_through_origin",
            "correlation",
            "rmse_of_difference_pct",
            "median_predicted_dlnF_dM",
            "median_measured_dlnF_dM",
        ),
        NUM,
    ),
}
_SR = {"mach_max_sr": NUM, "at_upper_bound": NUM, "n_local_max": NUM}
# the keys of the report (format of neuralfuel.privacy.allowed)
REPORT = {
    "format": STR,
    "neuralfuel_version": STR,
    "model": {
        "weights_sha256": STR,
        "n_params": NUM,
        "features": STR,
        "temperature": STR,
        "theta_exp": NUM,
        "size": STR,
        "aircraft": {
            **dict.fromkeys(
                ("mmo", "vmo_kt", "mass_kg", "alt_ft", "wing_area_m2"), NUM
            ),
            "type": STR,
            "engine": STR,
        },
        "aircraft_from_file": NUM,
        "name": STR,
    },
    "evaluated_splits": STR,
    "accuracy": dict.fromkeys((*_PHASES, "cruise"), _ACC),
    "trip_fuel": dict.fromkeys(
        ("flights", "median_abs_error_pct", "mean_bias_pct", "median_bias_pct"), NUM
    ),
    "steady_windows": _ACC,
    "mach_sensitivity": {
        "supported_cells": NUM,
        "mean_s_supported": NUM,
        "cells_s_above_1_over_mach": NUM,
        "mean_s_per_wd_t": {NUM_KEY: NUM},
        "cells": [
            dict.fromkeys(
                (
                    "wd_quantile",
                    "wd_t",
                    "mach",
                    "n_windows",
                    "supported",
                    "alt_ft",
                    "mass_kg",
                    "sat_k",
                    "s",
                    "one_over_mach",
                ),
                NUM,
            )
        ],
    },
    "max_specific_range": [
        {**dict.fromkeys(("wd_quantile", "wd_t", "alt_ft", "sat_k", "mass_kg"), NUM)}
        | _SR
    ],
    "temperature_test": {
        "point": STR,
        "alt_ft": NUM,
        "mass_kg": NUM,
        "rows": [{"dT_isa_k": NUM, "sat_k": NUM} | _SR],
        "shift_mach_max_sr": NUM,
    },
    "constraints": {
        "grid_points": NUM,
        "inside_support": NUM,
        **dict.fromkeys(
            ("inside", "outside", "all"),
            dict.fromkeys(
                (
                    "mach_convexity_violation_pct",
                    "mass_monotone_violation_pct",
                    "energy_monotone_violation_pct",
                ),
                NUM,
            ),
        ),
        "range_boundary": dict.fromkeys(
            ("points", "mach", "violation_pct", "min_mach_times_dlnF_dM"), NUM
        ),
        "drag_rise": {
            **dict.fromkeys(
                (
                    "points",
                    "mach",
                    "cl_max",
                    "tolerance",
                    "violation_pct",
                    "min_margin",
                    "table_sha256_match",
                ),
                NUM,
            ),
            "aerox_version": STR,
        },
    },
    "roughness": {
        "coordinates": STR,
        **dict.fromkeys(
            (
                "sd",
                "points",
                "inside_support",
                "mean_hessian_frobenius_inside",
                "mean_hessian_frobenius_outside",
                "mean_hessian_frobenius_all",
            ),
            NUM,
        ),
    },
    "pair_test": {k: _PAIR for k in PAIR_SETS},
    "smoothness": {
        **dict.fromkeys(
            _PHASES,
            dict.fromkeys(
                (
                    "pairs",
                    "rms_diff_pred_kgh",
                    "rms_diff_meas_kgh",
                    "ratio_pred_to_meas",
                ),
                NUM,
            ),
        ),
        "cruise_runs": NUM,
        "cruise_sd_ratio_median": NUM,
        "steady_windows": dict.fromkeys(
            ("windows", "sd_ratio_median", "sd_pred_median_kgh", "sd_meas_median_kgh"),
            NUM,
        ),
    },
    "noise_sensitivity": {
        "blocks": NUM,
        "noise_sd": {
            **dict.fromkeys(
                ("mach", "alt_ft", "mass_rel", "energy_rate", "sat_k"), NUM
            ),
            "energy_split": STR,
        },
        **{f"rms_rel_change_pct_{k}": NUM for k in (*NOISE_INPUTS, "joint")},
    },
    "envelope": {
        "source": STR,
        "evaluated_outside_pct": dict.fromkeys(_PHASES, NUM),
        "description": ENVELOPE,
    },
    "coverage": {"evaluated": _COVERAGE, "train": _COVERAGE},
    "config": _fields(ValidationConfig),
    "constraint_settings": dict.fromkeys(CONSTRAINT_SETTINGS, NUM)
    | {"drag_rise": DRAG_RISE},
    "prep": _fields(PrepConfig),
    "surveillance": {
        "variant": STR,
        "blocks": NUM,
        "blocks_without_inputs": NUM,
        "recorded_inputs": dict.fromkeys((*_PHASES, "cruise"), _ACC),
        "surveillance_inputs": dict.fromkeys((*_PHASES, "cruise"), _ACC),
    },
}
