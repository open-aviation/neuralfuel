"""Quality check of a contribution against the shared model it continued from.

A maintainer runs it without any flight data. Inputs: the contribution, the base
model (the shared model that the contribution continued from) and the run record of
the contribution. The run record holds the declared region of the contribution: the
envelope of its local training data ('local_envelope'). A run record must belong to
the contribution (weights_sha256) and name the base (init_weights_sha256);
otherwise check_model raises ValueError (weights) or the check 'base' fails. The run
record is checked with neuralfuel.aggregate.record_problems, the same check as in
`contribute` and `merge`.

This is a check for honest mistakes: wrong units or scale factors in the adapter,
training without the protection against forgetting, diverged training. It is not a
security barrier. It cannot detect training on inaccurate airspeed, and it cannot
detect a false statement of the data region. The maintainers review each pull request
and compare the model with reference models (neuralfuel.reference).

Checks (value, limits, pass / warn / fail):
  finite             all parameters and outputs finite (yes or no)
  compatible         architecture, inputs, temperature rule, aircraft settings and
                     standardization constants equal to those of the base (yes or no)
  base               the run record names the base, and the meta entry of the model
                     file names the same base (only with a run record)
  declared_region    the run record has the declared region, its counts agree with
                     the run record and the envelope of the contribution is the union
                     of the base envelope and the declared region (only with a run
                     record)
  protection         teacher weight of the protection against forgetting in the run
                     record; 0 fails (only with a run record)
  drag_table         the contribution has the P7 settings of the base
                     (constraints.drag_rise, without 'aerox_version', which continued
                     training sets to the installed version), and the installed aerox
                     rebuilds the stored drag-slope table of P7 of the base
                     (table_sha256); with the same settings the contribution has the
                     same table. Otherwise the P7 values of constraint_delta use a
                     table or settings that the models were not trained with (only
                     for a model with P7 settings). The settings are compared first,
                     so the table of the contribution is never built
  constraint_delta   largest increase (contribution - base, percentage points) of the
                     violation rates of P1, P2, P3 on the regular constraint grid of
                     the library (neuralfuel.validate, the whole grid), of P6 at
                     the MMO and, for a base trained with the drag-rise term, of P7
                     on its grid
  fuel_level         |median(fuel flow of the contribution / fuel flow of the base)
                     - 1| (%) on the cruise grid (CRUISE_GRID) inside the envelope of
                     the base and inside the declared region (without a run record:
                     inside the envelope of the base); a wrong unit or a scale factor
                     moves the level. The row gives the number of these states
                     ('states'); with none, the check is not measured (no value, a
                     warning)
  region_type        share of states drawn from the declared region (or, without a
                     run record, the envelope of the contribution) with a mass outside
                     0.9 x lowest to 1.1 x highest mass of the type or a Mach above
                     MMO + 0.02
  change_outside_mean, change_outside_max
                     100 |contribution / base - 1| (%) of the fuel flow at the states
                     inside the envelope of the base and outside the declared region
                     (65,536 collocation states over the aircraft envelope and 32,768
                     states drawn inside the envelope of the base); needs the run
                     record; fewer than MIN_OUTSIDE_STATES such states warn, because
                     then the two values say little
  mach_mean          |mean(delta s)|: the absolute value of the mean of the signed
                     change delta s of s = d ln F / dM on the cruise grid (CRUISE_GRID)
                     inside the envelope of the base (changes of opposite sign cancel)
  mach_max           max |delta s|: the largest absolute change on the same grid
  best_mach          largest absolute change of the Mach of maximum specific air range
                     (search MMO - 0.22 to MMO, step 0.001) at the (altitude, mass)
                     points of the grid whose state at MMO - 0.06 is inside the base
  smoothness         ratio contribution / base of the mean Frobenius norm of the
                     Hessian of ln F (standardized state coordinates) on the
                     constraint grid states inside the envelope of the base

Limits: LIMITS, one place. 'warn' and 'fail' are upper limits; a value above 'fail'
fails, above 'warn' warns. A non-finite value (NaN or infinite) fails. A check without
a value (a quantity that could not be measured, for example the change outside the
declared region without a run record) warns. The limits are the two rules of the
acceptance gate applied to 51 good simulated contributions on the A320 dataset,
trained and checked with this package (with the statistics above, for mach_mean
the absolute value of the signed mean): 'fail' = 1.25 x the largest good value (rule
'max', the loose limits), 'warn' = 1.25 x the 90 % quantile (rule 'q90'). They are
limits for that dataset and model size; another type or data source may need other
limits.

`cache`: a dict that keeps the results of the base model between calls with the same
base (for several contributions against one base).
"""

import copy

import numpy as np
import pandas as pd
import torch

from . import atmos
from . import drag as dragmod
from .aggregate import differences, finite_state, record_problems
from .losses import envelope, model_loss_config, sample_collocation, still_air_inputs
from .model import FuelFlowNet, state_hash
from .support import in_envelope, random_inputs
from .validate import (
    ValidationConfig,
    constraint_grid,
    constraints,
    hessian,
    stored_drag_table,
)

# name: (warn, fail), upper limits. The two rules of the acceptance gate on 51 good
# simulated contributions (A320 dataset, six scenarios, three seeds, trained and
# checked with this package): fail = 1.25 x the largest good value (rule 'max', the
# loose limits), warn = 1.25 x the 90 % quantile (rule 'q90'). region_type: the fixed
# gate limit.
LIMITS = {
    "constraint_delta": (2.393, 9.436),
    "fuel_level": (0.4046, 1.102),
    "region_type": (None, 0.05),
    "change_outside_mean": (0.7576, 0.9621),
    "change_outside_max": (20.11, 29.49),
    "mach_mean": (0.08061, 0.1395),
    "mach_max": (0.2558, 0.7488),
    "best_mach": (0.01125, 0.01625),
    "smoothness": (1.27, 1.285),
}
# lower limit: fewer states inside the base envelope and outside the declared region
# make the change_outside_* checks warn (about 1 % of the 98,304 test states)
MIN_OUTSIDE_STATES = 1000
LIMITS["outside_states"] = (MIN_OUTSIDE_STATES, None)
# cruise grid as fractions of the aircraft settings, at the ISA deviation in the
# middle of the level-flight range of the base envelope (rounded to 1 K); for the
# stored settings of the shared A320 models (MMO 0.82, 42.6 to 78 t, up to 39,800 ft)
# it is Mach 0.70 to 0.80, 29,000 to 39,000 ft, 47.9 to 74.8 t, at ISA + 10 K. The
# limits above were measured on this grid.
CRUISE_GRID = {
    "dmach_below_mmo": (0.12, 0.02),
    "mach_step": 0.01,
    "alt_below_max_ft": (10800.0, 800.0),
    "alt_step_ft": 1000.0,
    "mass_fraction": (0.15, 0.91),
    "mass_points": 6,
    "best_mach_span": 0.22,
    "best_mach_test": 0.06,
}
N_POOL, N_SHARED, N_REGION, SEED = 65536, 32768, 2000, 4242


def _m64(model):
    return copy.deepcopy(model).cpu().double().eval()


def _ln(model64, x):
    with torch.no_grad():
        return model64(torch.as_tensor(np.asarray(x, dtype=float))).numpy()


def _status(name, value):
    warn, fail = LIMITS[name]
    if value is None:  # not measured
        return "warn"
    if not np.isfinite(value):  # measured, not finite
        return "fail"
    if fail is not None and value > fail:
        return "fail"
    if warn is not None and value > warn:
        return "warn"
    return "pass"


def _row(name, value, note=""):
    warn, fail = LIMITS[name]
    return {
        "check": name,
        "value": None if value is None else float(value),
        "warn": warn,
        "fail": fail,
        "status": _status(name, value),
        "note": note,
    }


def cruise_grid(ac):
    g = CRUISE_GRID
    lo, hi = ac.mmo - g["dmach_below_mmo"][0], ac.mmo - g["dmach_below_mmo"][1]
    mach = np.round(np.arange(lo, hi + 1e-9, g["mach_step"]), 4)
    a_hi = ac.alt_ft[1] - g["alt_below_max_ft"][1]
    a_lo = ac.alt_ft[1] - g["alt_below_max_ft"][0]
    alt = np.arange(a_lo, a_hi + 1e-6, g["alt_step_ft"])
    m0, m1 = ac.mass_kg
    f0, f1 = g["mass_fraction"]
    mass = np.round(np.linspace(m0 + f0 * (m1 - m0), m0 + f1 * (m1 - m0), 6), -2)
    return mach, alt, mass


def level_inputs(mach, alt, mass, dT, sg=0.0, acc=0.0):
    """Model inputs of steady level flight (n, 6); sg and acc are the values of
    sin_gamma and acc_in (0, or the nearest value inside the level-flight range of
    an envelope)."""
    mach, alt, mass = (np.asarray(v, dtype=float).ravel() for v in (mach, alt, mass))
    sat = atmos.isa_temperature(alt) + dT
    z = np.ones(len(mach))
    return np.column_stack(
        [
            mach,
            np.log(atmos.isa_delta(alt)),
            np.log(mass),
            sg * z,
            acc * z,
            np.log(sat / atmos.T0),
        ]
    )


def dlnf_dm(model64, x):
    q = torch.tensor(np.asarray(x, dtype=float), requires_grad=True)
    return torch.autograd.grad(model64(q).sum(), q)[0][:, 0].numpy()


def best_mach(model64, ac, alt, mass, dT, sg=0.0, acc=0.0):
    g = CRUISE_GRID
    mg = np.round(np.arange(ac.mmo - g["best_mach_span"], ac.mmo + 1e-9, 0.001), 6)
    x = level_inputs(
        np.tile(mg, len(alt)),
        np.repeat(alt, len(mg)),
        np.repeat(mass, len(mg)),
        dT,
        sg,
        acc,
    )
    f = _ln(model64, x).reshape(len(alt), len(mg))
    i = np.argmax(np.log(mg)[None, :] - f, axis=1)
    return mg[i], i == len(mg) - 1


def _grid():
    """States of the library constraint grid (the whole grid, no support flag)."""
    cols = ["mach", "alt_ft", "mass_kg", "sin_gamma", "acc_in", "dT_isa"]
    return pd.DataFrame({c: np.zeros(0) for c in cols})


def _violations(model64, p, h, g, ac, lc, drag=None):
    c = constraints(
        model64,
        p,
        np.zeros(len(p), bool),
        g,
        h,
        ValidationConfig(),
        lc,
        ac,
        drag,
    )
    out = {
        k: c["all"][f"{k}_violation_pct"] or 0.0
        for k in ("mach_convexity", "mass_monotone", "energy_monotone")
    }
    out["range_boundary"] = c["range_boundary"]["violation_pct"]
    if drag:
        out["drag_rise"] = c["drag_rise"]["violation_pct"]
    return out


def region_type_share(region, ac, n=N_REGION):
    x = random_inputs(region, n, seed=0, phase="all")
    mass = np.exp(x[:, 2])
    bad = (mass < 0.9 * ac.mass_kg[0]) | (mass > 1.1 * ac.mass_kg[1])
    bad |= x[:, 0] > ac.mmo + 0.02
    return float(bad.mean())


def p7_settings(d):
    """The P7 settings (constraints.drag_rise) without 'aerox_version': continued
    training records the installed aerox version, and table_sha256 already tells
    that the table is the same."""
    return {k: v for k, v in d.items() if k != "aerox_version"} if d else None


def _flag(name, ok, note, value=None):
    return {
        "check": name,
        "value": bool(ok) if value is None else value,
        "warn": None,
        "fail": None,
        "status": "pass" if ok else "fail",
        "note": note,
    }


def check_model(
    contrib: FuelFlowNet,
    base: FuelFlowNet,
    run_record=None,
    declared=None,
    meta=None,
    cache=None,
) -> dict:
    """Run the checks. Returns {'checks': [...], 'ok': no check failed, ...}.

    run_record: the run record of the contribution; its 'local_envelope' is the
    declared region. A record of other weights raises ValueError. declared: a
    declared region without a run record (for example the union of the regions of
    merged contributions). meta: the meta entry of the contribution file
    (neuralfuel.model.read_meta), compared with the run record. cache: a dict that
    keeps the results of the base between calls (see the module docstring).
    """
    rows = []
    out = {
        "contribution_weights_sha256": state_hash(contrib),
        "base_weights_sha256": state_hash(base),
        "checks": rows,
    }
    if run_record is not None:
        problems = record_problems(contrib, run_record, base, meta)
        if problems["weights"]:
            raise ValueError("; ".join(problems["weights"]))
        declared = run_record.get("local_envelope")
    finite = finite_state(contrib)
    rows.append(
        _flag(
            "finite", finite, "" if finite else "parameters with NaN or infinite values"
        )
    )
    diff = differences(base, contrib)
    rows.append(_flag("compatible", not diff, "; ".join(diff)))
    if run_record is not None:
        init = run_record.get("init_weights_sha256")
        note = "; ".join(problems["base"])
        rows.append(_flag("base", not note, note))
        note = "; ".join(problems["region"])
        rows.append(_flag("declared_region", not note, note))
        tw = float((run_record.get("protection") or {}).get("teacher_weight") or 0.0)
        if init is not None:  # a continued model
            rows.append(
                _flag(
                    "protection",
                    tw > 0,
                    f"teacher weight {tw:g}"
                    + ("" if tw > 0 else ": the protection against forgetting was off"),
                    value=tw,
                )
            )
    # results of the base: kept in `cache` between calls with the same base
    memo = {}
    if cache is not None:
        memo = cache.setdefault(out["base_weights_sha256"], {})

    def of_base(key, fn):
        if key not in memo:
            memo[key] = fn()
        return memo[key]

    bd, cd = ((m.constraints or {}).get("drag_rise") for m in (base, contrib))
    if bd or cd:
        for name, d in (("base", bd), ("contribution", cd)):
            if d and not d.get("table_sha256"):
                raise ValueError(
                    f"the drag-slope table of P7 of the {name} model cannot be "
                    "checked: its P7 settings (constraints.drag_rise) have no "
                    "table_sha256"
                )
        # the settings of the contribution are compared first: only the table of the
        # base is rebuilt, so the contribution does not set the size of a table
        kept = p7_settings(cd) == p7_settings(bd)
        installed = f" with the installed aerox {dragmod.aerox_version()}"
        notes, base_ok = [], True
        if bd:
            base_ok = of_base(
                "drag_table", lambda: stored_drag_table(bd, "base model")[1]
            )
            notes.append(
                f"base: table of aerox {bd.get('aerox_version')} "
                + ("reproduced" if base_ok else "not reproduced")
                + installed
            )
        same = kept and base_ok
        if kept:  # the same table as the base (same grid and table_sha256)
            notes.append(
                f"contribution: table of aerox {cd.get('aerox_version')} "
                + ("reproduced" if same else "not reproduced")
                + installed
            )
        else:
            notes.append(
                "the P7 settings of the contribution (constraints.drag_rise) "
                "differ from those of the base; its table was not rebuilt"
            )
        rows.append(_flag("drag_table", same, "; ".join(notes)))
    if not finite or diff:
        out["ok"] = False
        out["warnings"] = 0
        return out
    ac = base.aircraft
    if ac is None or base.envelope is None:
        raise ValueError("the base model needs aircraft settings and an envelope")
    c64, s64 = _m64(contrib), _m64(base)
    env_s = base.envelope
    lc = model_loss_config(base)
    drag = bd

    # constraint grid: violations and smoothness
    p, _ = constraint_grid(_grid(), ValidationConfig(), envelope(ac, lc))
    _, gc, hc = hessian(c64, p)
    _, gs, hs = of_base("hessian", lambda: hessian(s64, p))
    vc = _violations(c64, p, hc, gc, ac, lc, drag)
    vs = of_base("violations", lambda: _violations(s64, p, hs, gs, ac, lc, drag))
    d = {k: vc[k] - vs[k] for k in vc}
    rows.append(
        _row(
            "constraint_delta",
            max(d.values()),
            ", ".join(f"{k} {v:+.2f}" for k, v in d.items()),
        )
    )
    x_grid = still_air_inputs(p.numpy())
    ins_grid = in_envelope(env_s, x_grid)
    sd = np.ones(5)
    sd[:4] = base.x_std.detach().cpu().numpy().astype(float)[:4]
    fro_c = np.sqrt(((hc * sd[None, :, None] * sd[None, None, :]) ** 2).sum((1, 2)))
    fro_s = of_base(
        "fro",
        lambda: np.sqrt(
            ((hs * sd[None, :, None] * sd[None, None, :]) ** 2).sum((1, 2))
        ),
    )
    if ins_grid.any():
        rows.append(
            _row(
                "smoothness",
                fro_c[ins_grid].mean() / fro_s[ins_grid].mean(),
                f"{int(ins_grid.sum())} grid states inside the envelope of the base",
            )
        )
    else:
        rows.append(_row("smoothness", None, "no grid state inside the base envelope"))

    # fuel level in steady cruise inside the envelope of S (and the declared region)
    g = CRUISE_GRID
    lv = env_s["phases"].get("level") or {}
    dT = float(np.round(0.5 * sum(lv["dT_isa_k"]))) if lv.get("dT_isa_k") else 0.0
    sg = float(np.clip(0.0, *lv["sin_gamma"])) if lv.get("sin_gamma") else 0.0
    acc = float(np.clip(0.0, *lv["acc_in"])) if lv.get("acc_in") else 0.0
    mach, alt, mass = cruise_grid(ac)
    M, A, W = np.meshgrid(mach, alt, mass, indexing="ij")
    xc = level_inputs(M, A, W, dT, sg, acc)
    ins = in_envelope(env_s, xc)
    sel = ins & in_envelope(declared, xc) if declared is not None else ins
    ratio = np.exp(_ln(c64, xc) - of_base("ln_cruise", lambda: _ln(s64, xc)))
    out["finite_outputs"] = bool(np.isfinite(ratio).all())
    if not out["finite_outputs"]:
        rows[0].update(
            value=False, status="fail", note="outputs with NaN or infinite values"
        )
    where = "the base envelope" + (
        " and the declared region" if declared is not None else ""
    )
    if sel.any():
        med = float(np.median(ratio[sel]))
        note = (
            f"median fuel flow contribution / base {med:.4f} at {int(sel.sum())} "
            f"cruise grid states inside {where}"
        )
        rows.append(_row("fuel_level", 100 * abs(med - 1), note))
    else:
        note = f"not measured: no cruise grid state inside {where}"
        rows.append(_row("fuel_level", None, note))
    rows[-1]["states"] = int(sel.sum())

    # declared region
    region = declared if declared is not None else contrib.envelope
    if region is not None:
        rows.append(
            _row(
                "region_type",
                region_type_share(region, ac),
                "declared region"
                if declared is not None
                else "envelope of the contribution",
            )
        )
    else:
        rows.append(_row("region_type", None, "no declared region and no envelope"))

    # change outside the declared region
    def pool():
        gen = torch.Generator().manual_seed(SEED)
        pts = sample_collocation(N_POOL, gen, "cpu", envelope(ac, lc))
        x = np.concatenate(
            [
                still_air_inputs(pts.numpy()),
                random_inputs(env_s, N_SHARED, seed=SEED, phase="all"),
            ]
        )
        return x, in_envelope(env_s, x), _ln(s64, x)

    if declared is not None:
        xp, in_base, ln_s = of_base("pool", pool)
        mask = in_base & ~in_envelope(declared, xp)
        dd = 100 * np.abs(np.expm1(_ln(c64, xp[mask]) - ln_s[mask]))
        n_out = int(mask.sum())
        note = f"{n_out} states inside the base envelope, outside the declared region"
        mean = float(dd.mean()) if len(dd) else 0.0
        mx = float(dd.max()) if len(dd) else 0.0
        rows.append(_row("change_outside_mean", mean, note))
        rows.append(_row("change_outside_max", mx, note))
        few = n_out < MIN_OUTSIDE_STATES
        rows.append(
            {
                "check": "outside_states",
                "value": float(n_out),
                "warn": float(MIN_OUTSIDE_STATES),
                "fail": None,
                "status": "warn" if few else "pass",
                "note": (
                    f"lower limit: below {MIN_OUTSIDE_STATES} states the "
                    "change_outside checks say little"
                    if few
                    else ""
                ),
            }
        )
    else:
        for k in ("change_outside_mean", "change_outside_max"):
            rows.append(_row(k, None, "no run record: the declared region is unknown"))

    # Mach sensitivity and best-range Mach in cruise
    if ins.any():
        # the same at any sg, acc
        ds = dlnf_dm(c64, xc[ins]) - of_base("s_cruise", lambda: dlnf_dm(s64, xc[ins]))
        n = int(ins.sum())
        j = int(np.argmax(np.abs(ds)))
        rows.append(
            _row(
                "mach_mean", abs(ds.mean()), f"mean change {ds.mean():+.4f}, {n} states"
            )
        )
        rows.append(
            _row("mach_max", abs(ds[j]), f"largest change {ds[j]:+.4f}, {n} states")
        )
    else:
        for k in ("mach_mean", "mach_max"):
            rows.append(_row(k, None, "no cruise grid state inside the base envelope"))
    A2, W2 = (v.ravel() for v in np.meshgrid(alt, mass, indexing="ij"))
    test = level_inputs(
        np.full(len(A2), ac.mmo - g["best_mach_test"]), A2, W2, dT, sg, acc
    )
    ok = in_envelope(env_s, test)
    if ok.any():
        bc, lim = best_mach(c64, ac, A2[ok], W2[ok], dT, sg, acc)
        bs, _ = of_base(
            "best_mach", lambda: best_mach(s64, ac, A2[ok], W2[ok], dT, sg, acc)
        )
        rows.append(
            _row(
                "best_mach",
                np.abs(bc - bs).max(),
                f"{int(ok.sum())} points, {int(lim.sum())} of the contribution at "
                "the MMO",
            )
        )
    else:
        rows.append(_row("best_mach", None, "no cruise point inside the base envelope"))
    out["ok"] = not any(r["status"] == "fail" for r in rows)
    out["warnings"] = sum(r["status"] == "warn" for r in rows)
    return out
