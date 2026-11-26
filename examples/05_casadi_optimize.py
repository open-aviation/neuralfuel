# Run: uv run --group casadi examples/05_casadi_optimize.py
r"""Import the shared A320 model into CasADi and solve a small optimization.

The ONNX model becomes a symbolic CasADi function (neuralfuel.inference.
casadi_function). The script finds the Mach and the pressure altitude of maximum
specific air range (true airspeed / fuel flow) in steady level flight at a given mass,
with IPOPT. The bounds come from the valid envelope in the ONNX metadata: the Mach,
altitude and W/delta range of level flight (both bounds), and the cruise limits of
W/delta and of the lift coefficient. The fixed inputs (mass, ISA deviation, zero
energy rate) are checked against the level-flight ranges first; a mass outside them
gives no optimum. The result is compared with a grid search with the PyTorch model.

The first part checks the import: it draws --states random states inside the
envelope (from the ONNX metadata, or from a validation report with --report for a
file without one) and prints the largest differences of ln fuel flow, its gradient
and its Hessian between CasADi and PyTorch in float64, and the CasADi evaluation time
per state of the function and of the gradient with the Hessian
(neuralfuel.inference.casadi_check). --expand checks the SX form.

Run from the repository root:

    uv run --group casadi examples/05_casadi_optimize.py
    uv run --group casadi examples/05_casadi_optimize.py \
        --model run/model.pt --onnx run/model.onnx

Run time: about 5 seconds.
"""

import argparse
import json

import casadi as ca
import numpy as np

from neuralfuel import atmos
from neuralfuel.inference import casadi_check, casadi_function, onnx_metadata, predict
from neuralfuel.model import load
from neuralfuel.support import random_inputs

DT_ISA = 10.0  # K; with the fixed temperature rule the optimum does not depend on it


def isa_delta_ca(h_ft):
    """ISA pressure ratio as a CasADi expression of pressure altitude (ft)."""
    h = h_ft * atmos.FT
    low = (1 - atmos.LAPSE * h / atmos.T0) ** atmos.EXP_P
    high = atmos.DELTA_TROP * ca.exp(
        -atmos.G0 * (h - atmos.H_TROP) / (atmos.R_GAS * atmos.T_TROP)
    )
    return ca.if_else(h <= atmos.H_TROP, low, high)


def isa_temperature_ca(h_ft):
    return ca.fmax(atmos.T0 - atmos.LAPSE * h_ft * atmos.FT, atmos.T_TROP)


def fixed_inputs_inside(env, mass_kg):
    """True if the fixed inputs (mass, ISA deviation DT_ISA, sin_gamma = acc_in = 0)
    lie inside the level-flight ranges of the envelope."""
    lv = env["phases"]["level"]
    fixed = (("mass_kg", mass_kg), ("dT_isa_k", DT_ISA), ("sin_gamma", 0.0))
    return all(lv[k][0] <= v <= lv[k][1] for k, v in fixed + (("acc_in", 0.0),))


def best_range(f, env, mass_kg):
    """(Mach, altitude ft, SAR in NM per 1000 kg, IPOPT status) at mass_kg; NaN and
    the status 'outside the envelope' if the fixed inputs are outside it."""
    if not fixed_inputs_inside(env, mass_kg):
        return np.nan, np.nan, np.nan, "outside the envelope"
    lv, cr = env["phases"]["level"], env["cruise"]
    opti = ca.Opti()
    mach = opti.variable()
    h = opti.variable()  # pressure altitude, ft
    delta = isa_delta_ca(h)
    sat = isa_temperature_ca(h) + DT_ISA
    x = ca.horzcat(
        mach, ca.log(delta), np.log(mass_kg), 0, 0, ca.log(sat / atmos.T0)
    )  # level flight: sin_gamma = acc_in = 0
    ff = ca.exp(f(x))  # kg/h
    tas_kt = mach * ca.sqrt(atmos.GAMMA * atmos.R_GAS * sat) / atmos.KT
    sar = 1000 * tas_kt / ff
    opti.minimize(-sar)
    lo = max(lv["alt_ft"][0], env["phase_rule"]["cruise_alt_ft"])
    opti.subject_to(opti.bounded(lv["mach"][0], mach, lv["mach"][1]))
    opti.subject_to(opti.bounded(lo, h, lv["alt_ft"][1]))
    wd = mass_kg / delta / 1000  # W/delta, t
    opti.subject_to(wd >= lv["wd_t"][0])
    opti.subject_to(wd <= lv["wd_t"][1])
    if cr["max_wd_t"] is not None:  # no cruise limit without steady windows
        opti.subject_to(wd <= cr["max_wd_t"])
    if cr["max_cl"] is not None:
        cl = (
            2
            * mass_kg
            * atmos.G0
            / (atmos.GAMMA * atmos.P0 * delta * mach**2 * env["wing_area_m2"])
        )
        opti.subject_to(cl <= cr["max_cl"])
    opti.set_initial(mach, 0.72)
    opti.set_initial(h, 30000.0)
    # bound_relax_factor 0: the solution keeps the envelope bounds exactly
    ipopt = {"print_level": 0, "sb": "yes", "bound_relax_factor": 0.0}
    opti.solver("ipopt", {"print_time": False}, ipopt)
    sol = opti.solve()
    status = sol.stats()["return_status"]
    return float(sol.value(mach)), float(sol.value(h)), float(sol.value(sar)), status


def grid_best(model, mass_kg, env):
    """Largest SAR on a grid (Mach step 0.001, altitude step 50 ft), inside the
    envelope, with the PyTorch model."""
    lv = env["phases"]["level"]
    lo = max(lv["alt_ft"][0], env["phase_rule"]["cruise_alt_ft"])
    M, H = np.meshgrid(
        np.arange(0.60, lv["mach"][1], 0.001), np.arange(lo, lv["alt_ft"][1], 50.0)
    )
    M, H = M.ravel(), H.ravel()
    sat = atmos.isa_temperature(H) + DT_ISA
    z = np.zeros(len(M))
    x = np.column_stack(
        [M, np.log(atmos.isa_delta(H)), np.full(len(M), np.log(mass_kg)), z, z]
        + [np.log(sat / atmos.T0)]
    )
    ff, outside = predict(model, x, outside=True)
    sar = np.where(outside, -np.inf, 1000 * M * atmos.sound_speed(sat) / atmos.KT / ff)
    i = int(np.argmax(sar))
    return M[i], H[i], sar[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="models/a320/nf-a320-cfm56_5b.pt")
    ap.add_argument("--onnx", default="models/a320/nf-a320-cfm56_5b.onnx")
    ap.add_argument("--states", type=int, default=1000, help="states of the check")
    ap.add_argument("--seed", type=int, default=0, help="seed of the check states")
    ap.add_argument(
        "--report", help="validation report with the envelope, for a file without one"
    )
    ap.add_argument("--expand", action="store_true", help="check as SX expressions")
    args = ap.parse_args()

    model = load(args.model)
    env = onnx_metadata(args.onnx)["envelope"]
    if env is None:
        if not args.report:
            ap.error("the ONNX file has no envelope; give --report")
        with open(args.report) as fh:
            env = json.load(fh)["envelope"]["description"]
    x = random_inputs(env, args.states, seed=args.seed)
    c = casadi_check(model, args.onnx, x, expand=args.expand)
    print(f"CasADi import against PyTorch (float64), {c['states']} states:")
    for k in ("ln_ff", "gradient", "hessian"):
        print(f"  largest difference of {k:9s} {c[f'max_abs_diff_{k}']:.2e}")
    print(
        f"  CasADi time per state: function {1e6 * c['time_per_function_eval_s']:.1f} "
        f"us, gradient and Hessian {1e6 * c['time_per_hessian_eval_s']:.1f} us"
    )

    f = casadi_function(args.onnx)
    print(f"\nmaximum specific air range, level flight, ISA {DT_ISA:+.0f} K")
    print(
        f"  {'mass kg':>8s} | {'Mach':>6s} {'alt ft':>7s} {'SAR':>6s} {'IPOPT':>16s} "
        f"| grid: {'Mach':>6s} {'alt ft':>7s} {'SAR':>6s}"
    )
    for mass in (56000.0, 62000.0, 68000.0):
        m, h, s, status = best_range(f, env, mass)
        gm, gh, gs = grid_best(model, mass, env)
        print(
            f"  {mass:8.0f} | {m:6.3f} {h:7.0f} {s:6.1f} {status:>16s} | "
            f"      {gm:6.3f} {gh:7.0f} {gs:6.1f}"
        )
    print("  SAR in NM per 1000 kg; the bounds come from the valid envelope")


if __name__ == "__main__":
    main()
