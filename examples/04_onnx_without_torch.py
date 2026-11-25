# Run: uv run examples/04_onnx_without_torch.py
"""Use the shared A320 model with onnxruntime and numpy only.

This script imports neither torch nor neuralfuel. It shows what a tool in another
environment needs: the ISA pressure ratio, the order of the six inputs, the output
(ln of fuel flow in kg/h), and the aircraft settings and the valid envelope from the
ONNX metadata. The envelope test below follows neuralfuel.support.in_envelope.

Run from the repository root:

    uv run examples/04_onnx_without_torch.py

Run time: about one second.
"""

import json
import math

import numpy as np
import onnxruntime as ort

ONNX = "models/a320/nf-a320-cfm56_5b.onnx"

# ISA constants (the values of neuralfuel.atmos)
T0, P0, G0, R, GAMMA = 288.15, 101325.0, 9.80665, 287.05287, 1.4
LAPSE, H_TROP, T_TROP, FT = 0.0065, 11000.0, 216.65, 0.3048
EXP = G0 / (LAPSE * R)
DELTA_TROP = (T_TROP / T0) ** EXP


def isa_delta(alt_ft):
    """ISA pressure ratio p / p0 at pressure altitude (ft)."""
    h = np.asarray(alt_ft, dtype=float) * FT
    low = (1 - LAPSE * np.minimum(h, H_TROP) / T0) ** EXP
    high = DELTA_TROP * np.exp(-G0 * (h - H_TROP) / (R * T_TROP))
    return np.where(h <= H_TROP, low, high)


def isa_temperature(alt_ft):
    h = np.asarray(alt_ft, dtype=float) * FT
    return np.where(h <= H_TROP, T0 - LAPSE * h, T_TROP)


def model_inputs(mach, alt_ft, mass_kg, sat_k, vs_fpm=0.0, acc_in=0.0):
    """(n, 6) float32: Mach, ln delta, ln mass, sin_gamma, acc_in, ln theta."""
    mach, alt_ft, mass_kg, sat_k, vs_fpm, acc_in = np.broadcast_arrays(
        *(
            np.asarray(v, dtype=float)
            for v in (mach, alt_ft, mass_kg, sat_k, vs_fpm, acc_in)
        )
    )
    tas_ms = mach * np.sqrt(GAMMA * R * sat_k)
    sin_gamma = vs_fpm * FT / 60.0 / tas_ms
    x = np.stack(
        [
            mach,
            np.log(isa_delta(alt_ft)),
            np.log(mass_kg),
            sin_gamma,
            acc_in,
            np.log(sat_k / T0),
        ],
        axis=-1,
    )
    return x.reshape(-1, 6).astype(np.float32)


def in_envelope(env, mach, alt_ft, mass_kg, sat_k, vs_fpm=0.0, acc_in=0.0):
    """True for the states inside the valid envelope (as neuralfuel.support)."""
    mach, alt_ft, mass_kg, sat_k, vs_fpm, acc_in = (
        np.ravel(v)
        for v in np.broadcast_arrays(mach, alt_ft, mass_kg, sat_k, vs_fpm, acc_in)
    )
    delta = isa_delta(alt_ft)
    tas_ms = mach * np.sqrt(GAMMA * R * sat_k)
    s = {
        "mach": mach,
        "alt_ft": alt_ft,
        "mass_kg": mass_kg,
        "wd_t": mass_kg / delta / 1000.0,  # W/delta: mass / pressure ratio, in t
        "dT_isa_k": sat_k - isa_temperature(alt_ft),
        "sin_gamma": vs_fpm * FT / 60.0 / tas_ms,
        "acc_in": acc_in,
    }
    rule = env["phase_rule"]
    lim = rule["level_fpm"]
    phase = np.where(vs_fpm > lim, "climb", np.where(vs_fpm < -lim, "descent", "level"))
    ok = np.zeros(len(mach), bool)
    for p in ("climb", "level", "descent"):
        box, sel = env["phases"].get(p), phase == p
        if not box or not box["blocks"] or not sel.any():
            continue
        inside = np.ones(int(sel.sum()), bool)
        for k, v in s.items():
            lo, hi = box[k]
            inside &= (v[sel] >= lo) & (v[sel] <= hi)
        ok[sel] = inside
    cruise = (phase == "level") & (alt_ft > rule["cruise_alt_ft"])
    cr = env["cruise"]
    if cr["max_wd_t"] is not None:
        ok &= ~cruise | (s["wd_t"] <= cr["max_wd_t"])
    if cr["max_cl"] is not None and env.get("wing_area_m2"):
        cl = 2 * mass_kg * G0 / (GAMMA * P0 * delta * mach**2 * env["wing_area_m2"])
        ok &= ~cruise | (cl <= cr["max_cl"])
    return ok


def main():
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    aircraft = json.loads(meta["neuralfuel_aircraft"])
    env = json.loads(meta["neuralfuel_envelope"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"{ONNX}")
    print(f"  input  '{inp.name}' {inp.shape} {inp.type}")
    print(f"  output '{out.name}' {out.shape} {out.type}")
    print(f"  metadata keys: {sorted(meta)}")
    print(
        f"  aircraft: type {aircraft['type']}, engine {aircraft['engine']}, MMO "
        f"{aircraft['mmo']}, VMO {aircraft['vmo_kt']} kt, mass {aircraft['mass_kg']} "
        f"kg, altitude {aircraft['alt_ft']} ft"
    )
    lv = env["phases"]["level"]
    print(
        f"  envelope, level flight: Mach {lv['mach'][0]:.3f} to {lv['mach'][1]:.3f}, "
        f"altitude {lv['alt_ft'][0]:.0f} to {lv['alt_ft'][1]:.0f} ft, ISA deviation "
        f"{lv['dT_isa_k'][0]:+.1f} to {lv['dT_isa_k'][1]:+.1f} K"
    )

    # cruise at 37,000 ft, ISA +10 K, three masses; and one climb state
    states = [
        (0.78, 37000.0, 56000.0, float(isa_temperature(37000.0)) + 10.0, 0.0),
        (0.78, 37000.0, 62000.0, float(isa_temperature(37000.0)) + 10.0, 0.0),
        (0.78, 37000.0, 72000.0, float(isa_temperature(37000.0)) + 10.0, 0.0),
        (0.70, 25000.0, 65000.0, float(isa_temperature(25000.0)) + 12.0, 1500.0),
    ]
    m, a, w, t, v = (np.array(c) for c in zip(*states, strict=True))
    x = model_inputs(m, a, w, t, v)
    ln_ff = sess.run(None, {inp.name: x})[0][:, 0]
    inside = in_envelope(env, m, a, w, t, v)
    print(
        f"\n  {'Mach':>5s} {'alt ft':>7s} {'mass kg':>8s} {'vs ft/min':>9s} "
        f"{'kg/h':>6s}  envelope"
    )
    for row, f, i in zip(states, np.exp(ln_ff), inside, strict=True):
        where = "inside" if i else "outside"
        print(
            f"  {row[0]:5.2f} {row[1]:7.0f} {row[2]:8.0f} {row[4]:9.0f} {f:6.0f}  "
            f"{where}"
        )
    print(f"\nISA pressure ratio at 37,000 ft: {math.exp(x[0, 1]):.4f}")


if __name__ == "__main__":
    main()
