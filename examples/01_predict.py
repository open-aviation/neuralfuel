# Run: uv run examples/01_predict.py
"""Fuel flow of the shared A320 model at single flight states.

Shows how to build the six model inputs from altitude, Mach, mass and temperature,
how climb and descent enter through sin_gamma and acc_in, and what the library
returns for a state inside and a state outside the valid envelope.

Run from the repository root:

    uv run examples/01_predict.py

Run time: a few seconds.
"""

import numpy as np

from neuralfuel import atmos
from neuralfuel.inference import fuel_flow, predict
from neuralfuel.model import load
from neuralfuel.support import in_envelope

MODEL = "models/a320/nf-a320-cfm56_5b.pt"
FPM = atmos.FT / 60.0  # m/s per ft/min


def inputs(mach, alt_ft, mass_kg, dT_isa_k, vs_fpm=0.0, dgs_dt=0.0, gs_over_tas=1.0):
    """One row of model inputs: Mach, ln delta, ln mass, sin_gamma, acc_in, ln theta.

    vs_fpm: geometric vertical speed (ft/min); dgs_dt: groundspeed rate (m/s^2);
    gs_over_tas: groundspeed / true airspeed (1 in still air).
    """
    sat_k = atmos.isa_temperature(alt_ft) + dT_isa_k  # static air temperature
    delta = atmos.isa_delta(alt_ft)  # ISA pressure ratio p / p0 at pressure altitude
    theta = sat_k / atmos.T0  # temperature ratio SAT / 288.15 K
    tas_ms = mach * atmos.sound_speed(sat_k)
    sin_gamma = vs_fpm * FPM / tas_ms
    acc_in = gs_over_tas * dgs_dt / atmos.G0
    return np.array(
        [mach, np.log(delta), np.log(mass_kg), sin_gamma, acc_in, np.log(theta)],
        dtype=float,
    )


def main():
    model = load(MODEL)
    print(f"model {MODEL}: size {model.size}, aircraft {model.aircraft.type}")

    # 1. steady cruise from flight conditions
    alt_ft, mach, mass_kg, dT = 37000.0, 0.78, 62000.0, 10.0
    sat_k = float(atmos.isa_temperature(alt_ft)) + dT
    ff = fuel_flow(model, mach=mach, alt_ft=alt_ft, mass_kg=mass_kg, sat_k=sat_k)
    print(
        f"\ncruise {alt_ft:.0f} ft, Mach {mach}, {mass_kg:.0f} kg, SAT {sat_k:.1f} K "
        f"(ISA {dT:+.0f} K): delta {float(atmos.isa_delta(alt_ft)):.4f}, "
        f"fuel flow {float(ff):.0f} kg/h"
    )

    # 2. the same state, climb and descent: the energy-rate inputs
    states = {
        "level": inputs(0.70, 25000.0, 65000.0, 12.0),
        "climb 1500 ft/min": inputs(0.70, 25000.0, 65000.0, 12.0, vs_fpm=1500.0),
        "climb, accelerating 0.3 m/s2": inputs(
            0.70, 25000.0, 65000.0, 12.0, vs_fpm=1500.0, dgs_dt=0.3
        ),
        "descent -2000 ft/min": inputs(0.70, 25000.0, 65000.0, 12.0, vs_fpm=-2000.0),
    }
    x = np.array(list(states.values()))
    ff, outside = predict(model, x, outside=True)
    print("\nMach 0.70, 25000 ft, 65000 kg, ISA +12 K:")
    print(f"  {'state':32s} {'sin_gamma':>9s} {'acc_in':>8s} {'kg/h':>7s}  envelope")
    for (name, row), f, o in zip(states.items(), ff, outside, strict=True):
        where = "outside" if o else "inside"
        print(f"  {name:32s} {row[3]:9.4f} {row[4]:8.4f} {f:7.0f}  {where}")

    # 3. the valid envelope: where the model has training data
    test = {
        "inside: cruise 35000 ft, Mach 0.76, 60 t": inputs(
            0.76, 35000.0, 60000.0, 10.0
        ),
        "outside: cruise 35000 ft, Mach 0.76, 75 t": inputs(
            0.76, 35000.0, 75000.0, 10.0
        ),
        "outside: cruise 37000 ft, ISA -10 K": inputs(0.78, 37000.0, 62000.0, -10.0),
    }
    x = np.array(list(test.values()))
    ff, outside = predict(model, x, outside=True)
    inside = in_envelope(model.envelope, x)
    print("\nvalid envelope (predict(..., outside=True) and support.in_envelope):")
    for name, f, o, i in zip(test, ff, outside, inside, strict=True):
        print(f"  {name:44s} {f:6.0f} kg/h  outside={bool(o)}  in_envelope={bool(i)}")
    print(
        "  outside the envelope the model still returns a number; the flag tells "
        "that no training data supports it"
    )


if __name__ == "__main__":
    main()
