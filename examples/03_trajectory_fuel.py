# Run: uv run examples/03_trajectory_fuel.py
"""Trip fuel along a trajectory from surveillance-type states.

The script builds a simple climb, cruise and descent profile as an ADS-B or Mode S
receiver could record it: time, barometric altitude, groundspeed, track, vertical
rate and Mach, one state every 4 s. It is a made-up profile, not a recorded flight.
neuralfuel.inference.surveillance_inputs turns these states into the six model
inputs. It needs time, altitude, groundspeed, one airspeed (here Mach), the static
air temperature (here ISA +10 K, in practice from a weather model) and the mass. It
derives the vertical speed and the groundspeed rate itself from the altitude and
groundspeed history, so the recorded vertical rate and the track are not used.

Mass: the mass at the first state is an assumption (MASS0_KG). The mass along the
flight is the first mass minus the fuel burned so far. Because the fuel flow depends
on the mass, the script repeats the calculation until the mass changes by less than
1 kg.

Run from the repository root:

    uv run examples/03_trajectory_fuel.py

Run time: a few seconds.
"""

import numpy as np

from neuralfuel import atmos
from neuralfuel.inference import predict, surveillance_inputs
from neuralfuel.model import load

MODEL = "models/a320/nf-a320-cfm56_5b.pt"
MASS0_KG = 66000.0  # assumed mass at the first state (after take-off, clean)
DT_ISA = 10.0  # K, temperature of the whole flight
HEADWIND_KT = 30.0  # constant wind along the track
STEP_S = 4  # time between two surveillance states
CRUISE_FT, CRUISE_MACH, CRUISE_MIN = 37000.0, 0.78, 60.0


def profile():
    """1 Hz climb-cruise-descent profile, then every STEP_S-th state.

    Climb: 250 kt CAS below 10,000 ft, then 290 kt CAS up to Mach 0.76, then Mach
    0.76 to the cruise altitude. Cruise at CRUISE_MACH. Descent: 280 kt CAS (Mach
    0.76 above the crossover), 250 kt CAS below 10,000 ft. Speed changes at most
    0.5 kt/s. The profile starts and ends at 3,000 ft in clean configuration.
    """
    h, t, phase, t_cr = 3000.0, 0, "climb", None
    tas = None
    rows = []
    while True:
        sat = float(atmos.isa_temperature(h)) + DT_ISA
        a_kt = float(atmos.sound_speed(sat)) / atmos.KT
        if phase == "climb":
            cas = 250.0 if h < 10000 else 290.0
            m_t = min(float(atmos.mach_from_cas(cas, h)), 0.76)
            vs = 2500.0 if h < 10000 else max(1000.0, 3000.0 * (1 - h / 42000))
            if h + vs / 60 >= CRUISE_FT:
                phase, t_cr, h, vs = "cruise", t, CRUISE_FT, 0.0
        elif phase == "cruise":
            m_t, vs = CRUISE_MACH, 0.0
            if t - t_cr >= CRUISE_MIN * 60:
                phase = "descent"
        else:
            cas = 280.0 if h > 10000 else 250.0
            m_t = min(float(atmos.mach_from_cas(cas, h)), 0.76)
            vs = -2200.0 if h > 10000 else -1500.0
            if h + vs / 60 < 3000:
                break
        target = m_t * a_kt
        tas = target if tas is None else tas + np.clip(target - tas, -0.5, 0.5)
        rows.append((t, h, tas / a_kt, tas - HEADWIND_KT, 90.0, vs, phase))
        h += vs / 60
        t += 1
    rows = rows[::STEP_S]
    cols = ("time_s", "alt_ft", "mach", "gs_kt", "track_deg", "vrate_fpm", "phase")
    return {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}


def fuel_along(model, s, mass_kg):
    """Fuel flow (kg/h) and outside flags at the states s for the masses mass_kg."""
    sat = atmos.isa_temperature(s["alt_ft"]) + DT_ISA
    x = surveillance_inputs(
        s["time_s"], s["alt_ft"], s["gs_kt"], mass_kg, sat, mach=s["mach"]
    )
    return predict(model, x, outside=True)


def burned(time_s, ff_kgh):
    """Fuel burned (kg) from the first state to each state (trapezoidal rule)."""
    dt_h = np.diff(time_s) / 3600.0
    return np.r_[0.0, np.cumsum(0.5 * (ff_kgh[1:] + ff_kgh[:-1]) * dt_h)]


def main():
    model = load(MODEL)
    s = profile()
    n = len(s["time_s"])
    print(
        f"profile: {n} states every {STEP_S} s, {s['time_s'][-1] / 60:.1f} min, "
        f"cruise {CRUISE_FT:.0f} ft Mach {CRUISE_MACH}, ISA {DT_ISA:+.0f} K, "
        f"headwind {HEADWIND_KT:.0f} kt"
    )
    print(f"assumed mass at the first state: {MASS0_KG:.0f} kg")

    mass = np.full(n, MASS0_KG)
    ff0, _ = fuel_along(model, s, mass)  # constant mass, for comparison
    for it in range(1, 11):  # noqa: B007 (it is used after the loop)
        ff, outside = fuel_along(model, s, mass)
        new = MASS0_KG - burned(s["time_s"], ff)
        change = np.abs(new - mass).max()
        mass = new
        if change < 1.0:
            break
    fuel = burned(s["time_s"], ff)
    print(f"mass update: {it} iterations, last mass change {change:.2f} kg")

    print(
        f"\n  {'phase':8s} {'states':>6s} {'min':>6s} {'fuel kg':>8s} {'outside':>8s}"
    )
    for ph in ("climb", "cruise", "descent"):
        sel = np.flatnonzero(s["phase"] == ph)
        i, j = sel[0], min(sel[-1] + 1, n - 1)
        minutes = (s["time_s"][j] - s["time_s"][i]) / 60
        print(
            f"  {ph:8s} {len(sel):6d} {minutes:6.1f} {fuel[j] - fuel[i]:8.0f} "
            f"{100 * outside[sel].mean():7.1f}%"
        )
    print(
        f"  {'trip':8s} {n:6d} {s['time_s'][-1] / 60:6.1f} {fuel[-1]:8.0f} "
        f"{100 * outside.mean():7.1f}%"
    )
    print("  outside: share of states outside the valid envelope of the model")
    print(f"\nmass at the last state: {mass[-1]:.0f} kg")
    print(
        f"trip fuel with the mass held at {MASS0_KG:.0f} kg: "
        f"{burned(s['time_s'], ff0)[-1]:.0f} kg"
    )


if __name__ == "__main__":
    main()
