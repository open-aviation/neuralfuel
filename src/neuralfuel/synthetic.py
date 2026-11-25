"""Synthetic example dataset. The data is ARTIFICIAL; it comes from simple physics.

Each flight climbs, cruises with two Mach changes and descends at 1 Hz. Fuel flow is
thrust-specific fuel consumption times thrust, with drag from a parabolic polar plus a
compressibility term, flight-path and acceleration terms, an idle floor and
measurement noise. The numbers are only plausible for a medium jet; they are not
data of any real aircraft.

The raw file uses a deliberately different format from the schema (see
neuralfuel.adapters.synthetic): CSV with columns
  leg          flight label
  utc          ISO 8601 time string
  palt_m       pressure altitude, m
  mach         Mach number
  gs_ms        groundspeed, m/s
  gw_t         gross mass, t
  ff_eng1_kgs  fuel flow engine 1, kg/s
  ff_eng2_kgs  fuel flow engine 2, kg/s
  oat_c        static air temperature, degC
  flap_deg     flap angle, deg (0 = clean)
  bank_deg     roll angle, deg

Usage: uv run python -m neuralfuel.synthetic OUT.csv [--flights 30] [--seed 0]
"""

import argparse
import math

import numpy as np
import pandas as pd

from .atmos import (
    FT,
    G0,
    GAMMA,
    KT,
    R_GAS,
    T0,
    isa_delta,
    isa_temperature,
    mach_from_cas,
)

S_REF = 122.6  # m^2
CD0, K_IND = 0.024, 0.0375
TSFC_C = 0.045  # kg/(N h) at M = 0, theta = 1
P0 = 101325.0


def _drag_coeff(cl, mach):
    m_crit = 0.87 - 0.1 * cl - 0.108  # Korn-type drag divergence minus Lock offset
    wave = 20.0 * max(mach - m_crit, 0.0) ** 4
    return CD0 + K_IND * cl * cl + wave


def _fuel_flow(mass, tas_ms, alt_ft, sat, sin_g, acc):
    """Fuel flow kg/h of the synthetic aircraft."""
    delta = float(isa_delta(alt_ft))
    rho = delta * P0 / (R_GAS * sat)
    q = 0.5 * rho * tas_ms**2
    mach = tas_ms / math.sqrt(GAMMA * R_GAS * sat)
    cl = mass * G0 / (q * S_REF)
    thrust = q * S_REF * _drag_coeff(cl, mach) + mass * G0 * sin_g + mass * acc
    tsfc = TSFC_C * (1 + 0.7 * mach) * math.sqrt(sat / T0)
    idle = 250.0 + 450.0 * delta
    return max(tsfc * thrust, idle)


def simulate_flight(rng, start):
    m = rng.uniform(56000, 72000)
    dT = rng.uniform(-8, 15)
    wind = rng.uniform(-50, 50)
    h_cr = float(np.clip(39000 - round((m - 56000) / 16000 * 3) * 2000, 33000, 39000))
    m_cr = rng.uniform(0.74, 0.79)
    t_cr = rng.uniform(2400, 4200)
    steps = rng.choice([-0.02, 0.02], 2)
    noise, imb = 0.0, rng.normal(0, 0.004)
    h, t, phase, t0 = 1500.0, 0, "climb", None
    sat = float(isa_temperature(h)) + dT
    v = float(mach_from_cas(180.0, h)) * math.sqrt(GAMMA * R_GAS * sat) / KT
    rows = []
    while True:
        sat = float(isa_temperature(h)) + dT
        a = math.sqrt(GAMMA * R_GAS * sat)
        if phase == "climb":
            cas = 250.0 if h < 10000 else 290.0
            m_t = min(float(mach_from_cas(cas, h)), 0.76)
            vs = max(800.0, 3000.0 * (1 - h / 42000))
            if h + vs / 60 >= h_cr:
                vs, h, phase, t0 = 0.0, h_cr, "cruise", t
        elif phase == "cruise":
            tc = t - t0
            m_t = m_cr + steps[0] * (tc > t_cr / 3) + steps[1] * (tc > 2 * t_cr / 3)
            vs = 0.0
            if tc > t_cr:
                phase = "descent"
        else:
            cas = 280.0 if h > 10000 else 250.0
            m_t = min(float(mach_from_cas(cas, h)), 0.76)
            vs = -2200.0 if h > 10000 else -1500.0
            if h < 2000:
                break
        v_t = m_t * a / KT
        rate = 0.3 if phase == "cruise" else 0.6
        dv = min(max(v_t - v, -rate), rate)
        v_ms = v * KT
        sin_g = (vs / 60 * FT) / v_ms
        ff = _fuel_flow(m, v_ms, h, sat, sin_g, dv * KT)
        noise = 0.95 * noise + rng.normal(0, 0.003)
        ff_meas = ff * (1 + noise + rng.normal(0, 0.004))
        wind += rng.normal(0, 0.05)
        flap = 5.0 if h < 4000 else 0.0
        rows.append(
            (
                t,
                h * FT,
                v_ms / a,
                (v + wind) * KT,
                m / 1000,
                ff_meas,
                imb,
                sat - 273.15,
                flap,
            )
        )
        v += dv
        h += vs / 60
        m -= ff_meas / 3600
        t += 1
    r = np.array(rows)
    n = len(r)
    ff = r[:, 5]
    return pd.DataFrame(
        {
            "utc": (start + pd.to_timedelta(r[:, 0], unit="s")).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "palt_m": np.round(r[:, 1], 1),
            "mach": np.round(r[:, 2], 4),
            "gs_ms": np.round(r[:, 3], 2),
            "gw_t": np.round(r[:, 4], 3),
            "ff_eng1_kgs": np.round(ff * (0.5 + r[:, 6]) / 3600, 6),
            "ff_eng2_kgs": np.round(ff * (0.5 - r[:, 6]) / 3600, 6),
            "oat_c": np.round(r[:, 7] * 4) / 4,
            "flap_deg": r[:, 8],
            "bank_deg": np.round(rng.normal(0, 1.0, n), 2),
        }
    )


def generate(n_flights=30, seed=0) -> pd.DataFrame:
    """Raw synthetic dataset (see module docstring)."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2030-01-01T06:00:00Z")
    out = []
    for i in range(n_flights):
        start = base + pd.Timedelta(hours=8 * i + float(rng.uniform(0, 2)))
        f = simulate_flight(rng, start)
        f.insert(0, "leg", f"SYN{i:04d}")
        out.append(f)
    return pd.concat(out, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Write the synthetic example dataset (CSV)."
    )
    ap.add_argument("out")
    ap.add_argument("--flights", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    df = generate(a.flights, a.seed)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows, {a.flights} synthetic flights")


if __name__ == "__main__":
    main()
