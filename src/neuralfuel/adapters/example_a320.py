"""Adapter for the raw format of the A320 QAR dataset (1 Hz parquet).

The imports are absolute, so a copy of this file in local/ works unchanged.

Raw columns and scale factors:
  timestamp               datetime (UTC)
  altitude                pressure altitude, ft
  CAS                     kt
  groundspeed             kt
  gross_weight            kg
  fuel_flow, ff1, ff2     kg/h (total, engine 1, engine 2)
  total_air_temperature   raw = 4 x degC; the raw floor -127 marks an invalid sample
  flap_raw                flap/slat position code; the clean configuration is
                          |raw| <= 1.5
  roll                    deg
"""

import numpy as np
import pandas as pd

RAW = [
    "flight_id",
    "timestamp",
    "altitude",
    "CAS",
    "groundspeed",
    "gross_weight",
    "fuel_flow",
    "ff1",
    "ff2",
    "total_air_temperature",
    "flap_raw",
    "roll",
]
# raw measurement columns: `profile --raw --adapter example_a320` gives their ranges
MEASUREMENTS = RAW[2:]
TAT_SCALE = 4.0  # raw counts per degC
TAT_FLOOR = -127.0  # raw sensor floor, invalid
CLEAN_TOL = 1.5  # |flap_raw| at or below this value is the clean configuration


def load(path) -> pd.DataFrame:
    from neuralfuel.adapters import read_table

    r = read_table(path, columns=RAW)
    tat_raw = r.total_air_temperature.to_numpy(dtype=float)
    tat_k = np.where(tat_raw > TAT_FLOOR, tat_raw / TAT_SCALE + 273.15, np.nan)
    time = pd.to_datetime(r.timestamp)
    if time.dt.tz is None:
        time = time.dt.tz_localize("UTC")
    return pd.DataFrame(
        {
            "flight_id": r.flight_id.astype(str),
            "time": time,
            "alt_ft": r.altitude.to_numpy(dtype=float),
            "cas_kt": r.CAS.to_numpy(dtype=float),
            "gs_kt": r.groundspeed.to_numpy(dtype=float),
            "mass_kg": r.gross_weight.to_numpy(dtype=float),
            "ff_kgh": r.fuel_flow.to_numpy(dtype=float),
            "ff1_kgh": r.ff1.to_numpy(dtype=float),
            "ff2_kgh": r.ff2.to_numpy(dtype=float),
            "tat_k": tat_k,
            "clean": np.abs(r.flap_raw.to_numpy(dtype=float)) <= CLEAN_TOL,
            "roll_deg": r.roll.to_numpy(dtype=float),
        }
    )
