"""Adapter for the raw format of the synthetic example dataset (neuralfuel.synthetic).

Raw units: pressure altitude m, groundspeed m/s, mass t, engine fuel flow kg/s,
static air temperature degC, flap angle deg, time as ISO 8601 text. The imports are
absolute, so a copy of this file in local/ works unchanged.

MEASUREMENTS lists the raw measurement columns: `profile --raw` gives value ranges
only for these columns (and for the numeric columns of the schema).
"""

import numpy as np
import pandas as pd

from neuralfuel.atmos import FT, KT

MEASUREMENTS = (
    "palt_m",
    "mach",
    "gs_ms",
    "gw_t",
    "ff_eng1_kgs",
    "ff_eng2_kgs",
    "oat_c",
    "flap_deg",
    "bank_deg",
)


def load(path) -> pd.DataFrame:
    from neuralfuel.adapters import read_table

    r = read_table(path)
    ff1 = r.ff_eng1_kgs.to_numpy(dtype=float) * 3600.0
    ff2 = r.ff_eng2_kgs.to_numpy(dtype=float) * 3600.0
    return pd.DataFrame(
        {
            "flight_id": r.leg.astype(str),
            "time": pd.to_datetime(r.utc, utc=True),
            "alt_ft": r.palt_m.to_numpy(dtype=float) / FT,
            "mach": r.mach.to_numpy(dtype=float),
            "gs_kt": r.gs_ms.to_numpy(dtype=float) / KT,
            "mass_kg": r.gw_t.to_numpy(dtype=float) * 1000.0,
            "ff_kgh": ff1 + ff2,
            "ff1_kgh": ff1,
            "ff2_kgh": ff2,
            "sat_k": r.oat_c.to_numpy(dtype=float) + 273.15,
            "clean": np.abs(r.flap_deg.to_numpy(dtype=float)) < 0.5,
            "roll_deg": r.bank_deg.to_numpy(dtype=float),
        }
    )
