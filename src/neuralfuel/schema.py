"""Canonical sample table: one row per recorded sample (1 Hz), one flight per flight_id.

An adapter converts a raw data file into this table. All later steps use only
these columns and units.

| column     | unit / type     | required            | meaning                         |
|------------|-----------------|---------------------|---------------------------------|
| flight_id  | string          | yes                 | identifier of the flight (stays |
|            |                 |                     | local)                          |
| time       | datetime64, UTC | yes                 | sample time                     |
| alt_ft     | ft              | yes                 | pressure altitude (standard     |
|            |                 |                     | setting)                        |
| cas_kt     | kt              | one of cas/tas/mach | calibrated airspeed             |
| tas_kt     | kt              | one of cas/tas/mach | true airspeed                   |
| mach       | -               | one of cas/tas/mach | Mach number                     |
| gs_kt      | kt              | yes                 | groundspeed                     |
| mass_kg    | kg              | yes                 | gross mass                      |
| ff_kgh     | kg/h            | yes                 | total fuel flow of all engines  |
| tat_k      | K               | one of tat/sat      | total air temperature           |
| sat_k      | K               | one of tat/sat      | static air temperature          |
| clean      | bool            | yes                 | True when flaps and slats are   |
|            |                 |                     | retracted                       |
| speedbrake | bool            | no                  | True when the speed brake is    |
|            |                 |                     | extended in flight              |
| gear_down  | bool            | no                  | True when the landing gear is   |
|            |                 |                     | not up and locked (extended or  |
|            |                 |                     | in transit)                     |
| anti_ice   | bool            | no                  | True when engine anti-ice is on |
| tail       | string          | no                  | neutral aircraft label, for     |
|            |                 |                     | example A or B (stays local)    |
| ff1_kgh    | kg/h            | no                  | fuel flow of engine 1           |
|            |                 |                     | (imbalance filter)              |
| ff2_kgh    | kg/h            | no                  | fuel flow of engine 2           |
|            |                 |                     | (imbalance filter)              |
| roll_deg   | deg             | no                  | roll angle (steady-window       |
|            |                 |                     | filter)                         |

If more than one airspeed is given, the pipeline uses CAS first, then Mach, then TAS.
If both temperatures are given, it uses TAT.

The model covers flight in clean configuration with the speed brake and the landing
gear retracted. `prep` removes samples with `clean` False, and samples with
`speedbrake` or `gear_down` True when these columns exist. It carries `anti_ice` and
`tail` through to the blocks; samples with anti-ice on are removed only if the
setting `exclude_anti_ice` of the preparation is set. A flag column must not have
missing values.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Column:
    name: str
    unit: str
    kind: str  # "str", "time", "float", "bool"
    required: bool
    lo: float | None = None  # hard physical bounds of valid values
    hi: float | None = None
    hint: str = ""  # likely cause when values are outside the bounds


COLUMNS = [
    Column("flight_id", "-", "str", True),
    Column("time", "UTC", "time", True),
    Column("alt_ft", "ft", "float", True, -2000, 51000, "altitude in metres or FL?"),
    Column("cas_kt", "kt", "float", False, 0, 450, "CAS in km/h or m/s?"),
    Column("tas_kt", "kt", "float", False, 0, 650, "TAS in km/h or m/s?"),
    Column("mach", "-", "float", False, 0, 1.0, "Mach scaled (for example x1000)?"),
    Column("gs_kt", "kt", "float", True, 0, 750, "groundspeed in km/h?"),
    Column("mass_kg", "kg", "float", True, 5000, 600000, "mass in t, lb or 10 kg?"),
    Column("ff_kgh", "kg/h", "float", True, 0, 30000, "fuel flow in kg/s or lb/h?"),
    Column("tat_k", "K", "float", False, 180, 340, "temperature in degC or raw?"),
    Column("sat_k", "K", "float", False, 180, 330, "temperature in degC or raw?"),
    Column("clean", "-", "bool", True),
    Column("speedbrake", "-", "bool", False),
    Column("gear_down", "-", "bool", False),
    Column("anti_ice", "-", "bool", False),
    Column("tail", "-", "str", False),
    Column("ff1_kgh", "kg/h", "float", False, 0, 15000, "fuel flow in kg/s?"),
    Column("ff2_kgh", "kg/h", "float", False, 0, 15000, "fuel flow in kg/s?"),
    Column("roll_deg", "deg", "float", False, -90, 90, "roll in rad?"),
]
BY_NAME = {c.name: c for c in COLUMNS}
ONE_OF = [("cas_kt", "tas_kt", "mach"), ("tat_k", "sat_k")]
# columns that identify a flight, an aircraft, a time or a position; never reported
IDENTIFYING = ("flight_id", "tail", "time")


@dataclass
class Problem:
    column: str
    level: str  # "error" or "warning"
    message: str

    def as_dict(self):
        return {"column": self.column, "level": self.level, "message": self.message}


class SchemaError(ValueError):
    """Raised when the sample table cannot be used. Holds the list of problems."""

    def __init__(self, problems):
        self.problems = problems
        super().__init__(
            "; ".join(
                f"{p.column}: {p.message}" for p in problems if p.level == "error"
            )
        )


def validate(df: pd.DataFrame, max_bad_share: float = 0.001) -> list[Problem]:
    """Check columns, types and hard unit bounds. Returns a list of problems.

    A column with more than `max_bad_share` of its non-missing values outside the hard
    bounds is an error, because this nearly always means a wrong unit or scale factor.
    """
    out = []
    for c in COLUMNS:
        if c.required and c.name not in df:
            out.append(Problem(c.name, "error", "required column is missing"))
    for group in ONE_OF:
        if not any(g in df for g in group):
            out.append(
                Problem(
                    "/".join(group), "error", "at least one of these columns is needed"
                )
            )
    for c in COLUMNS:
        if c.name not in df:
            continue
        s = df[c.name]
        if c.kind == "time":
            if not pd.api.types.is_datetime64_any_dtype(s):
                out.append(
                    Problem(c.name, "error", f"must be datetime64, is {s.dtype}")
                )
            continue
        if c.kind == "bool":
            if s.isna().any():
                out.append(Problem(c.name, "error", "has missing values"))
            elif not pd.api.types.is_bool_dtype(s):
                out.append(Problem(c.name, "error", f"must be bool, is {s.dtype}"))
            continue
        if c.kind == "str":
            if s.isna().any():
                out.append(Problem(c.name, "error", "has missing values"))
            continue
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            out.append(Problem(c.name, "error", f"must be numeric, is {s.dtype}"))
            continue
        v = s.to_numpy(dtype=float)
        fin = np.isfinite(v)
        miss = 1 - fin.mean() if len(v) else 0.0
        if miss > 0.05:
            out.append(
                Problem(c.name, "warning", f"{100 * miss:.1f} % of values are missing")
            )
        if c.lo is not None and fin.any():
            bad = ((v[fin] < c.lo) | (v[fin] > c.hi)).mean()
            if bad > max_bad_share:
                out.append(
                    Problem(
                        c.name,
                        "error",
                        f"{100 * bad:.2f} % of values outside [{c.lo:g}, {c.hi:g}] "
                        f"{c.unit}; {c.hint}",
                    )
                )
    return out


def require_valid(df: pd.DataFrame) -> None:
    """Raise SchemaError if validate() finds an error."""
    problems = validate(df)
    if any(p.level == "error" for p in problems):
        raise SchemaError(problems)
