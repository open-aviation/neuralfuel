"""Aircraft-type settings, model sizes and model file names.

The envelope values that the physical constraints use are settings of the aircraft
type and are stored in the model file:

  mmo      maximum operating Mach (upper Mach of the collocation states, Mach of the
           range-boundary condition P6, upper end of the search for the Mach of
           maximum specific air range)
  vmo_kt   maximum operating CAS (upper CAS of the collocation states)
  mass_kg  mass range of the collocation states
  alt_ft   pressure-altitude range of the collocation states; the upper value is also
           the top of the P6 altitude band
  wing_area_m2
           reference wing area (lift coefficient of the valid envelope,
           neuralfuel.support); optional, without it the lift coefficient is not given

`aircraft` takes the default values of a type from the aerox library
(aerox.load_aircraft with the upper-case ICAO type code):

  mmo           aerox mmo
  vmo_kt        aerox vmo
  mass_kg       (aerox oew, aerox mtow)
  alt_ft        (ALT_LOW_FT, aerox ceiling)
  wing_area_m2  aerox wing.area

A value that aerox does not have, and every value of a type that aerox does not have,
is NaN; give it as a setting (for example `--set aircraft.mmo=0.86`). A setting
overrides the aerox value. A model file holds its own aircraft settings, and a loaded
model uses them, not aerox.

File names: nf-<type>[-<engine family>][-xs], all lower case, with .pt for the
weights and .onnx for the ONNX graph, for example nf-a320-cfm56_5b.pt and
nf-a320-cfm56_5b-xs.pt. A model without an engine family is the generic model of
the type, trained on all its engine variants.
"""

import math
import re
from dataclasses import asdict, dataclass

# lower bound of the altitude range of the collocation states: a modelling choice of
# neuralfuel, not a property of the aircraft. Below 10,000 ft, ATC restrictions (for
# example the speed limit) set the flown states, so the constraints are not applied
ALT_LOW_FT = 10000.0
NAN2 = (math.nan, math.nan)
# model sizes: hidden units per layer, hidden layers
SIZES = {"standard": (64, 3), "xs": (32, 3)}
NAME_PART = re.compile(r"^[a-z0-9_]+$")


@dataclass
class Aircraft:
    type: str
    engine: str = ""  # engine family; empty for the generic model of the type
    mmo: float = math.nan
    vmo_kt: float = math.nan
    mass_kg: tuple = NAN2
    alt_ft: tuple = NAN2
    wing_area_m2: float = math.nan

    def check(self):
        """Raise ValueError if a value is missing or inconsistent."""
        for part in (self.type, self.engine):
            if part and not NAME_PART.match(part):
                raise ValueError(
                    f"aircraft type and engine family must be lower case letters, "
                    f"digits or '_', got '{part}'"
                )
        if not self.type:
            raise ValueError("aircraft type is empty")
        vals = [self.mmo, self.vmo_kt, *self.mass_kg, *self.alt_ft]
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError(
                f"aircraft '{self.type}': envelope values missing (the type is not in "
                "aerox, or aerox has no value); set aircraft.mmo, aircraft.vmo_kt, "
                "aircraft.mass_kg and aircraft.alt_ft"
            )
        if not (self.mass_kg[0] < self.mass_kg[1] and self.alt_ft[0] < self.alt_ft[1]):
            raise ValueError(f"aircraft '{self.type}': ranges must be (low, high)")
        if not (math.isnan(self.wing_area_m2) or self.wing_area_m2 > 0):
            raise ValueError(f"aircraft '{self.type}': wing_area_m2 must be positive")
        return self

    def as_dict(self):
        d = asdict(self)
        d["mass_kg"], d["alt_ft"] = list(self.mass_kg), list(self.alt_ft)
        if math.isnan(self.wing_area_m2):
            d["wing_area_m2"] = None  # JSON has no NaN
        return d


def aircraft(type_: str, engine: str = "") -> Aircraft:
    """Settings of a type with the default values from aerox (see the module
    docstring); NaN for a value that aerox does not have."""
    from aerox import get_available_aircraft, load_aircraft

    t = type_.lower()
    code = t.upper()
    if code not in get_available_aircraft():
        return Aircraft(t, engine.lower())
    ac = load_aircraft(code)

    def num(v):
        return math.nan if v is None else float(v)

    wing = getattr(ac, "wing", None)
    return Aircraft(
        t,
        engine.lower(),
        mmo=num(ac.mmo),
        vmo_kt=num(ac.vmo),
        mass_kg=(num(ac.oew), num(ac.mtow)),
        alt_ft=(ALT_LOW_FT, num(ac.ceiling)),
        wing_area_m2=num(wing.area if wing is not None else None),
    )


def from_dict(d) -> Aircraft | None:
    """Settings from a stored dict (Aircraft.as_dict); None for None."""
    if d is None:
        return None
    d = dict(d)
    d["mass_kg"], d["alt_ft"] = tuple(d["mass_kg"]), tuple(d["alt_ft"])
    if d.get("wing_area_m2") is None:
        d["wing_area_m2"] = math.nan
    return Aircraft(**d)


def size_of(hidden: int, layers: int) -> str:
    """Name of the model size ('standard', 'xs') or 'custom'."""
    return next((k for k, v in SIZES.items() if v == (hidden, layers)), "custom")


def model_name(type_: str, engine: str = "", size: str = "standard") -> str:
    """File stem nf-<type>[-<engine family>][-xs]."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {tuple(SIZES)}, got '{size}'")
    parts = ["nf", type_.lower(), *([engine.lower()] if engine else [])]
    for p in parts[1:]:
        if not NAME_PART.match(p):
            raise ValueError(f"name part '{p}' must be lower case letters, digits, '_'")
    if size == "xs":
        parts.append("xs")
    return "-".join(parts)
