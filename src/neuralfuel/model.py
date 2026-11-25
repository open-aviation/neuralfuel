"""The fuel-flow network.

Inputs (in this order): Mach, ln delta, ln mass (kg), sin_gamma, acc_in, ln theta,
with delta the ISA pressure ratio at pressure altitude and theta = SAT / 288.15 K.
Output: ln(total fuel flow in kg/h). The fixed standardization constants of inputs
and output are buffers of the model, so they travel with the weights.

Temperature rule (setting `temperature`):
  fixed  ln F = c ln theta + g(M, ln delta, ln m, sin_gamma, acc_in), c = theta_exp
         (default 0.5), not trained;
  learn  the same with c a trained parameter that starts at theta_exp;
  free   ln F = g(all six inputs), ln theta is a network input.
All settings take the same six inputs.

The model also holds the aircraft-type settings (`aircraft`, neuralfuel.aircraft:
type, engine family and the envelope values of the constraints), the valid envelope
(`envelope`, neuralfuel.support: where the model has training data) and the settings
of the physical constraints that the training used (`constraints`,
CONSTRAINT_SETTINGS of neuralfuel.losses.LossConfig: the envelope of the collocation
states and the limits of P1 to P6, and the P7 settings in 'drag_rise' when the
training used P7), so that the validation and the quality check test
the constraints that the model was trained with. `train` sets all three. A network
made in code (for example FuelFlowNet.from_data) has `envelope` and `constraints`
None, and `aircraft` None unless it is given; `save` and `load` keep these None
values. Without `constraints`, the validation and the quality check use the defaults
of LossConfig.

`load` reads only files of WEIGHTS_FORMAT. It refuses a file larger than
MAX_FILE_BYTES and an architecture outside MAX_LAYERS and MAX_HIDDEN before it builds
the network.
"""

import hashlib
import io
import os

import numpy as np
import torch
from torch import nn

from .aircraft import Aircraft, from_dict, size_of
from .prep import FEATURES

WEIGHTS_FORMAT = "neuralfuel-weights-1"
TEMPERATURE_RULES = ("fixed", "learn", "free")
MAX_FILE_BYTES = 64 * 2**20  # the standard size is about 0.1 MB
MAX_LAYERS, MAX_HIDDEN = 8, 1024
# a spread of an input below this share of its magnitude is a constant input
CONSTANT_RTOL = 1e-9
assert FEATURES[-1] == "ln_theta"


class FuelFlowNet(nn.Module):
    """MLP (by default 3 hidden layers of 64 softplus units) on standardized inputs.

    For 'fixed' and 'learn', x_mean and x_std hold the constants of the first five
    inputs and y_mean, y_std those of ln F - theta_exp ln theta.
    """

    def __init__(
        self,
        x_mean,
        x_std,
        y_mean,
        y_std,
        hidden=64,
        layers=3,
        temperature="fixed",
        theta_exp=0.5,
        aircraft: Aircraft | None = None,
    ):
        super().__init__()
        self.aircraft = aircraft
        self.envelope = None  # valid envelope (dict, neuralfuel.support) or None
        self.constraints = None  # constraint settings (dict) of the training or None
        if temperature not in TEMPERATURE_RULES:
            raise ValueError(
                f"temperature must be one of {TEMPERATURE_RULES}, got '{temperature}'"
            )
        self.features = list(FEATURES)
        self.hidden, self.layers = hidden, layers
        self.temperature = temperature
        n_in = len(self.features) - (temperature != "free")
        mods, width = [], n_in
        for _ in range(layers):
            mods += [nn.Linear(width, hidden), nn.Softplus()]
            width = hidden
        mods.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*mods)
        for name, v in [
            ("x_mean", x_mean),
            ("x_std", x_std),
            ("y_mean", y_mean),
            ("y_std", y_std),
        ]:
            self.register_buffer(name, torch.as_tensor(v, dtype=torch.float32))
        c = torch.tensor(float(theta_exp))
        if temperature == "learn":
            self.theta_exp = nn.Parameter(c)
        elif temperature == "fixed":
            self.register_buffer("theta_exp", c)

    @classmethod
    def from_data(
        cls,
        blocks,
        hidden=64,
        layers=3,
        temperature="fixed",
        theta_exp=0.5,
        aircraft=None,
    ):
        """New network with standardization constants from the training blocks."""
        x = blocks[FEATURES].to_numpy(dtype=float)
        y = blocks["ln_ff"].to_numpy(dtype=float)
        if temperature != "free":
            y = y - theta_exp * x[:, -1]
            x = x[:, :-1]
        mean, sd = x.mean(0), x.std(0)
        # a constant input (also one with a spread from rounding only) stays finite
        # after standardization
        sd[sd <= CONSTANT_RTOL * np.maximum(np.abs(mean), 1.0)] = 1.0
        return cls(
            mean,
            sd,
            y.mean(),
            y.std(),
            hidden,
            layers,
            temperature,
            theta_exp,
            aircraft,
        )

    @property
    def size(self):
        """'standard', 'xs' or 'custom' (see neuralfuel.aircraft.SIZES)."""
        return size_of(self.hidden, self.layers)

    def exponent(self):
        """The temperature exponent c (float), None for the 'free' setting."""
        return None if self.temperature == "free" else float(self.theta_exp.detach())

    def forward(self, x):
        """x: (..., 6) inputs -> ln fuel flow (kg/h), shape (...)."""
        if self.temperature == "free":
            z = self.net((x - self.x_mean) / self.x_std)
            return self.y_mean + self.y_std * z.squeeze(-1)
        z = self.net((x[..., :-1] - self.x_mean) / self.x_std)
        return self.theta_exp * x[..., -1] + (self.y_mean + self.y_std * z.squeeze(-1))

    def ln_ff_state(self, p):
        """ln fuel flow at flight states p = (M, ln delta, ln m, eps, ln theta).

        Still air: the energy rate eps is realized by the flight-path angle, so
        sin_gamma = eps and acc_in = 0.
        """
        m, lnd, lnm, eps, lnth = p.unbind(-1)
        return self.forward(
            torch.stack([m, lnd, lnm, eps, torch.zeros_like(eps), lnth], -1)
        )


def save(model: FuelFlowNet, path, meta=None):
    """Weights file: architecture, input names, temperature rule, aircraft settings
    and the state (with standardization)."""
    torch.save(
        {
            "format": WEIGHTS_FORMAT,
            "features": model.features,
            "hidden": model.hidden,
            "layers": model.layers,
            "temperature": model.temperature,
            "theta_exp": model.exponent(),
            "aircraft": model.aircraft.as_dict() if model.aircraft else None,
            "envelope": model.envelope,
            "constraints": model.constraints,
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "meta": meta or {},
        },
        path,
    )


def _read(path) -> dict:
    """The checkpoint dict of a weights file; ValueError for a file that is too large
    or not a weights file."""
    if os.path.getsize(path) > MAX_FILE_BYTES:
        raise ValueError(f"{path}: larger than {MAX_FILE_BYTES} bytes")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # pickle, zip and torch errors
        raise ValueError(f"{path}: not a {WEIGHTS_FORMAT} file ({e})") from e
    if not isinstance(ck, dict):
        raise ValueError(f"{path}: not a {WEIGHTS_FORMAT} file")  # noqa: TRY004
    return ck


def load(path, device="cpu") -> FuelFlowNet:
    ck = _read(path)
    fmt = ck.get("format")
    if fmt != WEIGHTS_FORMAT:
        raise ValueError(f"{path}: format '{fmt}' is not {WEIGHTS_FORMAT}")
    rule = ck.get("temperature")
    if list(ck.get("features") or []) != list(FEATURES):
        raise ValueError(f"{path}: inputs {ck.get('features')} differ from {FEATURES}")
    if rule not in TEMPERATURE_RULES:
        raise ValueError(f"{path}: unknown temperature rule '{rule}'")
    hidden, layers = ck.get("hidden"), ck.get("layers")
    if not (
        isinstance(hidden, int)
        and isinstance(layers, int)
        and 1 <= hidden <= MAX_HIDDEN
        and 1 <= layers <= MAX_LAYERS
    ):
        raise ValueError(
            f"{path}: architecture {layers} x {hidden} outside 1 to {MAX_LAYERS} "
            f"layers and 1 to {MAX_HIDDEN} units"
        )
    n = len(FEATURES) - (rule != "free")
    try:
        ac = from_dict(ck.get("aircraft"))
        model = FuelFlowNet(
            np.zeros(n), np.ones(n), 0.0, 1.0, hidden, layers, rule, 0.5, ac
        )
        model.load_state_dict(ck["state"])
    except (KeyError, TypeError, RuntimeError) as e:
        raise ValueError(f"{path}: invalid {WEIGHTS_FORMAT} file ({e})") from e
    model.envelope = ck.get("envelope")
    model.constraints = ck.get("constraints")
    return model.to(device).eval()


def read_meta(path) -> dict:
    """The 'meta' entry of a weights file (empty for a file without one)."""
    return _read(path).get("meta") or {}


def state_hash(model: FuelFlowNet) -> str:
    """SHA-256 of the parameters and buffers (independent of the file container)."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        buf = io.BytesIO()
        np.save(buf, v.detach().cpu().numpy())
        h.update(k.encode())
        h.update(buf.getvalue())
    return h.hexdigest()
