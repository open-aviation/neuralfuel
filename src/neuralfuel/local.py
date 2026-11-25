"""Private local correction: three offsets in ln fuel flow per aircraft.

The offsets correct the level of a model for one aircraft (or one fleet) of the user:
one constant per flight phase, climb, level and descent, added to ln fuel flow. The
phase of a state comes from its model inputs: the geometric vertical speed
sin_gamma x true airspeed, climb above +300 ft/min, descent below -300 ft/min, else
level (the rule of neuralfuel.support.phase_of). Each offset is the mean residual
ln F (measured) - ln F (model) of the training blocks of that aircraft and phase
(least squares); a phase without blocks gets 0.

The aircraft come from the optional `tail` column of the prepared blocks (neutral
labels such as A, B); without it all blocks are one aircraft, label 'all'.

The offsets are private. They are written to a separate JSON file and are never part
of a model file, a run record, a contribution or a validation report. The file holds
the weights hash of the model they were fitted against (model_weights_sha256);
`offsets_for` keeps it, and neuralfuel.inference.predict warns when the offsets are
applied to another model. The offsets
make a step in fuel flow at the phase borders, so the corrected model is for fuel
estimation, not for use inside a trajectory optimizer.
"""

import json
from pathlib import Path

import numpy as np

from .model import FuelFlowNet, state_hash
from .prep import FEATURES
from .privacy import dumps
from .support import input_states, phase_of

OFFSETS_FORMAT = "neuralfuel-offsets-1"
PHASES = ("climb", "level", "descent")
LEVEL_FPM = 300.0
ALL = "all"


def phases(x, level_fpm=LEVEL_FPM):
    """'climb', 'level' or 'descent' of model inputs x ((n, 6) array or DataFrame)."""
    return phase_of(input_states(x)["vs_fpm"], level_fpm)


def _ln_pred(model, x):
    from .inference import predict

    return np.log(predict(model, x))


def fit_offsets(model: FuelFlowNet, blocks, split="train", level_fpm=LEVEL_FPM):
    """Offsets per aircraft and phase from the blocks of `split` (all rows if the
    blocks have no 'split' column). Returns the dict of the offsets file."""
    b = blocks[blocks.split == split] if "split" in blocks else blocks
    if len(b) == 0:
        raise ValueError(f"no blocks in the '{split}' split")
    labels = b["tail"].astype(str).to_numpy() if "tail" in b else np.full(len(b), ALL)
    x = b[FEATURES].to_numpy(dtype=float)
    r = b["ln_ff"].to_numpy(dtype=float) - _ln_pred(model, x)
    ph = phases(x, level_fpm)
    out = {}
    for a in sorted(set(labels)):
        sel = labels == a
        e = {"offsets": {}, "blocks": {}}
        for p in PHASES:
            m = sel & (ph == p)
            e["offsets"][p] = float(r[m].mean()) if m.any() else 0.0
            e["blocks"][p] = int(m.sum())
        out[a] = e
    return {
        "format": OFFSETS_FORMAT,
        "private": "do not share; not part of any contribution",
        "units": "ln fuel flow, added to the model output",
        "level_fpm": level_fpm,
        "model_weights_sha256": state_hash(model),
        "aircraft": out,
    }


def write_offsets(offsets: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(offsets))
    return path


def read_offsets(path) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("format") != OFFSETS_FORMAT:
        raise ValueError(f"{path}: not a {OFFSETS_FORMAT} file")
    return d


def offsets_for(offsets: dict, label=None) -> dict:
    """The offsets {phase: value} of one aircraft of an offsets file (the only one if
    label is None and the file has one aircraft), with 'level_fpm' and the weights
    hash 'model_weights_sha256' of the model they were fitted against. The label is
    the 'tail' label of the prepared data; without a tail column it is 'all'."""
    ac = offsets["aircraft"]
    if label is None:
        if len(ac) != 1:
            raise ValueError(f"the offsets file has several aircraft: {sorted(ac)}")
        label = next(iter(ac))
    if label not in ac:
        raise ValueError(f"no offsets for aircraft '{label}'")
    return {
        **ac[label]["offsets"],
        "level_fpm": offsets.get("level_fpm", LEVEL_FPM),
        "model_weights_sha256": offsets.get("model_weights_sha256"),
    }


def offset_values(x, offsets: dict):
    """Offset (ln) per state of the inputs x for the offsets {phase: value} of one
    aircraft (offsets_for)."""
    ph = phases(x, offsets.get("level_fpm", LEVEL_FPM))
    out = np.zeros(len(ph))
    for p in PHASES:
        out[ph == p] = offsets.get(p, 0.0)
    return out
