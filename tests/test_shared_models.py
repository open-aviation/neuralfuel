"""The shared models in models/<type code>/: each .pt loads, has the hash of its
record (.run.json for a model trained directly, .merge.json for a merged model), lies
in the folder of its aircraft type and agrees with its .onnx file."""

import copy
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from neuralfuel.aggregate import average, merge_record_path, save_merged
from neuralfuel.inference import export_onnx
from neuralfuel.model import load, state_hash
from neuralfuel.support import random_inputs

MODELS = sorted((Path(__file__).parent.parent / "models").glob("*/*.pt"))


def record(path):
    """The record of a shared model: <name>.run.json or <name>.merge.json."""
    for f in (path.with_suffix(".run.json"), merge_record_path(path)):
        if f.exists():
            return json.loads(f.read_text())
    raise AssertionError(f"{path.name}: no .run.json and no .merge.json")


def check_shared(path):
    model = load(path)
    assert path.parent.name == model.aircraft.type.lower()
    assert path.stem.startswith(f"nf-{path.parent.name}")
    assert record(path)["weights_sha256"] == state_hash(model)
    x = random_inputs(model.envelope, 500, seed=0).astype(np.float32)
    with torch.no_grad():
        y = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(path.with_suffix(".onnx")))
    (y_onnx,) = sess.run(None, {sess.get_inputs()[0].name: x})
    assert np.abs(y_onnx.reshape(-1) - y).max() < 1e-4  # ln fuel flow


def test_shared_models_exist():
    assert MODELS


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.stem)
def test_shared_model(path):
    check_shared(path)


def test_merged_release_layout(tmp_path):
    """A merged model is published with its .merge.json and without .run.json."""
    base = load(MODELS[0])
    kid = copy.deepcopy(base)
    with torch.no_grad():
        kid.net[-1].bias.add_(1e-4)
    merged, rec = average(base, [kid], [base.envelope])
    folder = tmp_path / base.aircraft.type
    folder.mkdir()
    path = folder / "nf-a320-merged.pt"
    save_merged(merged, rec, path)
    export_onnx(merged, path.with_suffix(".onnx"), n_check=10)
    check_shared(path)
    merge_record_path(path).unlink()
    with pytest.raises(AssertionError, match="no .run.json"):
        check_shared(path)
