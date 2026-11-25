"""Valid envelope: description of the training data, in_envelope, storage."""

import json

import numpy as np
import pytest
import torch

from neuralfuel import atmos
from neuralfuel.aircraft import aircraft, from_dict
from neuralfuel.inference import export_onnx, onnx_metadata, predict
from neuralfuel.model import FuelFlowNet, load, save
from neuralfuel.prep import FEATURES
from neuralfuel.support import (
    ENVELOPE_VARS,
    data_envelope,
    in_envelope,
    input_states,
    lift_coefficient,
    random_inputs,
    union_envelope,
)
from neuralfuel.train import TrainConfig, train


@pytest.fixture(scope="module")
def trained(syn_prepared):
    blocks, windows, _ = syn_prepared
    cfg = TrainConfig(max_steps=4, eval_every=2, device="cpu", threads=2)
    cfg.loss.n_colloc = 256
    return train(blocks, windows, cfg, log=lambda s: None)


def test_alt_from_delta():
    alt = np.array([0.0, 5000.0, 20000.0, 36089.0, 39000.0, 45000.0])
    np.testing.assert_allclose(
        atmos.alt_from_delta(atmos.isa_delta(alt)), alt, atol=1e-6
    )


def test_lift_coefficient():
    # 62 t at M 0.78, FL370 (delta 0.2138), 122.6 m2: about 0.5
    cl = lift_coefficient(62000.0, 0.78, atmos.isa_delta(37000.0), 122.6)
    assert cl == pytest.approx(
        2 * 62000 * 9.80665 / (1.4 * 101325 * 0.2138 * 0.78**2 * 122.6), rel=2e-3
    )


def test_data_envelope(syn_prepared):
    blocks, windows, _ = syn_prepared
    tr, wtr = blocks[blocks.split == "train"], windows[windows.split == "train"]
    env = data_envelope(tr, wtr, 122.6)
    lv = tr[tr.phase == "level"]
    assert env["phases"]["level"]["blocks"] == len(lv)
    assert env["phases"]["level"]["mach"] == pytest.approx(
        np.quantile(lv.mach, [0.01, 0.99]).tolist()
    )
    assert env["cruise"]["max_wd_t"] == pytest.approx(wtr.wd_t.max())
    cl = lift_coefficient(wtr.mass_kg, wtr.mach, wtr.delta, 122.6)
    assert env["cruise"]["max_cl"] == pytest.approx(cl.max())
    json.dumps(env, allow_nan=False)
    # the variables of the blocks come back from the model inputs
    s = input_states(tr[FEATURES])
    np.testing.assert_allclose(s["mach"], tr.mach)
    np.testing.assert_allclose(s["mass_kg"], tr.mass_kg, rtol=1e-12)
    np.testing.assert_allclose(s["alt_ft"], tr.alt_ft, atol=30.0)
    # most training blocks are inside, a state far outside is not
    inside = in_envelope(env, tr[FEATURES])
    assert inside.mean() > 0.8
    x = tr[FEATURES].to_numpy()[:5].copy()
    x[:, 0] = 0.95
    assert not in_envelope(env, x).any()
    x = random_inputs(env, 300, seed=1)
    assert x.shape == (300, 6) and in_envelope(env, x).all()
    # without the wing area there is no lift coefficient limit
    assert data_envelope(tr, wtr)["cruise"]["max_cl"] is None
    assert set(ENVELOPE_VARS) <= set(env["phases"]["all"])


def test_envelope_stored_everywhere(trained, syn_prepared, tmp_path):
    blocks, windows, _ = syn_prepared
    model, rec = trained
    assert model.envelope is not None and rec["envelope"] == model.envelope
    tr = blocks[blocks.split == "train"]
    assert model.envelope["phases"]["all"]["blocks"] == len(tr)
    save(model, tmp_path / "m.pt")
    m2 = load(tmp_path / "m.pt")
    assert m2.envelope == model.envelope
    export_onnx(m2, tmp_path / "m.onnx")
    meta = onnx_metadata(tmp_path / "m.onnx")
    assert meta["envelope"] == model.envelope
    assert meta["aircraft"]["wing_area_m2"] == model.aircraft.wing_area_m2 > 0
    x = blocks[FEATURES].iloc[:200]
    ff, out = predict(m2, x, outside=True)
    np.testing.assert_array_equal(ff, predict(m2, x))
    np.testing.assert_array_equal(out, ~in_envelope(model.envelope, x))
    from neuralfuel.validate import validate

    rep = validate(m2, blocks, windows)
    assert rep["envelope"]["source"] == "model file"
    assert rep["envelope"]["description"] == json.loads(json.dumps(model.envelope))


def test_model_without_envelope(syn_prepared, tmp_path):
    """A network made in code (not trained) is saved and loaded without envelope."""
    blocks, windows, _ = syn_prepared
    torch.manual_seed(0)
    ac = aircraft("a320")
    model = FuelFlowNet.from_data(blocks[blocks.split == "train"], aircraft=ac)
    save(model, tmp_path / "m.pt")
    m2 = load(tmp_path / "m.pt")
    assert m2.envelope is None and m2.constraints is None
    assert m2.aircraft == ac
    with pytest.raises(ValueError):
        predict(m2, blocks[FEATURES].iloc[:5], outside=True)
    from neuralfuel.validate import validate

    rep = validate(m2, blocks, windows)
    assert rep["envelope"]["source"] == "training split of the validated data"


def test_unknown_type_without_wing_area():
    d = {
        "type": "x1",
        "engine": "",
        "mmo": 0.7,
        "vmo_kt": 300.0,
        "mass_kg": [2e4, 4e4],
        "alt_ft": [5e3, 3e4],
        "wing_area_m2": None,
    }
    a = from_dict(d).check()
    assert np.isnan(a.wing_area_m2) and a.as_dict()["wing_area_m2"] is None
    assert np.isnan(from_dict(a.as_dict()).wing_area_m2)


def test_union_envelope_cruise_limits(syn_prepared):
    blocks, windows, _ = syn_prepared
    tr, wtr = blocks[blocks.split == "train"], windows[windows.split == "train"]
    a = data_envelope(tr[tr.cruise], wtr, 122.6)
    # level blocks above the cruise altitude without windows: no cruise limit
    b = data_envelope(tr[tr.phase == "level"], wtr.iloc[:0], 122.6)
    assert b["cruise"]["max_wd_t"] is None
    u = union_envelope(a, b)
    assert u["cruise"]["max_wd_t"] is None and u["cruise"]["max_cl"] is None
    for e in (a, b):
        assert in_envelope(u, random_inputs(e, 2000, seed=1)).all()
    # climb and descent blocks only: their missing limit says nothing about cruise
    c = data_envelope(tr[tr.phase != "level"], wtr.iloc[:0], 122.6)
    u = union_envelope(a, c)
    assert u["cruise"]["max_wd_t"] == a["cruise"]["max_wd_t"]
    assert u["cruise"]["max_cl"] == a["cruise"]["max_cl"]
    # no wing area: no lift-coefficient limit, and the wing areas must agree
    with pytest.raises(ValueError, match="wing area"):
        union_envelope(a, data_envelope(tr, wtr))
    with pytest.raises(ValueError, match="quantiles"):
        union_envelope(a, data_envelope(tr, wtr, 122.6, quantiles=(0.05, 0.95)))
    with pytest.raises(ValueError, match="phase rule"):
        union_envelope(a, data_envelope(tr, wtr, 122.6, level_fpm=200.0))


def test_constant_inputs(syn_prepared):
    """A constant input keeps the standardization finite and the envelope usable."""
    blocks, windows, _ = syn_prepared
    tr = blocks[blocks.split == "train"].copy()
    wt = windows[windows.split == "train"].copy()
    tr["mach"] = 0.78
    net = FuelFlowNet.from_data(tr)
    assert net.x_std[0].item() == 1.0
    for t in (tr, wt):
        t["mass_kg"] = 62000.0
        t["ln_m"] = np.log(62000.0)
    assert FuelFlowNet.from_data(tr).x_std[2].item() == 1.0
    env = data_envelope(tr, wt, 122.6)
    x = random_inputs(env, 50, seed=0)
    assert np.allclose(np.exp(x[:, 2]), 62000.0) and in_envelope(env, x).all()


def test_onnx_metadata_without_external_data(trained, tmp_path):
    import onnx

    model, _ = trained
    path = tmp_path / "m.onnx"
    export_onnx(model, path, n_check=10)
    m = onnx.load(str(path))
    onnx.save(
        m, str(path), save_as_external_data=True, size_threshold=0, location="w.bin"
    )
    (tmp_path / "w.bin").unlink()  # the tensor data is not read
    assert onnx_metadata(path)["envelope"] == model.envelope
