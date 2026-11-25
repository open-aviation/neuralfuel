import json

import numpy as np
import torch

from neuralfuel.aircraft import aircraft
from neuralfuel.inference import export_onnx, fuel_flow, predict
from neuralfuel.model import FuelFlowNet, load, save, state_hash
from neuralfuel.prep import FEATURES
from neuralfuel.privacy import find_identifiers
from neuralfuel.train import TrainConfig, train
from neuralfuel.validate import validate


def small_model(blocks):
    torch.manual_seed(0)
    return FuelFlowNet.from_data(
        blocks[blocks.split == "train"], aircraft=aircraft("a320")
    )


def test_report_has_no_identifiers(syn_prepared, syn_samples):
    blocks, windows, _summary = syn_prepared
    rep = validate(small_model(blocks), blocks, windows, splits=["test", "val"])
    text = json.dumps(rep)  # valid JSON (no NaN)
    assert "NaN" not in text
    flights = set(syn_samples.flight_id.unique())
    assert find_identifiers(rep, extra_values=flights) == []
    assert not any(f in text for f in flights)
    for key in ("flight_id", "tail", "time", "lat", "lon", "registration"):
        assert f'"{key}"' not in text
    t0 = syn_samples.time.min().strftime("%Y-%m-%d")
    assert t0 not in text
    assert set(rep["accuracy"]) == {"all", "climb", "level", "descent", "cruise"}
    for k in (
        "trip_fuel",
        "steady_windows",
        "mach_sensitivity",
        "max_specific_range",
        "temperature_test",
        "constraints",
        "pair_test",
        "smoothness",
        "coverage",
    ):
        assert k in rep


def test_onnx_matches_pytorch(syn_prepared, tmp_path):
    blocks, _, _ = syn_prepared
    model = small_model(blocks)
    for casadi in (False, True):
        info = export_onnx(model, tmp_path / f"m{casadi}.onnx", casadi=casadi)
        assert info["max_abs_diff_ln_ff"] < 1e-5
        assert "Squeeze" not in info["ops"]
        assert ("Softplus" in info["ops"]) != casadi


def test_save_load_predict(syn_prepared, tmp_path):
    blocks, _, _ = syn_prepared
    model = small_model(blocks)
    save(model, tmp_path / "m.pt")
    m2 = load(tmp_path / "m.pt")
    assert state_hash(m2) == state_hash(model)
    x = blocks[FEATURES].iloc[:50]
    np.testing.assert_allclose(predict(m2, x), predict(model, x.to_numpy()), rtol=1e-6)
    b = blocks.iloc[:50]
    f = fuel_flow(m2, b.mach, b.alt_ft, b.mass_kg, b.sat_k, b.sin_gamma, b.acc_in)
    # block inputs use logs of block means; per-row recomputation agrees closely
    np.testing.assert_allclose(f, predict(m2, x), rtol=1e-3)


def test_continue_keeps_standardization(syn_prepared):
    blocks, windows, _ = syn_prepared
    init = small_model(blocks)
    init.x_mean += 0.5  # constants differ from the ones the data would give
    xm = init.x_mean.clone()
    cfg = TrainConfig(max_epochs=1, device="cpu", threads=2)
    cfg.loss.n_colloc = 256
    model, rec = train(blocks, windows, cfg, init=init, log=lambda s: None)
    assert torch.equal(model.x_mean, xm)
    assert rec["init"] == "continued from given weights"


def test_training_is_seeded(syn_prepared):
    blocks, windows, _ = syn_prepared
    cfg = TrainConfig(max_epochs=1, device="cpu", threads=2)
    cfg.loss.n_colloc = 256
    a = train(blocks, windows, cfg, log=lambda s: None)[0]
    b = train(blocks, windows, cfg, log=lambda s: None)[0]
    assert state_hash(a) == state_hash(b)
