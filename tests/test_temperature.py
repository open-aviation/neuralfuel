"""Temperature rule settings: save and load, continuation, ONNX, inputs, report."""

import copy

import numpy as np
import pytest
import torch

from neuralfuel import atmos
from neuralfuel.aircraft import aircraft
from neuralfuel.inference import export_onnx, fuel_flow, predict, surveillance_inputs
from neuralfuel.model import (
    TEMPERATURE_RULES,
    WEIGHTS_FORMAT,
    FuelFlowNet,
    load,
    save,
    state_hash,
)
from neuralfuel.prep import FEATURES, derive
from neuralfuel.train import TrainConfig, train
from neuralfuel.validate import ValidationConfig, temperature_test, validate


def quiet(_):
    pass


def small_cfg(rule, epochs=2):
    cfg = TrainConfig(max_epochs=epochs, device="cpu", threads=2, temperature=rule)
    cfg.loss.n_colloc = 256
    return cfg


@pytest.fixture(scope="module")
def trained(syn_prepared):
    blocks, windows, _ = syn_prepared
    return {
        r: train(blocks, windows, small_cfg(r), log=quiet)[0] for r in TEMPERATURE_RULES
    }


def test_default_is_fixed_half():
    assert TrainConfig().temperature == "fixed" and TrainConfig().theta_exp == 0.5


@pytest.mark.parametrize("rule", TEMPERATURE_RULES)
def test_save_load(trained, syn_prepared, tmp_path, rule):
    blocks, _, _ = syn_prepared
    model = trained[rule]
    save(model, tmp_path / "m.pt")
    m2 = load(tmp_path / "m.pt")
    assert m2.temperature == rule and m2.exponent() == model.exponent()
    assert state_hash(m2) == state_hash(model)
    x = blocks[FEATURES].iloc[:200]
    np.testing.assert_array_equal(predict(m2, x), predict(model, x))


@pytest.mark.parametrize("rule", TEMPERATURE_RULES)
def test_continue_keeps_rule(trained, syn_prepared, rule):
    blocks, windows, _ = syn_prepared
    init = trained[rule]
    c0 = init.exponent()
    cfg = small_cfg("free" if rule != "free" else "fixed", epochs=1)
    model, rec = train(blocks, windows, cfg, init=init, log=quiet)
    assert (
        model.temperature == rule == rec["temperature"] == rec["config"]["temperature"]
    )
    if rule == "fixed":
        assert model.exponent() == c0 == 0.5
    if rule == "learn":
        assert model.exponent() != c0  # the exponent is trained


@pytest.mark.parametrize("rule", TEMPERATURE_RULES)
def test_onnx_matches_pytorch(trained, tmp_path, rule):
    for casadi in (False, True):
        info = export_onnx(trained[rule], tmp_path / f"m{casadi}.onnx", casadi=casadi)
        assert info["max_rel_diff_ff"] < 1e-5
        assert info["temperature"] == rule
        assert "Squeeze" not in info["ops"]
        assert not {"Slice", "Gather"} & set(info["ops"])
        assert ("Softplus" in info["ops"]) != casadi


@pytest.mark.parametrize("rule", TEMPERATURE_RULES)
def test_surveillance_inputs(trained, syn_samples, rule):
    one = syn_samples[syn_samples.flight_id == syn_samples.flight_id.iloc[0]]
    s, _, _ = derive(one)
    x = surveillance_inputs(
        s.time_s, s.alt_ft, s.gs_kt, s.mass_kg, s.sat_k, tas_kt=s.tas_kt
    )
    ok = np.isfinite(x.to_numpy()).all(axis=1)
    f = predict(trained[rule], x[ok])
    assert ok.sum() > 100 and np.isfinite(f).all() and (f > 0).all()


def test_fixed_rule_scales_with_sqrt_theta(trained):
    model = trained["fixed"]
    f = fuel_flow(model, 0.78, 35000.0, 62000.0, [210.0, 240.0])
    assert f[1] / f[0] == pytest.approx((240.0 / 210.0) ** 0.5, rel=1e-5)


@pytest.mark.parametrize(
    "fmt",
    ["neuralfuel-weights-2", "neuralfuel-weights-4", "neuralfuel-weights-0", None],
)
def test_load_refuses_other_format(syn_prepared, tmp_path, fmt):
    """Only files of WEIGHTS_FORMAT load; the same content with another format
    string is refused."""
    blocks, _, _ = syn_prepared
    torch.manual_seed(0)
    model = FuelFlowNet.from_data(blocks[blocks.split == "train"], temperature="free")
    save(model, tmp_path / "m.pt")
    ck = torch.load(tmp_path / "m.pt", weights_only=True)
    assert ck["format"] == WEIGHTS_FORMAT
    m2 = load(tmp_path / "m.pt")
    assert m2.temperature == "free" and m2.exponent() is None
    x = blocks[FEATURES].iloc[:100]
    np.testing.assert_array_equal(predict(m2, x), predict(model, x))
    ck["format"] = fmt
    torch.save(ck, tmp_path / "other.pt")
    with pytest.raises(ValueError, match=f"is not {WEIGHTS_FORMAT}"):
        load(tmp_path / "other.pt")


def test_report_temperature_test(trained, syn_prepared):
    blocks, windows, _ = syn_prepared
    rep = validate(trained["fixed"], blocks, windows)
    tt = rep["temperature_test"]
    assert [r["dT_isa_k"] for r in tt["rows"]] == [-20, -10, 0, 10, 20]
    assert tt["shift_mach_max_sr"] == pytest.approx(
        tt["rows"][-1]["mach_max_sr"] - tt["rows"][0]["mach_max_sr"]
    )
    # the test point is the median altitude and mass of the steady windows
    assert tt["alt_ft"] == pytest.approx(windows.alt_ft.median())
    assert tt["mass_kg"] == pytest.approx(windows.mass_kg.median())
    t_isa = float(atmos.isa_temperature(tt["alt_ft"]))
    assert tt["rows"][0]["sat_k"] == pytest.approx(t_isa - 20)
    # with a fixed exponent the Mach of maximum specific air range cannot move
    assert tt["shift_mach_max_sr"] == 0
    assert rep["model"]["temperature"] == "fixed"
    assert rep["model"]["theta_exp"] == 0.5
    # the free network has no such constraint; the test runs for it as well
    tf = temperature_test(
        copy.deepcopy(trained["free"]).double(),
        ValidationConfig(),
        aircraft("a320"),
        35000.0,
        62000.0,
    )
    assert len(tf["rows"]) == 5
