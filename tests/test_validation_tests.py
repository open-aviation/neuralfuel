"""Validation tests: smoothness, input noise, roughness, surveillance inputs, pair tests
and the data-derived grids of the Mach and temperature tests."""

import copy
import json

import numpy as np
import pytest
import torch
from conftest import run_cli, write_prep
from torch.func import hessian as t_hessian

from neuralfuel.aircraft import Aircraft, aircraft
from neuralfuel.model import FuelFlowNet
from neuralfuel.privacy import find_identifiers
from neuralfuel.validate import (
    PAIR_SETS,
    ValidationConfig,
    cell_grid,
    hessian,
    mach_grid,
    mode_s_blocks,
    noise_sensitivity,
    roughness,
    state,
    steady_window_sd,
    validate,
    window_pairs,
)


def small_model(blocks):
    torch.manual_seed(0)
    return FuelFlowNet.from_data(
        blocks[blocks.split == "train"], aircraft=aircraft("a320")
    )


@pytest.fixture(scope="module")
def report(syn_prepared, syn_samples):
    blocks, windows, _ = syn_prepared
    return validate(small_model(blocks), blocks, windows, samples=syn_samples)


def test_new_sections(report, syn_samples):
    for k in ("roughness", "noise_sensitivity", "envelope", "surveillance"):
        assert k in report
    assert "NaN" not in json.dumps(report)
    flights = set(syn_samples.flight_id.unique())
    assert find_identifiers(report, extra_values=flights) == []
    sm = report["smoothness"]
    for ph in ("all", "climb", "level", "descent"):
        r = sm[ph]
        if r["pairs"]:
            assert r["ratio_pred_to_meas"] == pytest.approx(
                r["rms_diff_pred_kgh"] / r["rms_diff_meas_kgh"]
            )
    assert sm["steady_windows"]["windows"] > 0


def test_pair_tests_name_their_flights(report, syn_prepared):
    _, windows, _ = syn_prepared
    pt = report["pair_test"]
    assert set(pt) == set(PAIR_SETS)
    assert pt["test"]["flights_used"] == "test split"
    assert pt["test_and_val"]["splits"] == ["test", "val"]
    cfg = ValidationConfig()
    for key, (splits, _) in PAIR_SETS.items():
        w = windows[windows.split.isin(splits)].reset_index(drop=True)
        i, _ = window_pairs(w, cfg)
        assert pt[key]["pairs"] == len(i)
        assert pt[key]["flights"] == len(set(w.flight_id[i]))
    assert pt["test_and_val"]["pairs"] >= pt["test"]["pairs"]


def test_steady_window_sd_of_measurement_is_one(syn_prepared):
    blocks, windows, _ = syn_prepared
    te = blocks[blocks.split == "test"].reset_index(drop=True)
    wt = windows[windows.split == "test"].reset_index(drop=True)
    cfg = ValidationConfig()
    r = steady_window_sd(te.ff_kgh.to_numpy(), te, wt, cfg)
    assert r["windows"] > 0 and r["sd_ratio_median"] == pytest.approx(1.0)
    # a 300 s window holds at most 30 complete 10 s blocks
    r = steady_window_sd(
        te.ff_kgh.to_numpy(), te, wt, ValidationConfig(window_min_blocks=31)
    )
    assert r["windows"] == 0


def test_noise_sensitivity(syn_prepared):
    blocks, _, _ = syn_prepared
    m = copy.deepcopy(small_model(blocks)).double()
    te = blocks[blocks.split == "test"].reset_index(drop=True)
    zero = ValidationConfig(
        noise_mach=0, noise_alt_ft=0, noise_mass_rel=0, noise_energy=0, noise_sat_k=0
    )
    r0 = noise_sensitivity(m, te, zero)
    assert all(v == 0 for k, v in r0.items() if k.startswith("rms_rel"))
    r1 = noise_sensitivity(m, te, ValidationConfig())
    r2 = noise_sensitivity(m, te, ValidationConfig(noise_mach=0.004))
    assert r1["rms_rel_change_pct_mach"] > 0
    # small noise: the response is linear in the noise SD
    assert r2["rms_rel_change_pct_mach"] / r1["rms_rel_change_pct_mach"] == (
        pytest.approx(2.0, rel=0.02)
    )
    # the fixed temperature rule: ln F changes by 0.5 d ln theta
    sat = te.sat_k.to_numpy()
    z = np.random.default_rng(0).standard_normal((len(te), 6))[:, 5]
    ref = 100 * np.sqrt((np.expm1(0.5 * np.log1p(0.5 * z / sat)) ** 2).mean())
    assert r1["rms_rel_change_pct_sat_k"] == pytest.approx(ref, rel=1e-9)


def test_hessian_and_roughness(syn_prepared):
    blocks, _, _ = syn_prepared
    m = copy.deepcopy(small_model(blocks)).double()
    p = state([0.6, 0.78], [20000.0, 37000.0], [60000.0, 70000.0], 0.01, 230.0)
    _, _, h = hessian(m, p)
    for k in range(len(p)):
        ref = t_hessian(m.ln_ff_state)(p[k]).detach().numpy()
        np.testing.assert_allclose(h[k], ref, atol=1e-10)
    np.testing.assert_allclose(h, h.transpose(0, 2, 1), atol=1e-10)
    # a network with a zero output layer gives ln F = 0.5 ln theta + const: no curvature
    with torch.no_grad():
        m.net[-1].weight.zero_()
    _, _, h0 = hessian(m, p)
    r = roughness(h0, np.array([True, False]), np.ones(5))
    assert r["mean_hessian_frobenius_inside"] == 0
    assert r["mean_hessian_frobenius_outside"] == 0


def test_grids_follow_the_data_and_the_type(syn_prepared):
    _, windows, _ = syn_prepared
    cfg = ValidationConfig()
    a = aircraft("a320")
    wd, mach = cell_grid(windows, cfg, a)
    np.testing.assert_allclose(
        wd, np.round(np.quantile(windows.wd_t, cfg.wd_quantiles), 1)
    )
    assert mach.max() <= a.mmo + 1e-9 and np.allclose(np.diff(mach), cfg.mach_step)
    # a type with a three times larger W/delta and a lower MMO
    big = windows.copy()
    big["wd_t"] *= 3
    b = Aircraft("x1", mmo=0.7, vmo_kt=300.0, mass_kg=(1e5, 2e5), alt_ft=(1e4, 3e4))
    wd3, mach3 = cell_grid(big, cfg, b)
    np.testing.assert_allclose(wd3, 3 * wd, rtol=1e-3)
    assert mach3.max() <= 0.7 + 1e-9
    mg = mach_grid(cfg, b)
    assert mg[0] == pytest.approx(0.7 - cfg.sr_mach_span) and mg[-1] == 0.7


def test_report_grids(report, syn_prepared):
    _, windows, _ = syn_prepared
    cells = report["mach_sensitivity"]["cells"]
    wd = sorted({c["wd_t"] for c in cells})
    np.testing.assert_allclose(
        wd, np.round(np.quantile(windows.wd_t, ValidationConfig().wd_quantiles), 1)
    )
    for c in cells:
        assert c["supported"] == (c["n_windows"] >= 30)


def test_mode_s_blocks(syn_prepared, syn_samples):
    blocks, _, _ = syn_prepared
    te = blocks[blocks.split == "test"].reset_index(drop=True)
    # with a negligible resolution the rebuilt inputs equal the recorded ones
    fine = ValidationConfig(mode_s_mach=1e-12, mode_s_alt_ft=1e-9, mode_s_gs_kt=1e-9)
    x = mode_s_blocks(te, syn_samples, fine)
    ok = np.isfinite(x.to_numpy()).all(axis=1)
    assert ok.mean() > 0.99
    for c in x.columns:
        np.testing.assert_allclose(x[c][ok], te[c][ok], rtol=1e-6, atol=1e-8)
    xs = mode_s_blocks(te, syn_samples, ValidationConfig())
    d = np.abs(xs.mach - te.mach)[ok]
    assert 0 < d.max() <= 0.002 + 1e-9  # rounding to 0.004, block mean
    # samples of other flights than the blocks: an input error, not an IndexError
    other = syn_samples.assign(flight_id="x" + syn_samples.flight_id.astype(str))
    with pytest.raises(ValueError, match="no flight of the evaluated blocks"):
        mode_s_blocks(te, other, ValidationConfig())


def test_surveillance_section(report):
    sv = report["surveillance"]
    assert sv["blocks"] > 0 and "Mode S" in sv["variant"]
    assert set(sv["recorded_inputs"]) == {"all", "climb", "level", "descent", "cruise"}


def test_validate_cli_surveillance_flag(capsys, syn_prepared, syn_csv, tmp_path):
    from neuralfuel.model import save

    blocks, windows, _ = syn_prepared
    prep = write_prep(tmp_path / "prep", blocks, windows)
    save(small_model(blocks), tmp_path / "m.pt")
    args = ["validate", str(prep), "--model", str(tmp_path / "m.pt")]
    code, out = run_cli(capsys, *args, "--out", str(tmp_path / "a.json"))
    assert code == 0 and "surveillance" not in out
    code, out = run_cli(
        capsys,
        *args,
        "--out",
        str(tmp_path / "b.json"),
        "--surveillance",
        "--data",
        str(syn_csv),
        "--adapter",
        "synthetic",
    )
    assert code == 0 and out["surveillance"]["blocks"] > 0


def test_report_format_and_splits(report, syn_prepared):
    from neuralfuel.privacy import allowed
    from neuralfuel.validate import REPORT

    assert allowed(report, REPORT)[1] == []  # every key of the report is in REPORT
    blocks, windows, _ = syn_prepared
    with pytest.raises(ValueError, match="unknown split"):
        validate(small_model(blocks), blocks, windows, splits=["test", "local/x"])


def test_validation_takes_the_preparation_settings(syn_samples):
    """20 s blocks: the smoothness pairs and the surveillance test use the block
    length of the preparation."""
    from neuralfuel.prep import PrepConfig, prepare

    pc = PrepConfig(block_s=20)
    blocks, windows, _ = prepare(syn_samples, pc)
    cfg = ValidationConfig(grid_n=(4, 4, 3, 3, 2))
    rep = validate(
        small_model(blocks),
        blocks,
        windows,
        cfg=cfg,
        samples=syn_samples,
        prep_config=pc,
    )
    assert rep["config"]["block_s"] == 20 and rep["prep"]["block_s"] == 20
    assert rep["smoothness"]["all"]["pairs"] > 0
    assert rep["surveillance"]["blocks"] > 0.9 * rep["accuracy"]["all"]["n"]
