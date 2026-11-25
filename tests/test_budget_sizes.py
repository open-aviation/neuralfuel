"""Training budget in update steps, model sizes, aircraft settings and file names."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import run_cli, write_prep

from neuralfuel.aggregate import differences, run_record_problems
from neuralfuel.aircraft import (
    ALT_LOW_FT,
    SIZES,
    Aircraft,
    aircraft,
    model_name,
    size_of,
)
from neuralfuel.inference import predict
from neuralfuel.losses import LossConfig, envelope
from neuralfuel.model import FuelFlowNet, load, save, state_hash
from neuralfuel.prep import FEATURES
from neuralfuel.train import TrainConfig, train
from neuralfuel.validate import validate

SHARED_XS = Path(__file__).parent.parent / "models/a320/nf-a320-cfm56_5b-xs.pt"
# aircraft settings stored in the shared A320 models
SHARED_A320 = {
    "mmo": 0.82,
    "vmo_kt": 350.0,
    "mass_kg": (42600.0, 78000.0),
    "alt_ft": (10000.0, 39800.0),
    "wing_area_m2": 122.0,
}


def quiet(_):
    pass


def cfg(**kw):
    c = TrainConfig(device="cpu", threads=2, **kw)
    c.loss.n_colloc = 256
    return c


def test_names_and_sizes():
    assert model_name("a320", "cfm56_5b") == "nf-a320-cfm56_5b"
    assert model_name("A320", "CFM56_5B", "xs") == "nf-a320-cfm56_5b-xs"
    assert model_name("a320") == "nf-a320"
    assert model_name("a320", size="xs") == "nf-a320-xs"
    with pytest.raises(ValueError):
        model_name("a320", "cfm56-5b")
    with pytest.raises(ValueError):
        model_name("a320", size="huge")
    assert SIZES == {"standard": (64, 3), "xs": (32, 3)}
    assert size_of(64, 3) == "standard" and size_of(32, 3) == "xs"
    assert size_of(128, 3) == "custom"


def test_aircraft_settings():
    from aerox import load_aircraft

    a = aircraft("A320", "CFM56_5B").check()
    assert a.type == "a320" and a.engine == "cfm56_5b"
    ref = load_aircraft("A320")
    assert a.mmo == ref.mmo and a.vmo_kt == ref.vmo
    assert a.mass_kg == (ref.oew, ref.mtow) and a.alt_ft == (ALT_LOW_FT, ref.ceiling)
    assert a.wing_area_m2 == ref.wing.area
    assert (a.mmo, a.vmo_kt, a.mass_kg, a.alt_ft, a.wing_area_m2) == (
        0.82,
        350.0,
        (42600.0, 78000.0),
        (10000.0, 39800.0),
        122.0,
    )
    assert isinstance(a.vmo_kt, float)  # --set aircraft.vmo_kt takes a number
    # a type that aerox does not have: all values NaN
    u = aircraft("x1")
    assert all(math.isnan(v) for v in (u.mmo, u.vmo_kt, *u.mass_kg, *u.alt_ft))
    assert math.isnan(u.wing_area_m2)
    with pytest.raises(ValueError, match="not in aerox"):
        u.check()
    # a value that aerox does not have (PC12: no MMO) is NaN
    b = aircraft("pc12")
    assert math.isnan(b.mmo) and b.vmo_kt > 0
    with pytest.raises(ValueError, match="envelope values missing"):
        b.check()
    b = Aircraft("x1", mmo=0.7, vmo_kt=300.0, mass_kg=(20e3, 40e3), alt_ft=(5e3, 3e4))
    env = envelope(b.check(), LossConfig())
    assert env.mach == (0.45, 0.7) and env.cas_kt == (180.0, 300.0)
    assert env.alt_ft == (5e3, 3e4) and env.mass_kg == (20e3, 40e3)


def test_step_budget(syn_prepared):
    blocks, windows, _ = syn_prepared
    n = int((blocks.split == "train").sum())
    _, rec = train(
        blocks, windows, cfg(max_steps=30, eval_every=10, batch=4096), log=quiet
    )
    spe = math.ceil(n / 4096)
    assert rec["budget"] == "steps" and rec["steps_run"] == 30
    assert rec["steps_per_epoch"] == spe
    assert [h["step"] for h in rec["history"]] == [10, 20, 30]
    assert rec["epochs_run"] == pytest.approx(30 / spe)
    assert rec["best_step"] in (10, 20, 30)
    assert rec["config"]["max_steps"] == 30 and rec["config"]["max_epochs"] is None
    # the budget is in steps: with a smaller batch the same steps are fewer passes
    _, rec2 = train(
        blocks, windows, cfg(max_steps=30, eval_every=10, batch=1024), log=quiet
    )
    assert rec2["steps_run"] == 30 and rec2["epochs_run"] < rec["epochs_run"]


def test_step_patience(syn_prepared):
    blocks, windows, _ = syn_prepared
    _, rec = train(
        blocks,
        windows,
        cfg(max_steps=400, eval_every=5, patience_steps=10, lr=0.0),
        log=quiet,
    )
    # with lr 0 the validation loss never improves after the first evaluation
    assert rec["best_step"] == 5 and rec["steps_run"] == 15


def test_epoch_budget_kept(syn_prepared):
    blocks, windows, _ = syn_prepared
    n = int((blocks.split == "train").sum())
    spe = math.ceil(n / 1024)
    _, rec = train(blocks, windows, cfg(max_epochs=2, batch=1024), log=quiet)
    assert rec["budget"] == "epochs" and rec["epochs_run"] == 2
    assert rec["steps_run"] == 2 * spe
    assert [h["step"] for h in rec["history"]] == [spe, 2 * spe]


def test_lr_schedule(syn_prepared):
    """Cosine over the steps, from lr to 0.01 lr."""
    blocks, windows, _ = syn_prepared
    seen = []
    orig = torch.optim.AdamW.step

    def spy(self, *a, **k):
        seen.append(self.param_groups[0]["lr"])
        return orig(self, *a, **k)

    torch.optim.AdamW.step = spy
    try:
        train(blocks, windows, cfg(max_steps=20, eval_every=20), log=quiet)
    finally:
        torch.optim.AdamW.step = orig
    lr = TrainConfig().lr
    ref = [
        lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * k / 20))) for k in range(20)
    ]
    np.testing.assert_allclose(seen, ref, rtol=1e-6)


def test_xs_size_and_aircraft_in_file(syn_prepared, tmp_path):
    blocks, windows, _ = syn_prepared
    c = cfg(max_steps=5, eval_every=5, hidden=32, aircraft=aircraft("a320", "cfm56_5b"))
    model, rec = train(blocks, windows, c, log=quiet)
    assert rec["size"] == "xs" and rec["n_params"] == 2337
    assert rec["aircraft"]["engine"] == "cfm56_5b" and rec["aircraft"]["mmo"] == 0.82
    save(model, tmp_path / "m.pt")
    m2 = load(tmp_path / "m.pt")
    assert m2.aircraft == model.aircraft and m2.size == "xs"
    rep = validate(m2, blocks, windows)
    assert rep["model"]["size"] == "xs"
    assert rep["model"]["name"] == "nf-a320-cfm56_5b-xs"
    assert rep["model"]["aircraft"]["type"] == "a320"
    assert rep["constraints"]["range_boundary"]["mach"] == 0.82


def test_mmo_moves_the_constraints(syn_prepared):
    blocks, windows, _ = syn_prepared
    ac = aircraft("a320")
    ac.mmo = 0.78
    model, _ = train(
        blocks, windows, cfg(max_steps=5, eval_every=5, aircraft=ac), log=quiet
    )
    rep = validate(model, blocks, windows)
    assert rep["constraints"]["range_boundary"]["mach"] == 0.78
    assert all(r["mach_max_sr"] <= 0.78 for r in rep["temperature_test"]["rows"])


def test_model_without_aircraft(syn_prepared, tmp_path):
    """A network made in code without aircraft settings is saved and loaded without
    them and validates with them given."""
    blocks, windows, _ = syn_prepared
    torch.manual_seed(0)
    model = FuelFlowNet.from_data(blocks[blocks.split == "train"])
    save(model, tmp_path / "m.pt")
    m2 = load(tmp_path / "m.pt")
    assert m2.aircraft is None and state_hash(m2) == state_hash(model)
    x = blocks[FEATURES].iloc[:50]
    np.testing.assert_array_equal(predict(m2, x), predict(model, x))
    with pytest.raises(ValueError):
        validate(m2, blocks, windows)
    rep = validate(m2, blocks, windows, aircraft=aircraft("a320"))
    assert rep["model"]["aircraft_from_file"] is False
    # continued training gives the file the aircraft settings of the configuration
    m3, rec = train(blocks, windows, cfg(max_steps=2, eval_every=2), init=m2, log=quiet)
    assert m3.aircraft == aircraft("a320") and rec["aircraft"]["mmo"] == 0.82


def test_set_overrides_aerox(capsys, syn_prepared, tmp_path):
    blocks, windows, _ = syn_prepared
    prep = write_prep(tmp_path / "prep", blocks, windows)
    code, out = run_cli(
        capsys,
        "train",
        str(prep),
        "--out",
        str(tmp_path / "run"),
        *("--steps", "2", "--device", "cpu", "--threads", "2"),
        *("--set", "eval_every=2", "--set", "loss.n_colloc=256"),
        *("--set", "aircraft.mass_kg=[45000,78000]"),
        *("--set", "aircraft.alt_ft=[10000,39800]"),
        *("--set", "aircraft.wing_area_m2=122.6"),
    )
    assert code == 0
    assert out["aircraft"]["mass_kg"] == [45000.0, 78000.0]
    assert out["aircraft"]["alt_ft"] == [10000.0, 39800.0]
    assert out["aircraft"]["wing_area_m2"] == 122.6
    assert out["aircraft"]["mmo"] == aircraft("a320").mmo  # from aerox
    m = load(tmp_path / "run" / "model.pt")
    assert m.aircraft.mass_kg == (45000.0, 78000.0)
    # a type that aerox does not have needs the values as settings
    code, out = run_cli(
        capsys,
        "train",
        str(prep),
        "--out",
        str(tmp_path / "run2"),
        "--aircraft",
        "x1",
        *("--steps", "2", "--device", "cpu", "--threads", "2"),
        *("--set", "eval_every=2", "--set", "loss.n_colloc=256"),
    )
    assert code == 2 and "not in aerox" in out["error"]


def test_shared_model_keeps_stored_settings(capsys, syn_prepared, tmp_path):
    """A loaded shared model uses its stored aircraft settings, not aerox, and a model
    continued from it keeps them, so that it can be merged with the base."""
    base = load(SHARED_XS)
    ac = base.aircraft
    assert (ac.mmo, ac.vmo_kt, ac.mass_kg, ac.alt_ft, ac.wing_area_m2) == tuple(
        SHARED_A320.values()
    )
    assert ac == aircraft("a320", "cfm56_5b")  # trained with the aerox values
    blocks, windows, _ = syn_prepared
    prep = write_prep(tmp_path / "prep", blocks, windows)
    code, out = run_cli(
        capsys,
        "train",
        str(prep),
        "--out",
        str(tmp_path / "run"),
        "--init",
        str(SHARED_XS),
        *("--steps", "2", "--device", "cpu", "--threads", "2"),
        *("--set", "eval_every=2", "--set", "loss.n_colloc=256"),
        *("--set", "loss.n_teacher_points=2048"),
    )
    assert code == 0 and out["init_weights_sha256"] == state_hash(base)
    m = load(tmp_path / "run" / "model.pt")
    assert m.aircraft == ac
    assert out["aircraft"] == json.loads(json.dumps(ac.as_dict()))
    assert differences(base, m) == []
    rec = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run_record_problems(m, rec, base=base) == []


def test_release_cli(capsys, syn_prepared, tmp_path):
    blocks, windows, _ = syn_prepared
    prep = write_prep(tmp_path / "prep", blocks, windows)
    code, out = run_cli(
        capsys,
        "release",
        str(prep),
        "--out",
        str(tmp_path / "models"),
        "--engine",
        "cfm56_5b",
        "--seeds",
        "0,1",
        "--steps",
        "4",
        "--set",
        "eval_every=2",
        "--set",
        "loss.n_colloc=256",
        "--device",
        "cpu",
        "--threads",
        "2",
    )
    assert code == 0, out
    names = sorted(p.name for p in (tmp_path / "models").iterdir())
    assert names == sorted(
        f"nf-a320-cfm56_5b{s}.{e}"
        for s in ("", "-xs")
        for e in ("pt", "onnx", "run.json", "validation.json")
    )
    for size, stem in (("standard", "nf-a320-cfm56_5b"), ("xs", "nf-a320-cfm56_5b-xs")):
        r = out["models"][size]
        assert r["name"] == stem and set(r["seed_val_losses"]) == {"0", "1"}
        assert r["val_loss"] == min(r["seed_val_losses"].values())
        m = load(tmp_path / "models" / f"{stem}.pt")
        assert m.size == size and m.aircraft.engine == "cfm56_5b"
        rec = json.loads((tmp_path / "models" / f"{stem}.run.json").read_text())
        rep = json.loads((tmp_path / "models" / f"{stem}.validation.json").read_text())
        assert rec["weights_sha256"] == rep["model"]["weights_sha256"] == state_hash(m)
        assert rec["size"] == size and rep["model"]["name"] == stem
    code, out = run_cli(
        capsys,
        "train",
        str(prep),
        "--out",
        str(tmp_path / "run"),
        "--size",
        "xs",
        "--aircraft",
        "x1",
        "--steps",
        "2",
        "--device",
        "cpu",
    )
    assert code == 2 and "envelope values missing" in out["error"]


def test_release_jobs_same_models(capsys, syn_prepared, tmp_path):
    """Trainings in parallel processes give the same models as one after the other."""
    blocks, windows, _ = syn_prepared
    prep = write_prep(tmp_path / "prep", blocks, windows)
    hashes = {}
    for jobs in ("1", "2"):
        out_dir = tmp_path / f"models{jobs}"
        code, out = run_cli(
            capsys,
            "release",
            str(prep),
            "--out",
            str(out_dir),
            "--seeds",
            "0,1",
            "--sizes",
            "xs",
            "--steps",
            "4",
            "--set",
            "eval_every=2",
            "--set",
            "loss.n_colloc=256",
            "--device",
            "cpu",
            "--threads",
            "1",
            "--jobs",
            jobs,
        )
        assert code == 0, out
        rec = json.loads(next(out_dir.glob("*.run.json")).read_text())
        hashes[jobs] = (rec["weights_sha256"], rec["seed_val_losses"])
    assert hashes["1"] == hashes["2"]


def test_windows_need_positive_fuel_flow(syn_samples, syn_prepared):
    """A steady window with zero fuel flow is not kept; a non-finite loss stops the
    training with a clear error."""
    from neuralfuel.prep import prepare

    blocks, windows, _ = syn_prepared
    d = syn_samples.drop(columns=["ff1_kgh", "ff2_kgh"])
    n = len(prepare(d)[1])
    d = d.copy()
    win = windows[windows.split == "train"].iloc[0]
    sel = d.flight_id == win.flight_id
    t = (d.time - d.time[sel].min()).dt.total_seconds()
    d.loc[sel & t.between(win.time_s, win.time_s + 299), "ff_kgh"] = 0.0
    _, w, _ = prepare(d)
    assert np.isfinite(w[["ln_ff", *FEATURES]].to_numpy()).all() and len(w) == n - 1
    bad = windows.copy()
    bad.loc[bad.split == "train", "ln_ff"] = -np.inf
    with pytest.raises(ValueError, match="non-finite training loss at update step 1"):
        train(blocks, bad, cfg(max_steps=2, eval_every=1), log=quiet)


def test_run_record_is_standard_json(syn_prepared, tmp_path):
    from neuralfuel.train import write_run

    blocks, windows, _ = syn_prepared
    c = cfg(max_steps=1, eval_every=1)
    c.aircraft.type, c.aircraft.wing_area_m2 = "example", math.nan
    model, rec = train(blocks, windows, c, log=quiet)
    _, rp = write_run(model, rec, tmp_path)
    text = rp.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["config"]["aircraft"]["wing_area_m2"] is None


def test_validation_uses_constraint_settings_of_the_model(syn_prepared, tmp_path):
    """A type with a 28,000 ft ceiling, trained with the P6 band from 23,000 ft."""
    blocks, windows, _ = syn_prepared
    c = cfg(max_steps=2, eval_every=2)
    c.aircraft.alt_ft = (10000.0, 28000.0)
    c.loss.p6_alt_min_ft = 23000.0
    model, _ = train(blocks, windows, c, log=quiet)
    save(model, tmp_path / "m.pt")
    m = load(tmp_path / "m.pt")
    assert m.constraints["p6_alt_min_ft"] == 23000.0
    from neuralfuel.validate import ValidationConfig

    rep = validate(m, blocks, windows, cfg=ValidationConfig(grid_n=(4, 4, 3, 3, 2)))
    assert rep["constraint_settings"]["p6_alt_min_ft"] == 23000.0
    # a continued model keeps the constraint settings of the given model
    _, rec = train(blocks, windows, cfg(max_steps=2, eval_every=2), init=m, log=quiet)
    assert rec["config"]["loss"]["p6_alt_min_ft"] == 23000.0


def test_release_size_follows_the_network(capsys, syn_prepared, tmp_path):
    blocks, windows, _ = syn_prepared
    prep = write_prep(tmp_path / "prep", blocks, windows)
    for kv in ("hidden=32", "layers=2"):
        code, out = run_cli(
            capsys,
            *(
                "release",
                str(prep),
                "--out",
                str(tmp_path / "m"),
                "--sizes",
                "standard",
            ),
            *("--set", kv, "--steps", "1", "--device", "cpu"),
        )
        assert code == 2 and "--sizes" in out["error"]
