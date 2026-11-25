"""End-to-end CLI cycle on the synthetic dataset."""

import json
import shutil

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
import torch
from conftest import run_cli

from neuralfuel import synthetic
from neuralfuel.checks import profile
from neuralfuel.cli import main
from neuralfuel.inference import onnx_metadata
from neuralfuel.model import load, read_meta, save, state_hash
from neuralfuel.privacy import find_identifiers
from neuralfuel.support import random_inputs


def ok(*argv):
    assert main([*argv, "--json"]) == 0, argv


@pytest.fixture(scope="module")
def cycle(tmp_path_factory, syn_csv):
    """prep, a short training and the report of the test split, through the CLI."""
    work = tmp_path_factory.mktemp("cycle")
    prep, run = work / "prep", work / "run"
    ok("prep", str(syn_csv), "--adapter", "synthetic", "--out", str(prep))
    train = ["train", str(prep), "--out", str(run), "--device", "cpu"]
    ok(*train, "--epochs", "2", "--threads", "2", "--set", "loss.n_colloc=512")
    report = ["validate", str(prep), "--model", str(run / "model.pt")]
    ok(*report, "--out", str(run / "validation.json"), "--threads", "2")
    return work


def test_profile_raw_has_no_identifiers(capsys, syn_csv):
    code, out = run_cli(capsys, "profile", str(syn_csv), "--raw")
    assert code == 0 and out["rows"] > 0
    raw = pd.read_csv(syn_csv, usecols=["leg", "utc"])
    text = json.dumps(out)
    assert not any(f in text for f in raw.leg.unique())
    assert not any(t[:10] in text for t in raw.utc.iloc[[0, -1]])
    cols = {c["name"]: c for c in out["columns"]}
    assert "min" not in cols["leg"] and "min" not in cols["utc"]
    assert cols["mach"]["max"] < 0.9
    # without the adapter, a raw column that is not in the schema has no range
    assert "min" not in cols["palt_m"] and "distinct_values" not in cols["leg"]
    code, out = run_cli(
        capsys, "profile", str(syn_csv), "--raw", "--adapter", "synthetic"
    )
    cols = {c["name"]: c for c in out["columns"]}
    assert code == 0 and cols["palt_m"]["max"] > 9000  # listed in MEASUREMENTS
    assert "min" not in cols["leg"] and "min" not in cols["utc"]


def test_profile_ranges_only_for_measurements():
    df = pd.DataFrame(
        {"x_coord": [4.1, 4.2], "ac_id": [31415, 31416], "on": [True, False]}
    )
    cols = {c["name"]: c for c in profile(df)["columns"]}
    for c in cols.values():
        assert set(c) == {"name", "dtype", "missing_fraction", "ranges"}
    assert "31415" not in json.dumps(profile(df)) and "4.1" not in json.dumps(cols)
    cols = {c["name"]: c for c in profile(df, measurements=["x_coord"])["columns"]}
    assert cols["x_coord"]["max"] == 4.2 and "min" not in cols["ac_id"]


def test_profile_and_check_after_adapter(capsys, syn_csv):
    code, out = run_cli(capsys, "profile", str(syn_csv), "--adapter", "synthetic")
    assert code == 0 and out["sampling"]["median_step_s"] == 1.0
    assert "min" not in {c["name"]: c for c in out["columns"]}["time"]
    code, out = run_cli(capsys, "check", str(syn_csv), "--adapter", "synthetic")
    assert code == 0 and out["ok"], out


def test_errors(capsys, syn_csv, tmp_path):
    code, out = run_cli(
        capsys, "check", str(tmp_path / "missing.csv"), "--adapter", "synthetic"
    )
    assert code == 2 and "not found" in out["error"]
    code, out = run_cli(capsys, "check", str(syn_csv), "--adapter", "nope")
    assert code == 2 and "unknown adapter" in out["error"]
    # the canonical adapter on the raw synthetic file: schema errors name the columns
    code, out = run_cli(capsys, "check", str(syn_csv), "--adapter", "canonical")
    assert code == 1
    cols = {p["column"] for p in out["schema"] if p["level"] == "error"}
    assert {"flight_id", "time", "alt_ft", "mass_kg"} <= cols
    code, out = run_cli(
        capsys, "prep", str(syn_csv), "--adapter", "canonical", "--out", str(tmp_path)
    )
    assert code == 2 and out["column"] == "flight_id"


def test_cycle(capsys, cycle):
    prep, run, work = cycle / "prep", cycle / "run", cycle
    summary = json.loads((prep / "prep.json").read_text())
    assert summary["blocks"]["train"]["blocks"] > 0 and summary["windows"]["train"] > 0
    rec = json.loads((run / "run.json").read_text())
    assert rec["epochs_run"] == 2 and rec["config"]["loss"]["n_colloc"] == 512
    assert rec["prep"] == summary["config"]
    # P7 is on by default for the A320 (aerox has its drag model)
    assert rec["drag_rise"]["weight"] == 0.01 and "drag_rise_note" not in rec
    rep = json.loads((run / "validation.json").read_text())
    assert rep["accuracy"]["all"]["n"] > 0 and rep["prep"] == summary["config"]
    assert rep["constraints"]["drag_rise"]["table_sha256_match"] is True
    code, out = run_cli(
        capsys,
        "train",
        str(prep),
        "--out",
        str(work / "run2"),
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--threads",
        "2",
        "--init",
        str(run / "model.pt"),
        "--aircraft",
        "x9",
    )
    assert code == 0 and out["init"] == "continued from given weights"
    assert "ignored with --init" in out["notes"][0] and len(out["notes"]) == 1
    assert out["drag_rise"] == rec["drag_rise"]  # the contribution keeps P7
    code, out = run_cli(
        capsys,
        "train",
        str(prep),
        "--out",
        str(work / "run3"),
        "--init",
        str(run / "model.pt"),
        "--set",
        "aircraft.mmo=0.8",
    )
    assert code == 2 and "not possible with --init" in out["error"]
    code, out = run_cli(
        capsys,
        "export",
        "--model",
        str(run / "model.pt"),
        "--out",
        str(run / "model.onnx"),
        "--casadi",
    )
    assert code == 0 and "Squeeze" not in out["ops"] and "Softplus" not in out["ops"]
    code, out = run_cli(
        capsys,
        "contribute",
        "--run",
        str(run),
        "--report",
        str(run / "validation.json"),
        "--out",
        str(work / "share"),
    )
    assert code == 0 and out["dropped_fields"] == []  # the current formats
    names = sorted(p.name for p in (work / "share").iterdir())
    assert names == [
        "MANIFEST.json",
        "model.onnx",
        "model.pt",
        "run.json",
        "validation.json",
    ]
    assert (
        load(work / "share" / "model.pt").constraints["drag_rise"] == rec["drag_rise"]
    )
    assert (
        json.loads((work / "share" / "run.json").read_text())["drag_rise"]
        == (rec["drag_rise"])
    )
    # a report from other weights is refused
    code, out = run_cli(
        capsys,
        "contribute",
        "--run",
        str(work / "run2"),
        "--report",
        str(run / "validation.json"),
        "--out",
        str(work / "share2"),
    )
    assert code == 2 and not (work / "share2").exists()
    # a run record of other weights is refused
    stale = work / "stale"
    stale.mkdir()
    shutil.copyfile(run / "model.pt", stale / "model.pt")
    shutil.copyfile(work / "run2" / "run.json", stale / "run.json")
    code, out = run_cli(
        capsys,
        "contribute",
        "--run",
        str(stale),
        "--report",
        str(run / "validation.json"),
        "--out",
        str(work / "share4"),
    )
    assert code == 2 and "run.json" in out["error"]
    assert not (work / "share4").exists()


def contribute(capsys, run, report, out):
    return run_cli(
        capsys,
        "contribute",
        "--run",
        str(run),
        "--report",
        str(report),
        "--out",
        str(out),
    )


def test_contribute_writes_onnx_from_the_weights(capsys, cycle, tmp_path):
    run = cycle / "run"
    # --onnx is gone: the ONNX file always comes from the verified weights
    code, out = run_cli(
        capsys,
        *("contribute", "--run", str(run), "--report", str(run / "validation.json")),
        *("--onnx", str(run / "validation.json"), "--out", str(tmp_path / "s")),
    )
    assert code == 2 and "--onnx" in out["error"] and not (tmp_path / "s").exists()
    code, out = contribute(capsys, run, run / "validation.json", tmp_path / "s")
    assert code == 0
    model = load(tmp_path / "s" / "model.pt")
    assert state_hash(model) == state_hash(load(run / "model.pt"))
    x = random_inputs(model.envelope, 200, seed=1).astype(np.float32)
    sess = ort.InferenceSession(str(tmp_path / "s" / "model.onnx"))
    (y,) = sess.run(None, {"x": x})
    with torch.no_grad():
        np.testing.assert_allclose(y.ravel(), model(torch.from_numpy(x)), atol=1e-4)
    assert onnx_metadata(tmp_path / "s" / "model.onnx")["envelope"] == model.envelope


def test_contribute_needs_new_or_empty_folder(capsys, cycle, tmp_path):
    run, rep = cycle / "run", cycle / "run" / "validation.json"
    share = tmp_path / "share"
    share.mkdir()
    (share / "offsets.json").write_text('{"private": "synthetic example"}')
    code, out = contribute(capsys, run, rep, share)
    assert code == 2 and "not empty" in out["error"]
    assert sorted(p.name for p in share.iterdir()) == ["offsets.json"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert contribute(capsys, run, rep, empty)[0] == 0
    code, out = contribute(capsys, run, rep, empty)  # a second run into the same folder
    assert code == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["empty", "share"]  # no stage


def test_contribute_copies_allowed_fields_only(capsys, cycle, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(cycle / "run", run)
    model = load(run / "model.pt")
    model.envelope["note"] = "local/private-example.csv"
    meta = {**read_meta(run / "model.pt"), "source_path": "local/private-example.csv"}
    save(model, run / "model.pt", meta=meta)
    rec = json.loads((run / "run.json").read_text())
    rec["notes"] = "local/private-example.csv"
    rec["protection"]["comment"] = "local/private-example.csv"
    rec["config"]["seed"] = "local/private-example.csv"  # a text in a number field
    (run / "run.json").write_text(json.dumps(rec))
    rep = json.loads((run / "validation.json").read_text())
    rep["trip_fuel"]["source"] = "local/private-example.csv"
    (run / "validation.json").write_text(json.dumps(rep))
    code, out = contribute(capsys, run, run / "validation.json", tmp_path / "share")
    assert code == 0
    assert set(out["dropped_fields"]) == {
        "run.json.notes",
        "run.json.protection.comment",
        "run.json.config.seed",
        "validation.json.trip_fuel.source",
        "model.pt.envelope.note",
        "model.pt.meta.source_path",
    }
    for name in ("run.json", "validation.json", "MANIFEST.json"):
        assert "private-example" not in (tmp_path / "share" / name).read_text()
    assert "private-example" not in json.dumps(
        read_meta(tmp_path / "share" / "model.pt")
    )
    assert "note" not in load(tmp_path / "share" / "model.pt").envelope
    assert set(read_meta(tmp_path / "share" / "model.pt")) == {
        "weights_sha256",
        "init_weights_sha256",
    }


def test_contribute_needs_test_split_report(capsys, cycle, tmp_path):
    run = cycle / "run"
    code, out = run_cli(
        capsys,
        *("validate", str(cycle / "prep"), "--model", str(run / "model.pt")),
        *("--splits", "train", "--out", str(tmp_path / "train.json")),
    )
    assert code == 0 and out["evaluated_splits"] == ["train"]
    code, out = contribute(capsys, run, tmp_path / "train.json", tmp_path / "share")
    assert code == 2 and "test split" in out["error"]
    assert not (tmp_path / "share").exists()
    code, out = run_cli(
        capsys,
        *("validate", str(cycle / "prep"), "--model", str(run / "model.pt")),
        *("--splits", "test,local/example", "--out", str(tmp_path / "x.json")),
    )
    assert code == 2 and "unknown split" in out["error"]


def test_cli_input_errors(capsys, cycle, tmp_path):
    prep = cycle / "prep"
    train = ["train", str(prep), "--out", str(tmp_path / "r"), "--device", "cpu"]
    for opt, words in (
        (["--steps", "0"], "--steps"),
        (["--steps", "-1"], "--steps"),
        (["--steps", "x"], "invalid int"),
        (["--set", "wrong.key=1"], "unknown setting"),
        (["--set", "loss.nope=1"], "unknown setting"),
        (["--set", "lr=abc"], "a number"),
        (["--set", "eval_every=0"], "eval_every"),
        (["--set", "loss=1"], "group of settings"),
        (["--set", "aircraft.mass_kg=[1]"], "2 numbers"),
    ):
        code, out = run_cli(capsys, *train, *opt)
        assert code == 2 and words in out["error"], (opt, out)
    assert not (tmp_path / "r").exists()
    code, out = run_cli(
        capsys, "prep", "--out", str(tmp_path / "p"), "--set", "block_s=0"
    )
    assert code == 2 and "block_s" in out["error"]
    # a file that is not a model file
    (tmp_path / "x.pt").write_text("not a model")
    code, out = run_cli(
        capsys,
        "export",
        "--model",
        str(tmp_path / "x.pt"),
        "--out",
        str(tmp_path / "o"),
    )
    assert code == 2 and "not a neuralfuel-weights" in out["error"]
    # an argument error with --json: one JSON object on stdout
    code = main(["export", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 2 and out["exit_code"] == 2 and out["command"] == "export"
    assert "required" in out["error"]
    # release takes the size from --sizes only
    code, out = run_cli(
        capsys,
        *("release", str(prep), "--out", str(tmp_path / "m"), "--sizes", "standard"),
        *("--set", "hidden=32", "--steps", "1", "--device", "cpu"),
    )
    assert code == 2 and "--sizes" in out["error"] and not (tmp_path / "m").exists()


def test_model_file_bounds(tmp_path, cycle):
    ck = torch.load(cycle / "run" / "model.pt", weights_only=True)
    for k, v in (("layers", 9), ("hidden", 2048), ("hidden", "64")):
        torch.save({**ck, k: v}, tmp_path / "big.pt")
        with pytest.raises(ValueError, match="architecture"):
            load(tmp_path / "big.pt")
    torch.save({**ck, "state": {}}, tmp_path / "empty.pt")
    with pytest.raises(ValueError, match="invalid"):
        load(tmp_path / "empty.pt")


def test_climb_descent_only_dataset(capsys, tmp_path):
    """A dataset without steady cruise windows works through prep, train, validate."""
    raw = synthetic.generate(n_flights=12, seed=2)
    raw = raw[raw.palt_m < 6000.0]  # below 19,700 ft: no cruise, no steady window
    raw.to_csv(tmp_path / "low.csv", index=False)
    prep = tmp_path / "prep"
    code, out = run_cli(
        capsys,
        "prep",
        str(tmp_path / "low.csv"),
        "--adapter",
        "synthetic",
        "--out",
        str(prep),
    )
    assert code == 0 and sum(out["windows"].values()) == 0
    code, out = run_cli(
        capsys,
        *("train", str(prep), "--out", str(tmp_path / "run"), "--steps", "4"),
        *("--set", "eval_every=2", "--set", "loss.n_colloc=256", "--device", "cpu"),
    )
    assert code == 0 and out["n_train_windows"] == 0
    code, out = run_cli(
        capsys,
        *("validate", str(prep), "--model", str(tmp_path / "run" / "model.pt")),
        *("--out", str(tmp_path / "v.json"), "--threads", "2"),
    )
    assert code == 0 and out["accuracy"]["all"]["n"] > 0
    assert out["mach_sensitivity"]["supported_cells"] == 0


def test_train_takes_phase_rule_from_prep(capsys, cycle, tmp_path):
    prep = tmp_path / "prep2"
    shutil.copytree(cycle / "prep", prep)
    rec = json.loads((prep / "prep.json").read_text())
    rec["config"]["level_fpm"] = 250.0
    (prep / "prep.json").write_text(json.dumps(rec))
    code, out = run_cli(
        capsys,
        "train",
        str(prep),
        "--out",
        str(tmp_path / "run3"),
        "--steps",
        "4",
        "--device",
        "cpu",
        "--set",
        "eval_every=2",
        "--set",
        "loss.n_colloc=256",
    )
    assert code == 0 and out["envelope"]["phase_rule"]["level_fpm"] == 250.0
    assert out["prep"]["level_fpm"] == 250.0


def test_contribute_refuses_identifiers(capsys, cycle, tmp_path):
    run = cycle / "run"
    rep = json.loads((run / "validation.json").read_text())
    rep["coverage"]["flight_id"] = "X"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(rep))
    code, out = contribute(capsys, run, bad, tmp_path / "share3")
    assert code == 1 and out["identifying_fields"] == [
        "validation.json.coverage.flight_id"
    ]
    assert find_identifiers(rep) == ["$.coverage.flight_id"]
    # a key and its value name one field once
    assert find_identifiers({"date": "2030-01-01"}) == ["$.date"]
