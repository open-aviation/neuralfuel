"""Drag-rise term P7: aerox table, penalty, default and availability, stored settings,
validation and continued training."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import run_cli, write_prep

from neuralfuel import atmos, drag
from neuralfuel.aircraft import aircraft
from neuralfuel.losses import (
    DRAG_RISE,
    LossConfig,
    drag_rise_penalty,
    drag_rise_setup,
    drag_rise_states,
    model_loss_config,
)
from neuralfuel.model import load, save, state_hash
from neuralfuel.privacy import allowed, clean_json
from neuralfuel.quality import check_model
from neuralfuel.support import union_envelope
from neuralfuel.train import RUN_RECORD, TrainConfig, train, write_run
from neuralfuel.validate import REPORT, drag_rise_test, validate
from neuralfuel.validate import drag_rise_states as vdrag_states

ENV = {"cruise": {"max_cl": 0.678}}


def short_cfg(**loss):
    cfg = TrainConfig(max_steps=6, eval_every=3, device="cpu", threads=2)
    cfg.loss.n_colloc = 256
    for k, v in loss.items():
        setattr(cfg.loss, k, v)
    return cfg


class Linear(torch.nn.Module):
    """ln F = c M (a function with f_M = c everywhere)."""

    def __init__(self, c):
        super().__init__()
        self.c = c

    def ln_ff_state(self, p):
        return self.c * p[:, 0] + 0 * p[:, 0] ** 2  # f_M depends on p (f_MM = 0)


def test_table_against_aerox():
    ac = aircraft("a320", "cfm56_5b")
    settings, table = drag_rise_setup(ac, LossConfig(), ENV)
    assert settings["cl_max"] == pytest.approx(1.1 * 0.678)
    assert settings["table_sha256"] == drag.table_hash(
        drag.build_table(settings["table"])
    )
    model, area = drag.aerox_drag_model("a320")
    rng = np.random.default_rng(0)
    n = 5000
    m = rng.uniform(0.70, 0.82, n)
    cl = rng.uniform(0.15, settings["cl_max"], n)
    ld = np.log(atmos.isa_delta(rng.uniform(10000, 39800, n)))
    direct = drag.drag_slope(model, area, ac.wing_area_m2, m, cl, ld)
    spec = settings["table"]
    interp = drag.lookup(table, spec, m, cl, ld)
    assert np.abs(interp - direct).max() < 0.01
    t = [torch.tensor(v) for v in (m, cl, ld)]
    assert np.allclose(drag.lookup(table, spec, *t).numpy(), interp)
    # the slope has the drag rise: far above 1 / M at the MMO
    g = drag.drag_slope(model, area, ac.wing_area_m2, 0.82, 0.5, np.log(0.2))
    assert g > 2.0


def test_penalty_zero_when_bound_holds():
    ac = aircraft("a320", "cfm56_5b")
    cfg = LossConfig()
    settings, table = drag_rise_setup(ac, cfg, ENV)
    gen = torch.Generator().manual_seed(0)
    p = drag_rise_states(2048, gen, "cpu", ac, cfg, settings["cl_max"])
    assert (p[:, 0] >= 0.70).all() and (p[:, 3] == 0).all()
    cl = drag.level_cl(p[:, 0], p[:, 1], p[:, 2], ac.wing_area_m2)
    assert (cl <= settings["cl_max"] + 1e-6).all()
    tt = torch.tensor(table, dtype=torch.float32)
    assert drag_rise_penalty(Linear(10.0), p, tt, settings, ac.wing_area_m2) == 0
    assert drag_rise_penalty(Linear(0.0), p, tt, settings, ac.wing_area_m2) > 0.1


def test_default_on_settings_stored_and_used(syn_prepared, tmp_path):
    """P7 is on by default for the A320; its settings go into the model file and the
    run record, the validation tests them, and a continued model keeps them."""
    assert LossConfig().drag_rise == 0.01
    blocks, windows, _ = syn_prepared
    model, rec = train(
        blocks, windows, short_cfg(drag_rise_mach_min=0.75), log=lambda s: None
    )
    dr = rec["drag_rise"]
    assert dr["weight"] == 0.01 and dr["mach_min"] == 0.75 and dr["cl_factor"] == 1.1
    assert set(dr) == set(DRAG_RISE) and "drag_rise_note" not in rec
    assert dr["table"]["mach"][:2] == [0.75, 0.82] and dr["aerox_version"]
    assert "drag_rise" in rec["history"][-1]
    assert model.constraints["drag_rise"] == dr
    assert allowed(clean_json(rec), RUN_RECORD)[1] == []
    mp, _ = write_run(model, rec, tmp_path / "run")
    m2 = load(mp)
    assert m2.constraints["drag_rise"] == dr
    assert model_loss_config(m2).drag_rise_mach_min == 0.75
    report = validate(m2, blocks, windows)
    assert allowed(report, REPORT)[1] == []
    v = report["constraints"]["drag_rise"]
    assert v["points"] > 0 and v["table_sha256_match"] is True
    assert v["mach"] == [0.75, 0.82] and 0 <= v["violation_pct"] <= 100
    assert report["constraint_settings"]["drag_rise"] == dr
    # a model continued from it keeps the settings, also against other cfg values
    m3, rec3 = train(
        blocks, windows, short_cfg(drag_rise=0.0), init=m2, log=lambda s: None
    )
    assert rec3["drag_rise"] == dr and m3.constraints["drag_rise"] == dr
    assert "drag_rise" in rec3["history"][-1]
    # the quality check compares the P7 violation rate
    res = check_model(m3, m2, rec3)
    note = next(r for r in res["checks"] if r["check"] == "constraint_delta")["note"]
    assert "drag_rise" in note


def test_weight_zero_switches_the_term_off(syn_prepared, monkeypatch):
    """With weight 0 no part of P7 runs; a model without P7 settings validates
    without the P7 test and is continued without P7."""
    blocks, windows, _ = syn_prepared

    def fail(*a, **k):
        raise AssertionError("P7 used with weight 0")

    for name in ("drag_rise_setup", "drag_rise_states", "drag_rise_penalty"):
        monkeypatch.setattr(f"neuralfuel.train.{name}", fail)
    lines = []
    model, rec = train(blocks, windows, short_cfg(drag_rise=0.0), log=lines.append)
    assert rec["drag_rise"] is None and rec["drag_rise_note"] == "loss.drag_rise is 0"
    assert "drag_rise" not in model.constraints
    assert "drag-rise term P7 off: loss.drag_rise is 0" in lines
    assert allowed(clean_json(rec), RUN_RECORD)[1] == []
    report = validate(model, blocks, windows)
    assert "drag_rise" not in report["constraints"]
    assert "drag_rise" not in report["constraint_settings"]
    assert model_loss_config(model).drag_rise == 0
    # continued with the default weight: still off, the settings come from the model
    m2, rec2 = train(blocks, windows, short_cfg(), init=model, log=lambda s: None)
    assert rec2["drag_rise"] is None and "drag_rise" not in m2.constraints
    assert rec2["drag_rise_note"] == "the given model has no P7 settings"


@pytest.mark.parametrize("sets", [(), ("--set", "loss.drag_rise=0.01")])
def test_type_without_aerox_runs_without_the_term(capsys, syn_prepared, tmp_path, sets):
    """A type without an aerox drag model trains without P7 and says so, with the
    default weight and with an explicit positive weight."""
    blocks, windows, _ = syn_prepared
    prep = write_prep(tmp_path / "prep", blocks, windows)
    run = tmp_path / "run"
    code, out = run_cli(
        capsys,
        *("train", str(prep), "--out", str(run), "--aircraft", "x1"),
        *("--steps", "2", "--device", "cpu", "--threads", "2"),
        *("--set", "aircraft.mmo=0.82", "--set", "aircraft.vmo_kt=350"),
        *("--set", "aircraft.mass_kg=[42000,78000]"),
        *("--set", "aircraft.alt_ft=[10000,39000]"),
        *("--set", "aircraft.wing_area_m2=122", "--set", "loss.n_colloc=256"),
        *sets,
    )
    assert code == 0, out
    reason = "aerox has no type 'x1'"
    assert out["notes"] == [f"drag-rise term P7 off: {reason}"]
    rec = json.loads((run / "run.json").read_text())
    assert rec["drag_rise"] is None and rec["drag_rise_note"] == reason
    assert "drag_rise" not in load(run / "model.pt").constraints


def test_continued_training_checks_the_table(
    capsys, syn_prepared, tmp_path, monkeypatch
):
    """train --init rebuilds the table of the shared model and stops (exit code 2)
    when it does not have the stored hash: other aerox values or no stored hash."""
    blocks, windows, _ = syn_prepared
    prep = write_prep(tmp_path / "prep", blocks, windows)
    shared = Path(__file__).parent.parent / "models/a320/nf-a320-cfm56_5b-xs.pt"
    stored = load(shared).constraints["drag_rise"]

    def cont(init, out):
        return run_cli(
            capsys,
            *("train", str(prep), "--init", str(init), "--out", str(tmp_path / out)),
            *("--steps", "2", "--device", "cpu", "--threads", "2"),
            *("--set", "loss.n_colloc=256"),
        )

    code, out = cont(shared, "same")
    assert code == 0, out
    rec = json.loads((tmp_path / "same/run.json").read_text())
    assert rec["drag_rise"]["table_sha256"] == stored["table_sha256"]
    # the continued model records the installed aerox and passes 'drag_table'
    assert rec["drag_rise"]["aerox_version"] == drag.aerox_version()
    res = check_model(load(tmp_path / "same/model.pt"), load(shared), rec)
    row = next(r for r in res["checks"] if r["check"] == "drag_table")
    assert row["status"] == "pass", row
    build = drag.build_table
    monkeypatch.setattr(drag, "build_table", lambda spec: build(spec) + 1e-6)
    monkeypatch.setattr(drag, "aerox_version", lambda: "9.9")
    code, out = cont(shared, "other")
    assert code == 2 and not (tmp_path / "other/model.pt").exists()
    assert "of the shared model cannot be reproduced" in out["error"]
    assert "installed aerox 9.9" in out["error"]
    assert f"built with aerox {stored['aerox_version']}" in out["error"]
    monkeypatch.undo()
    m = load(shared)
    del m.constraints["drag_rise"]["table_sha256"]
    save(m, tmp_path / "nohash.pt")
    code, out = cont(tmp_path / "nohash.pt", "nohash")
    assert code == 2 and "have no table_sha256" in out["error"]


def test_quality_check_rebuilds_the_tables(capsys, tmp_path, monkeypatch):
    """check-model rebuilds the tables of P7 of the base and of the contribution:
    the shared model against itself passes, other aerox values fail 'drag_table',
    and P7 settings without a stored hash are bad input (exit code 2)."""
    shared = Path(__file__).parent.parent / "models/a320/nf-a320-cfm56_5b-xs.pt"
    m = load(shared)
    version = m.constraints["drag_rise"]["aerox_version"]
    res = check_model(m, m)
    row = next(r for r in res["checks"] if r["check"] == "drag_table")
    assert res["ok"] and row["status"] == "pass" and row["value"] is True
    assert f"base: table of aerox {version} reproduced" in row["note"]
    assert f"installed aerox {drag.aerox_version()}" in row["note"]
    # a NaN weight keeps the check short: it stops after the flag checks
    nan = load(shared)
    with torch.no_grad():
        nan.net[0].weight[0, 0] = float("nan")
    build = drag.build_table
    monkeypatch.setattr(drag, "build_table", lambda spec: build(spec) + 1e-6)
    monkeypatch.setattr(drag, "aerox_version", lambda: "9.9")
    res = check_model(nan, m)
    row = next(r for r in res["checks"] if r["check"] == "drag_table")
    assert not res["ok"] and row["status"] == "fail" and row["value"] is False
    assert f"contribution: table of aerox {version} not reproduced" in row["note"]
    assert "installed aerox 9.9" in row["note"]
    monkeypatch.undo()
    nohash = load(shared)
    del nohash.constraints["drag_rise"]["table_sha256"]
    save(nohash, tmp_path / "nohash.pt")
    code, out = run_cli(
        capsys,
        *("check-model", str(tmp_path / "nohash.pt"), "--base", str(shared)),
        *("--run", str(shared.with_suffix(".run.json"))),
    )
    assert code == 2 and "have no table_sha256" in out["error"]
    dr = nohash.constraints["drag_rise"]
    with pytest.raises(ValueError, match="have no table_sha256"):
        drag_rise_test(nohash, model_loss_config(nohash), nohash.aircraft, dr)


def test_quality_check_needs_the_p7_settings_of_the_base():
    """A contribution with other P7 settings than the base fails 'drag_table'."""
    shared = Path(__file__).parent.parent / "models/a320/nf-a320-cfm56_5b-xs.pt"
    base = load(shared)
    for change in ("mach_min", "drop"):
        c = load(shared)
        if change == "drop":
            del c.constraints["drag_rise"]
        else:
            c.constraints["drag_rise"]["mach_min"] += 0.01
        res = check_model(c, base)
        row = next(r for r in res["checks"] if r["check"] == "drag_table")
        assert not res["ok"] and row["status"] == "fail", change
        assert "P7 settings of the contribution" in row["note"]
    res = check_model(load(shared), base)
    assert next(r for r in res["checks"] if r["check"] == "drag_table")["value"]
    # only the aerox version differs (continued training records the installed one)
    c = load(shared)
    c.constraints["drag_rise"]["aerox_version"] = "0.0.1"
    res = check_model(c, base)
    row = next(r for r in res["checks"] if r["check"] == "drag_table")
    assert (
        row["status"] == "pass" and "contribution: table of aerox 0.0.1" in row["note"]
    )


def test_quality_check_builds_only_the_table_of_the_base(monkeypatch):
    """check-model compares the P7 settings first and builds only the table of the
    base, so the grid of a contribution does not set the size of a table; the table
    builder refuses a grid that is too large."""
    shared = Path(__file__).parent.parent / "models/a320/nf-a320-cfm56_5b-xs.pt"
    base, c = load(shared), load(shared)
    c.constraints["drag_rise"]["table"]["mach"] = [0.7, 0.82, 10**9]
    with torch.no_grad():
        c.net[0].weight[0, 0] = float("nan")  # stops after the flag checks
    built = []
    build = drag.build_table
    monkeypatch.setattr(
        drag, "build_table", lambda spec: built.append(spec) or build(spec)
    )
    res = check_model(c, base)
    row = next(r for r in res["checks"] if r["check"] == "drag_table")
    assert row["status"] == "fail" and "was not rebuilt" in row["note"]
    assert built == [base.constraints["drag_rise"]["table"]]
    monkeypatch.undo()
    for spec in (
        {"mach": [0.7, 0.82, 10**9], "cl": [0.05, 0.8, 76], "ln_delta": [-2, -0.4, 12]},
        {
            "mach": [0.7, 0.82, 1000],
            "cl": [0.05, 0.8, 1000],
            "ln_delta": [-2, -0.4, 12],
        },
        {"mach": [0.7, "x", 25], "cl": [0.05, 0.8, 76], "ln_delta": [-2, -0.4, 12]},
    ):
        with pytest.raises(ValueError, match="P7 table"):
            drag.build_table(
                {"type": "a320", "wing_area_m2": 122.0, "dmach": 1e-4, **spec}
            )


def test_merge_without_quality_check_fails(capsys, tmp_path):
    """merge writes the files but exits with code 1 when the quality check of the
    merged model cannot run (here: P7 settings of the base without table_sha256)."""
    shared = Path(__file__).parent.parent / "models/a320/nf-a320-cfm56_5b-xs.pt"
    base = load(shared)
    del base.constraints["drag_rise"]["table_sha256"]
    save(base, tmp_path / "base.pt")
    dirs = []
    for i, b in enumerate((0.001, 0.002)):
        c = load(shared)
        with torch.no_grad():
            c.net[-1].bias.add_(b)
        c.envelope = union_envelope(base.envelope, base.envelope)
        rec = {
            "weights_sha256": state_hash(c),
            "init_weights_sha256": state_hash(base),
            "local_envelope": base.envelope,
        }
        write_run(c, rec, tmp_path / f"c{i}")
        dirs.append(str(tmp_path / f"c{i}"))
    out_pt = tmp_path / "merged.pt"
    code, out = run_cli(
        capsys,
        "merge",
        *dirs,
        "--base",
        str(tmp_path / "base.pt"),
        "--out",
        str(out_pt),
    )
    assert code == 1 and not out["ok"]
    assert out["quality_check"]["ok"] is None
    assert "have no table_sha256" in out["quality_check"]["error"]
    mr = json.loads((tmp_path / "merged.merge.json").read_text())
    assert out_pt.exists() and mr["quality_check"]["ok"] is None


def test_unavailable_reasons():
    ac = aircraft("a320")
    with pytest.raises(drag.Unavailable, match="Mach band is empty"):
        drag_rise_setup(ac, LossConfig(drag_rise_mach_min=0.9), ENV)
    with pytest.raises(drag.Unavailable, match="largest lift coefficient"):
        drag_rise_setup(ac, LossConfig(), {})
    with pytest.raises(drag.Unavailable, match="no wing area"):
        drag_rise_setup(
            dataclasses.replace(ac, wing_area_m2=math.nan), LossConfig(), ENV
        )
    with pytest.raises(ValueError, match="cl_factor"):
        drag_rise_setup(ac, LossConfig(drag_rise_cl_factor=0.0), ENV)


def test_validation_tests_the_drag_slope():
    ac = aircraft("a320", "cfm56_5b")
    lc = LossConfig()
    settings, table = drag_rise_setup(ac, lc, ENV)
    q, cl = vdrag_states(lc, ac, settings)
    g = drag.lookup(table, settings["table"], q[:, 0].numpy(), cl, q[:, 1].numpy())
    res = drag_rise_test(Linear(1.6), lc, ac, settings)
    assert res["min_margin"] == pytest.approx(1.6 - g.max(), abs=1e-9)
    assert res["violation_pct"] == pytest.approx(100 * (1.6 - g < -0.05).mean())
    assert 0 < res["violation_pct"] < 100 and res["table_sha256_match"] is True
