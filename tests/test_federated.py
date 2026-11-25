"""Federated workflow on the synthetic data: protection against forgetting,
aggregation, private local offsets, quality check and reference comparison."""

import copy
import json
import math
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import run_cli, write_prep

from neuralfuel.aggregate import (
    BUFFERS,
    average,
    combine,
    differences,
    record_problems,
)
from neuralfuel.cli import main
from neuralfuel.inference import predict
from neuralfuel.local import fit_offsets, offset_values, offsets_for, phases
from neuralfuel.losses import LossConfig, teacher_points
from neuralfuel.model import load, read_meta, save, state_hash
from neuralfuel.quality import _row, check_model
from neuralfuel.reference import compare, model_inputs, model_reference
from neuralfuel.support import in_envelope, union_envelope
from neuralfuel.train import TrainConfig, train, write_run

LB = 2.20462
SHARED = Path(__file__).parent.parent / "models" / "a320"


def cfg(steps=300, seed=0, **loss):
    c = TrainConfig(seed=seed, max_steps=steps, device="cpu", threads=2)
    for k, v in loss.items():
        setattr(c.loss, k, v)
    return c


@pytest.fixture(scope="module")
def data(syn_prepared):
    blocks, windows, _ = syn_prepared
    return blocks, windows


@pytest.fixture(scope="module")
def base(data):
    return train(*data, cfg(400), log=lambda s: None)[0]


@pytest.fixture(scope="module")
def base_cache():
    """Results of the base in check_model, kept between the calls of this module."""
    return {}


@pytest.fixture(scope="module")
def fuel_run(base, data):
    """A contribution with fuel flow 10 % too high (a scale error)."""
    return continued(base, *scaled(*data, fuel=1.1), steps=300, seed=3)


def child(model, b):
    """A copy of model with the output bias moved by b."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.net[-1].bias.add_(b)
    return m


def cruise_only(blocks, windows):
    return blocks[blocks.cruise].reset_index(drop=True), windows


def scaled(blocks, windows, mass=1.0, fuel=1.0):
    out = []
    for t in (blocks.copy(), windows.copy()):
        t["mass_kg"] = t.mass_kg * mass
        t["ln_m"] = t.ln_m + math.log(mass)
        t["ff_kgh"] = t.ff_kgh * fuel
        t["ln_ff"] = t.ln_ff + math.log(fuel)
        if "wd_t" in t:
            t["wd_t"] = t.wd_t * mass
        out.append(t)
    return out


def continued(base, blocks, windows, steps=300, seed=1, **loss):
    return train(
        blocks, windows, cfg(steps, seed, **loss), copy.deepcopy(base), lambda s: None
    )


# ---------------------------------------------------------------- protection
def test_protection_keeps_model_outside_local_region(base, data):
    b, w = cruise_only(*data)
    b = b.copy()
    b["ln_ff"] = b.ln_ff + math.log(1.05)  # the user has a different level
    prot, rec = continued(base, b, w)
    plain, rec0 = continued(base, b, w, teacher=0.0)
    assert rec["protection"]["teacher_weight"] == 10.0
    assert rec["protection"]["points_in_shared_envelope"] > 0
    assert rec0["protection"]["teacher_weight"] == 0.0
    # states of the shared model outside the cruise-only region of the user
    x = teacher_points(base.aircraft, LossConfig(), base.envelope)
    out = in_envelope(base.envelope, x) & ~in_envelope(rec["local_envelope"], x)
    assert out.sum() > 1000

    def move(m):
        return np.abs(np.log(predict(m, x[out]) / predict(base, x[out]))).mean()

    assert move(prot) < 0.5 * move(plain)
    # inside the user's region the protected model still follows the user's data
    tr = b[b.split == "train"]
    r = np.log(predict(prot, tr) / predict(base, tr)).mean()
    assert r > 0.02


def test_continued_envelope_is_union(base, data):
    b, w = cruise_only(*data)
    model, rec = continued(base, b, w, steps=50)
    loc = rec["local_envelope"]
    assert loc["phases"]["climb"]["blocks"] == 0
    assert model.envelope == rec["envelope"]
    assert model.envelope["phases"]["climb"] == base.envelope["phases"]["climb"]
    a = model.envelope["phases"]["all"]
    assert (
        a["blocks"]
        == base.envelope["phases"]["all"]["blocks"] + loc["phases"]["all"]["blocks"]
    )
    assert rec["init_weights_sha256"] == state_hash(base)


def test_train_keeps_init_and_logs(base, data):
    before = copy.deepcopy(base)
    h = state_hash(base)
    lines = []
    b = data[0].copy()
    b["tail"] = np.where(np.arange(len(b)) % 3, "A", "B")
    model, rec = train(b, data[1], cfg(10), base, lines.append)
    assert model is not base and state_hash(base) == h
    assert base.envelope == before.envelope and base.aircraft == before.aircraft
    assert rec["n_aircraft"] == 2
    assert any("continued from the model" in s and "protection" in s for s in lines)
    assert any(s.startswith("columns:") and "epoch = passes" in s for s in lines)
    _, rec = train(*data, cfg(10, teacher=0.0), base, lines.append)
    assert "protection against forgetting off" in " ".join(lines)
    assert rec["n_aircraft"] == 1


def test_train_phase_rule(base, data):
    rule = {"level_fpm": 200.0, "cruise_alt_ft": 28000.0}
    model, rec = train(*data, cfg(5), log=lambda s: None, phase_rule=rule)
    assert model.envelope["phase_rule"] == rule
    assert rec["local_envelope"]["phase_rule"] == rule


def test_predict_keeps_device(base, data):
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    m = copy.deepcopy(base).to("cuda")
    x = data[0][data[0].split == "test"].iloc[:50]
    np.testing.assert_allclose(predict(m, x), predict(base, x), rtol=1e-5)
    assert next(m.parameters()).device.type == "cuda"


def test_init_without_envelope_warns(base, data):
    old = copy.deepcopy(base)
    old.envelope = None
    model, rec = continued(old, *data, steps=20)
    assert "no valid envelope" in rec["protection"]["warning"]
    assert rec["protection"]["points_in_shared_envelope"] == 0
    assert model.envelope == rec["local_envelope"]


def test_union_envelope(base, data):
    b, w = cruise_only(*data)
    e1 = base.envelope
    _, rec = continued(base, b, w, steps=10)
    e2 = rec["local_envelope"]
    u = union_envelope(e1, None, e2)
    for k in ("mach", "alt_ft", "mass_kg"):
        lo = min(e1["phases"]["level"][k][0], e2["phases"]["level"][k][0])
        assert u["phases"]["level"][k][0] == lo
    assert u["cruise"]["windows"] == e1["cruise"]["windows"] + e2["cruise"]["windows"]
    assert union_envelope(None) is None


def test_cli_train_protects_by_default(capsys, tmp_path, base, data):
    prep = write_prep(tmp_path / "prep", *data)
    save(base, tmp_path / "base.pt")
    args = ["train", str(prep), "--init", str(tmp_path / "base.pt"), "--device", "cpu"]
    code, out = run_cli(capsys, *args, "--out", str(tmp_path / "r1"), "--steps", "20")
    assert code == 0 and out["protection"]["teacher_weight"] == 10.0
    code, out = run_cli(
        capsys, *args, "--out", str(tmp_path / "r2"), "--steps", "20", "--no-protect"
    )
    assert code == 0 and out["protection"]["teacher_weight"] == 0.0


# ---------------------------------------------------------------- aggregation
def test_average(base, data):
    b, w = cruise_only(*data)
    c1, r1 = continued(base, *data, steps=30, seed=1)
    c2, r2 = continued(base, b, w, steps=30, seed=2)
    regions = [r1["local_envelope"], r2["local_envelope"]]
    m, rec = average(base, [c1, c2], regions)
    s1, s2, s = c1.state_dict(), c2.state_dict(), m.state_dict()
    k = "net.0.weight"
    assert torch.allclose(s[k], 0.5 * (s1[k] + s2[k]))
    # the base blocks and windows are counted once: base plus the local data
    for p in ("all", "level"):
        n = base.envelope["phases"][p]["blocks"]
        n += sum(r["phases"][p]["blocks"] for r in regions)
        assert m.envelope["phases"][p]["blocks"] == n
    n = base.envelope["cruise"]["windows"] + sum(
        r["cruise"]["windows"] for r in regions
    )
    assert m.envelope["cruise"]["windows"] == n
    m, rec = average(base, [c1, c2], regions, [3, 1])
    assert torch.allclose(m.state_dict()[k], 0.75 * s1[k] + 0.25 * s2[k])
    for bad in ([float("inf"), 1], [float("nan"), 1], [0, 1], [-1, 2]):
        with pytest.raises(ValueError, match="finite positive"):
            average(base, [c1, c2], regions, bad)
    with pytest.raises(ValueError, match="same weights"):
        average(base, [c1, c1], [regions[0]] * 2)
    with pytest.raises(ValueError, match="the base"):
        average(base, [copy.deepcopy(base)], regions[:1])
    broken = copy.deepcopy(c1)
    with torch.no_grad():
        broken.net[0].weight[0, 0] = float("inf")
    with pytest.raises(ValueError, match="not finite"):
        average(base, [broken, c2], regions)
    assert rec["base_weights_sha256"] == state_hash(base)
    assert [c["weights_sha256"] for c in rec["contributions"]] == [
        state_hash(c1),
        state_hash(c2),
    ]


def test_average_copies_buffers_of_the_base():
    shared = load(SHARED / "nf-a320-cfm56_5b-xs.pt")
    kids = [child(shared, b) for b in (0.001, 0.002, 0.003)]
    m, _ = average(shared, kids, [shared.envelope] * 3)
    assert differences(shared, m) == []
    for k in (*BUFFERS, "theta_exp"):
        assert torch.equal(getattr(m, k), getattr(shared, k))
    assert torch.allclose(m.net[-1].bias, shared.net[-1].bias + 0.002)
    # large weights are normalized without an overflow
    m, rec = average(shared, kids[:2], [shared.envelope] * 2, [1e308, 1e308])
    assert [c["weight"] for c in rec["contributions"]] == [0.5, 0.5]
    assert torch.equal(m.x_std, shared.x_std)
    with pytest.raises(ValueError, match="too big"):
        average(shared, kids[:2], [shared.envelope] * 2, [5e-324, 1e308])


def test_average_refuses_different_standardization(base, data):
    b, w = data
    other, _ = train(b[b.cruise].reset_index(drop=True), w, cfg(20), log=lambda s: None)
    assert any("standardization" in d for d in differences(base, other))
    with pytest.raises(ValueError, match="standardization"):
        average(base, [other], [other.envelope])
    xs = copy.deepcopy(base)
    xs.temperature = "free"
    assert differences(base, xs)


def test_cli_merge(capsys, tmp_path, base, data):
    save(base, tmp_path / "base.pt")
    bp = str(tmp_path / "base.pt")
    dirs = []
    for seed in (1, 2):
        m, rec = continued(base, *data, steps=30, seed=seed)
        d = tmp_path / f"c{seed}"
        write_run(m, rec, d)
        dirs.append(str(d))
    assert read_meta(tmp_path / "c1" / "model.pt")["init_weights_sha256"] == state_hash(
        base
    )
    out_pt = tmp_path / "merged.pt"
    code, out = run_cli(
        capsys,
        "merge",
        *dirs,
        "--base",
        bp,
        "--out",
        str(out_pt),
        "--weights",
        "blocks",
    )
    assert code == 0, out
    meta = read_meta(out_pt)["aggregation"]
    assert meta["base_weights_sha256"] == state_hash(base)
    assert len(meta["contributions"]) == 2
    assert load(out_pt).envelope is not None
    # the merge record next to the weights file
    mr = json.loads((tmp_path / "merged.merge.json").read_text())
    assert mr["method"] == "average" and mr["base_weights_sha256"] == state_hash(base)
    assert mr["envelope"] == load(out_pt).envelope
    assert {r["check"] for r in mr["quality_check"]["checks"]} >= {"fuel_level"}
    assert out["merge_record"] == str(tmp_path / "merged.merge.json")
    # the P7 settings travel from the base through the contributions to the merge
    dr = base.constraints["drag_rise"]
    assert dr["weight"] > 0
    assert json.loads((tmp_path / "c1" / "run.json").read_text())["drag_rise"] == dr
    assert load(out_pt).constraints["drag_rise"] == dr

    def merge(*a):
        return run_cli(
            capsys, "merge", *a, "--base", bp, "--out", str(tmp_path / "x.pt")
        )

    # no run record: bad input
    (tmp_path / "norec").mkdir()
    shutil.copyfile(tmp_path / "c1" / "model.pt", tmp_path / "norec" / "model.pt")
    code, out = merge(str(tmp_path / "norec"))
    assert code == 2 and "no run record" in out["error"]
    # the run record of another model
    (tmp_path / "swap").mkdir()
    shutil.copyfile(tmp_path / "c1" / "model.pt", tmp_path / "swap" / "model.pt")
    shutil.copyfile(tmp_path / "c2" / "run.json", tmp_path / "swap" / "run.json")
    code, out = merge(str(tmp_path / "swap"))
    assert code == 1 and "other weights" in out["differences"][0]
    # a contribution from another base (same standardization, other weights)
    other_base, _ = continued(base, *data, steps=10, seed=5)
    m, rec = continued(other_base, *data, steps=10, seed=6)
    write_run(m, rec, tmp_path / "c3")
    code, out = merge(dirs[0], str(tmp_path / "c3"))
    assert code == 1 and "another base" in out["differences"][0]
    # the same contribution twice, and the base as a contribution
    code, out = merge(dirs[0], dirs[0])
    assert code == 1 and "same weights" in out["differences"][0]
    rb = {"weights_sha256": state_hash(base), "init_weights_sha256": state_hash(base)}
    (tmp_path / "self").mkdir()
    shutil.copyfile(bp, tmp_path / "self" / "model.pt")
    (tmp_path / "self" / "run.json").write_text(
        json.dumps({**rb, "local_envelope": base.envelope})
    )
    code, out = merge(str(tmp_path / "self"))
    assert code == 1 and "the base" in out["differences"][0]
    # non-finite weights or parameters
    code, out = merge(*dirs, "--weights", "inf,1")
    assert code == 2 and "finite positive" in out["error"]
    nan = load(tmp_path / "c1" / "model.pt")
    with torch.no_grad():
        nan.net[0].weight[0, 0] = float("nan")
    r1 = json.loads((tmp_path / "c1" / "run.json").read_text())
    r1["weights_sha256"] = state_hash(nan)
    write_run(nan, r1, tmp_path / "nan")
    code, out = merge(str(tmp_path / "nan"))
    assert code == 1 and "NaN" in out["differences"][0]
    # a model from scratch (other standardization) is refused
    other, _ = train(
        data[0][data[0].cruise].reset_index(drop=True),
        data[1],
        cfg(20),
        log=lambda s: None,
    )
    save(other, tmp_path / "other.pt")
    (tmp_path / "other.run.json").write_text(json.dumps({"weights_sha256": "x"}))
    code, out = merge(str(tmp_path / "other.pt"))
    assert code == 1 and out["differences"]


def test_one_record_check(capsys, tmp_path, base, data):
    """contribute, check-model and merge use one check of the run record."""
    m, rec = continued(base, *data, steps=10, seed=1)
    other, _ = continued(base, *data, steps=10, seed=5)  # another compatible base
    assert record_problems(m, rec, base) == {"weights": [], "base": [], "region": []}
    save(base, tmp_path / "base.pt")
    bp = str(tmp_path / "base.pt")
    # the model file names another base than its run record
    d = tmp_path / "lineage"
    write_run(m, rec, d)
    meta = {"weights_sha256": state_hash(m), "init_weights_sha256": state_hash(other)}
    save(m, d / "model.pt", meta=meta)
    code, out = run_cli(capsys, "merge", str(d), "--base", bp, "--out", str(d / "x.pt"))
    assert code == 1 and "different base models" in out["differences"][0]
    # without the declared region, or with a region that does not give the envelope
    p = record_problems(
        m, {k: v for k, v in rec.items() if k != "local_envelope"}, base
    )
    assert "no declared region" in p["region"][0]
    moved = copy.deepcopy(rec)
    moved["local_envelope"]["phases"]["level"]["mach"][1] += 0.05
    assert "not the union" in record_problems(m, moved, base)["region"][0]
    moved = {**rec, "n_train_blocks": rec["n_train_blocks"] + 1}
    assert "n_train_blocks" in record_problems(m, moved, base)["region"][0]
    d = tmp_path / "noregion"
    write_run(m, {k: v for k, v in rec.items() if k != "local_envelope"}, d)
    code, out = run_cli(capsys, "merge", str(d), "--base", bp, "--out", str(d / "x.pt"))
    assert code == 1 and "declared region" in out["differences"][0]
    # check-model fails (not warns) without the declared region; a NaN weight keeps
    # the check short
    nan = copy.deepcopy(m)
    with torch.no_grad():
        nan.net[0].weight[0, 0] = float("nan")
    r = {k: v for k, v in rec.items() if k != "local_envelope"}
    write_run(nan, {**r, "weights_sha256": state_hash(nan)}, tmp_path / "nan")
    code, out = run_cli(
        capsys, "check-model", str(tmp_path / "nan" / "model.pt"), "--base", bp
    )
    st = {c["check"]: c["status"] for c in out["checks"]}
    assert code == 1 and st["declared_region"] == "fail" and st["base"] == "pass"


def test_combine_keeps_settings_of_the_base(base, data):
    c = copy.deepcopy(base)
    c.envelope["phase_rule"]["level_fpm"] = 450.0
    kid = child(c, 0.001)
    m, _ = combine(
        c, [kid], [c.envelope], cfg(1), n_points=128, n_val=64, log=lambda s: None
    )
    assert m.envelope["phase_rule"]["level_fpm"] == 450.0
    # envelopes that cannot be joined stop combine before the training
    lines = []
    with pytest.raises(ValueError, match="phase rule"):
        combine(base, [child(base, 0.001)], [c.envelope], cfg(1), 128, 64, lines.append)
    assert lines == []


def test_combine(base, data):
    b, w = cruise_only(*data)
    c1, r1 = continued(base, *data, steps=30, seed=1)
    c2, r2 = continued(base, b, w, steps=30, seed=2)
    regions = [r1["local_envelope"], r2["local_envelope"]]
    lines = []
    m, rec = combine(
        base, [c1, c2], regions, cfg(40), n_points=8192, n_val=2048, log=lines.append
    )
    assert rec["method"] == "combine" and rec["steps_run"] == 40
    assert not any("epoch" in s for s in lines)
    assert any(s.startswith("columns:") and "validation points" in s for s in lines)
    assert not differences(base, m)
    assert m.envelope["phases"]["climb"]["blocks"] > 0
    assert m.constraints["drag_rise"] == base.constraints["drag_rise"]


# ---------------------------------------------------------------- local offsets
def test_local_offsets(capsys, tmp_path, base, data):
    b, w = data
    b = b.copy()
    b["tail"] = np.where(np.arange(len(b)) % 2, "A", "B")
    b.loc[b["tail"] == "B", "ln_ff"] += 0.03
    off = fit_offsets(base, b)
    assert set(off["aircraft"]) == {"A", "B"}
    d = {
        p: off["aircraft"]["B"]["offsets"][p] - off["aircraft"]["A"]["offsets"][p]
        for p in ("climb", "level", "descent")
    }
    assert all(abs(v - 0.03) < 0.005 for v in d.values())
    x = b[b.split == "test"]
    o = offsets_for(off, "B")
    assert o["model_weights_sha256"] == state_hash(base)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ratio = predict(base, x, offsets=o) / predict(base, x)
    assert np.allclose(np.log(ratio), offset_values(x, o), atol=1e-6)
    other = copy.deepcopy(base)
    with torch.no_grad():
        other.net[0].bias[0] += 0.1
    with pytest.warns(UserWarning, match="another model"):
        predict(other, x, offsets=o)
    # without a tail column all blocks are one aircraft, label 'all'
    one = fit_offsets(base, data[0])
    assert set(one["aircraft"]) == {"all"} and offsets_for(one, "all")
    assert set(np.unique(phases(x))) <= {"climb", "level", "descent"}

    # the offsets stay out of model files, run records, reports and contributions
    prep = write_prep(tmp_path / "prep", b, w)
    run = tmp_path / "run"
    m, rec = continued(base, b, w, steps=20)
    write_run(m, rec, run)
    h = state_hash(load(run / "model.pt"))
    raw = (run / "model.pt").read_bytes()
    code, out = run_cli(
        capsys,
        "local-fit",
        str(prep),
        "--model",
        str(run / "model.pt"),
        "--out",
        str(tmp_path / "private" / "offsets.json"),
    )
    assert code == 0 and set(out["aircraft"]) == {"A", "B"}
    assert (run / "model.pt").read_bytes() == raw and state_hash(
        load(run / "model.pt")
    ) == h
    code, out = run_cli(
        capsys,
        "validate",
        str(prep),
        "--model",
        str(run / "model.pt"),
        "--out",
        str(run / "validation.json"),
    )
    assert code == 0
    code, out = run_cli(
        capsys,
        "contribute",
        "--run",
        str(run),
        "--report",
        str(run / "validation.json"),
        "--out",
        str(tmp_path / "share"),
    )
    assert code == 0
    private = json.loads((tmp_path / "private" / "offsets.json").read_text())
    vals = [
        repr(v) for e in private["aircraft"].values() for v in e["offsets"].values()
    ]
    for name in ("run.json", "validation.json", "MANIFEST.json"):
        text = (tmp_path / "share" / name).read_text()
        assert "offsets" not in text and not any(v in text for v in vals)
    assert "offsets" not in read_meta(run / "model.pt")


# ---------------------------------------------------------------- quality check
def test_quality_check(base, data, base_cache, fuel_run):
    good, rg = continued(base, *data, steps=300, seed=3)
    res = check_model(good, base, rg, cache=base_cache)
    assert res["ok"], [r for r in res["checks"] if r["status"] == "fail"]
    rows = {r["check"]: r for r in res["checks"]}
    assert rows["finite"]["value"] is True and rows["compatible"]["value"] is True
    assert rows["base"]["status"] == "pass" and rows["protection"]["value"] == 10.0
    # the same data as the base: few states outside the declared region warn
    n_out = rows["outside_states"]["value"]
    assert rows["outside_states"]["status"] == ("warn" if n_out < 1000 else "pass")
    assert "largest change" in rows["mach_max"]["note"]
    assert not any("F_C" in r["note"] or "of C" in r["note"] for r in res["checks"])
    # protection off fails; a record from scratch fails 'base'; other weights raise
    # (a model with a NaN weight keeps the test short: the check stops early)
    nan = copy.deepcopy(good)
    with torch.no_grad():
        nan.net[0].weight[0, 0] = float("nan")
    rn = {**rg, "weights_sha256": state_hash(nan)}
    res = check_model(nan, base, {**rn, "protection": {"teacher_weight": 0.0}})
    st = {r["check"]: r["status"] for r in res["checks"]}
    assert st["protection"] == "fail" and st["base"] == "pass"
    res = check_model(nan, base, {**rn, "init_weights_sha256": None})
    assert {r["check"]: r["status"] for r in res["checks"]}["base"] == "fail"
    with pytest.raises(ValueError, match="other weights"):
        check_model(good, base, {**rg, "weights_sha256": "0" * 64})
    for kind in ("pounds", "fuel"):
        if kind == "pounds":
            bad, rb = continued(base, *scaled(*data, mass=LB), steps=300, seed=3)
        else:
            bad, rb = fuel_run
        res = check_model(bad, base, rb, cache=base_cache)
        failed = {r["check"] for r in res["checks"] if r["status"] == "fail"}
        assert not res["ok"], (kind, res["checks"])
        assert ("region_type" if kind == "pounds" else "fuel_level") in failed
        res = check_model(bad, base, cache=base_cache)  # without the run record
        assert not res["ok"], (kind, res["checks"])
    broken = copy.deepcopy(good)
    with torch.no_grad():
        broken.net[0].weight[0, 0] = float("nan")
    res = check_model(broken, base)
    assert not res["ok"] and res["checks"][0]["status"] == "fail"


def test_quality_limits():
    assert _row("change_outside_max", float("inf"))["status"] == "fail"
    assert _row("mach_mean", float("nan"))["status"] == "fail"
    assert _row("mach_mean", None)["status"] == "warn"  # not measured


def test_cli_check_model(capsys, tmp_path, base, fuel_run):
    save(base, tmp_path / "base.pt")
    m, rec = fuel_run
    write_run(m, rec, tmp_path / "run")
    code, out = run_cli(
        capsys,
        "check-model",
        str(tmp_path / "run" / "model.pt"),
        "--base",
        str(tmp_path / "base.pt"),
        "--run",
        str(tmp_path / "run" / "run.json"),
    )
    assert code == 1 and not out["ok"]
    assert "fuel_level" in {c["check"] for c in out["checks"] if c["status"] == "fail"}
    assert {"check", "value", "warn", "fail", "status"} <= set(out["checks"][0])
    # the run record next to model.pt is found without --run (a NaN weight keeps
    # the check short); the text output shows yes or no
    mp, bp = str(tmp_path / "run" / "model.pt"), str(tmp_path / "base.pt")
    nan = load(mp)
    with torch.no_grad():
        nan.net[0].weight[0, 0] = float("nan")
    write_run(nan, {**rec, "weights_sha256": state_hash(nan)}, tmp_path / "nan")
    assert main(["check-model", str(tmp_path / "nan" / "model.pt"), "--base", bp]) == 1
    text = capsys.readouterr().err
    assert "finite" in text and " no " in text and " yes " in text
    # a missing run record and the run record of another model are bad input
    (tmp_path / "solo").mkdir()
    shutil.copyfile(mp, tmp_path / "solo" / "model.pt")
    code, out = run_cli(
        capsys, "check-model", str(tmp_path / "solo" / "model.pt"), "--base", bp
    )
    assert code == 2 and "no run record" in out["error"]
    rec2 = {**rec, "weights_sha256": state_hash(base)}
    (tmp_path / "solo" / "run.json").write_text(json.dumps(rec2))
    code, out = run_cli(
        capsys, "check-model", str(tmp_path / "solo" / "model.pt"), "--base", bp
    )
    assert code == 2 and "other weights" in out["error"]


# ---------------------------------------------------------------- comparison
def test_compare_with_model_reference(base, data):
    res = compare(base, model_reference(base))
    for q in res["phases"].values():
        assert abs(q["median"] - 1) < 1e-9
    assert (
        abs(res["cruise_mach_sensitivity"]["ratio_model_over_reference"]["median"] - 1)
        < 0.01
    )
    hidden = compare(base, model_reference(base), ratios_only=True)
    assert "reference" not in hidden["cruise_mach_sensitivity"]
    assert all("reference" not in r for r in hidden["best_range_mach"]["rows"])


def test_compare_flags_best_mach_outside_envelope():
    shared = load(SHARED / "nf-a320-cfm56_5b.pt")
    # a level-flight Mach range that ends below the best-range Mach
    shared.envelope["phases"]["level"]["mach"][1] = 0.70
    rows = compare(shared, model_reference(shared))["best_range_mach"]
    dT = rows["dT_isa_k"]
    for r in rows["rows"]:
        x = model_inputs([r["model"], r["reference"]], r["alt_ft"], r["mass_kg"], dT, 0)
        out = ~in_envelope(shared.envelope, x)
        assert [r["model_outside_envelope"], r["reference_outside_envelope"]] == list(
            out
        )
        assert isinstance(r["model_at_lower_bound"], bool)
    assert any(r["model"] > 0.70 and r["model_outside_envelope"] for r in rows["rows"])


def test_cli_compare(capsys, tmp_path, base):
    pytest.importorskip("openap")
    for name, m in (("a", base), ("b", child(base, 0.01))):
        (tmp_path / name).mkdir()
        save(m, tmp_path / name / "model.pt")
    code, out = run_cli(
        capsys,
        "compare",
        "--model",
        str(tmp_path / "a" / "model.pt"),
        "--against",
        str(tmp_path / "a" / "model.pt"),
        "--against",
        str(tmp_path / "b" / "model.pt"),
    )
    assert code == 0
    h = state_hash(base)[:8], state_hash(child(base, 0.01))[:8]
    assert set(out["references"]) == {
        "openap:A320",
        f"model:model-{h[0]}",
        f"model:model-{h[1]}",
    }
    assert str(tmp_path) not in json.dumps(out)
    assert 0.5 < out["references"]["openap:A320"]["phases"]["cruise"]["median"] < 2
    rows = out["references"]["openap:A320"]["best_range_mach"]["rows"]
    assert all(isinstance(r["model_at_upper_bound"], bool) for r in rows)


def test_openap_sees_model_pressure_and_temperature():
    pytest.importorskip("openap")
    from openap.aero import Aero

    from neuralfuel import atmos
    from neuralfuel.reference import _enroute, openap_conditions

    aero = Aero()

    class Spy:  # an OpenAP-style FuelFlow that returns the pressure ratio it sees
        def __init__(self):
            self.aero = aero

        def enroute(self, mass, tas, alt, vs, dT=0):
            h = np.asarray(alt) * atmos.FT
            self.mach = aero.tas2mach(np.asarray(tas) * atmos.KT, h, dT=dT)
            self.sat = aero.temperature(h, dT=dT)
            self.gamma = np.arctan2(
                np.asarray(vs) * atmos.FT / 60, np.asarray(tas) * atmos.KT
            )
            return aero.pressure(h, dT=dT) / atmos.P0 / 3600.0

    alt = np.array([35000.0, 37000.0, 20000.0, 30000.0])
    dT = np.array([10.0, 10.0, 5.0, -10.0])
    mach = np.full(4, 0.78)
    vs = np.array([0.0, 0.0, 1500.0, -1000.0])
    spy = Spy()
    delta = _enroute(spy, mach, alt, np.full(4, 60000.0), dT, vs)
    np.testing.assert_allclose(delta, atmos.isa_delta(alt), rtol=1e-9)
    sat = atmos.isa_temperature(alt) + dT
    np.testing.assert_allclose(spy.sat, sat, atol=1e-6)
    np.testing.assert_allclose(spy.mach, mach, rtol=1e-9)
    tas = mach * atmos.sound_speed(sat)
    np.testing.assert_allclose(np.sin(spy.gamma), vs * atmos.FT / 60 / tas, rtol=1e-6)
    # beyond the OpenAP range of the shift the pressure still matches
    h, shift, _, _ = openap_conditions(aero, 0.6, 10000.0, 18.0, 0.0)
    assert shift == pytest.approx(15.0)
    p = aero.pressure(h * atmos.FT, dT=shift) / atmos.P0
    assert p == pytest.approx(atmos.isa_delta(10000.0), rel=1e-9)


def test_bada_needs_local_folder(capsys, tmp_path, base, monkeypatch):
    pytest.importorskip("openap")
    monkeypatch.setenv("NEURALFUEL_BADA", str(tmp_path / "missing"))
    save(base, tmp_path / "base.pt")
    code, out = run_cli(
        capsys, "compare", "--model", str(tmp_path / "base.pt"), "--reference", "bada:X"
    )
    assert code == 2 and "BADA" in out["error"]
