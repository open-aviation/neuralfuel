"""Command-line interface.

Exit codes: 0 success, 1 a check or privacy test failed or merge refused a
contribution, 2 bad input (missing file, unknown adapter, schema error, missing or
mismatched run record, an option or setting out of range), 3 unexpected error.
Every command accepts --json and then prints exactly one JSON object on stdout, also
for an error in the command-line arguments.
"""

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from . import __version__

EXIT_OK, EXIT_FAIL, EXIT_INPUT, EXIT_ERROR = 0, 1, 2, 3


class InputError(Exception):
    def __init__(self, msg, column=None):
        super().__init__(msg)
        self.column = column


class ArgumentError(Exception):
    """An error in the command-line arguments (argparse)."""


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentError(f"{self.prog}: {message}")


def read_json(path, name=None):
    """The JSON file at path; InputError if it is not valid JSON."""
    try:
        return json.loads(Path(path).read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise InputError(f"{name or path}: not a JSON file ({e})") from e


# ---------------------------------------------------------------- helpers
def load_samples(args, raw=False):
    from .adapters import get_adapter, read_table
    from .config import resolve

    data, adapter = resolve(args.data, args.adapter)
    if not data:
        raise InputError(
            "no data path: give DATA, set NEURALFUEL_DATA or write local.toml"
        )
    if not Path(data).exists():
        raise InputError(f"data file not found: {data}")
    if raw:
        return read_table(data), "raw"
    try:
        fn = get_adapter(adapter)
    except KeyError as e:
        raise InputError(e.args[0]) from e
    except (ImportError, AttributeError) as e:
        raise InputError(str(e)) from e
    try:
        return fn(data), adapter
    except KeyError as e:
        raise InputError(
            f"adapter '{adapter}': raw column {e} not found", str(e)
        ) from e
    except Exception as e:
        raise InputError(f"adapter '{adapter}' failed: {e}") from e


def load_prepared(d):
    """(blocks, windows, PrepConfig) of a folder written by 'prep'; the preparation
    settings come from its prep.json."""
    from .prep import PrepConfig

    d = Path(d)
    if not (d / "blocks.parquet").exists():
        raise InputError(f"{d}: no blocks.parquet; run 'neuralfuel prep' first")
    if not (d / "prep.json").exists():
        raise InputError(
            f"{d}: no prep.json (it records the preparation settings); run "
            "'neuralfuel prep' again"
        )
    cfg = read_json(d / "prep.json").get("config")
    try:
        cfg = PrepConfig(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()}
        )
    except (TypeError, AttributeError) as e:
        raise InputError(f"{d}/prep.json: invalid preparation settings ({e})") from e
    blocks = pd.read_parquet(d / "blocks.parquet")
    return blocks, pd.read_parquet(d / "windows.parquet"), cfg


def load_weights(path):
    from .model import load

    if not Path(path).exists():
        raise InputError(f"weights file not found: {path}")
    try:
        return load(path)
    except ValueError as e:
        raise InputError(str(e)) from e


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _typed(cur, v, text, key):
    """The value v (JSON-decoded text) in the type of the current value cur."""

    def bad(kind):
        return InputError(f"--set {key}: needs {kind}, got '{text}'")

    def number(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    if dataclasses.is_dataclass(cur):
        raise InputError(f"--set {key}: a group of settings; give one of its fields")
    if isinstance(cur, bool):
        if not isinstance(v, bool):
            raise bad("true or false")
        return v
    if isinstance(cur, int) or cur is None:
        if v is None and cur is None:
            return None
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        raise bad("an integer")
    if isinstance(cur, float):
        if not number(v):
            raise bad("a number")
        return float(v)
    if isinstance(cur, str):
        return text
    if isinstance(cur, tuple):
        if not (isinstance(v, list) and len(v) == len(cur) and all(map(number, v))):
            raise bad(f"a list of {len(cur)} numbers")
        return tuple(float(x) for x in v)
    raise bad(type(cur).__name__)


def set_field(obj, key, value):
    """Set a (nested, dot-separated) dataclass field from a text value; InputError for
    an unknown field or a value of the wrong type."""
    *parents, last = key.split(".")
    for name in [*parents, last]:
        if not dataclasses.is_dataclass(obj) or name not in {
            f.name for f in dataclasses.fields(obj)
        }:
            raise InputError(f"unknown setting '{key}'")
        if name != last:
            obj = getattr(obj, name)
    try:
        v = json.loads(value)
    except json.JSONDecodeError:
        v = value
    setattr(obj, last, _typed(getattr(obj, last), v, value, key))


def set_fields(obj, items):
    """Apply the KEY=VALUE items of --set to the dataclass obj."""
    for kv in items or []:
        if "=" not in kv:
            raise InputError(f"--set needs key=value, got '{kv}'")
        set_field(obj, *kv.split("=", 1))


def positive(name, v):
    if v is not None and v < 1:
        raise InputError(f"{name} must be 1 or more, got {v}")


def fmt_number(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


# ---------------------------------------------------------------- commands
def cmd_profile(args):
    from .adapters import get_adapter, measurements
    from .checks import profile
    from .config import resolve

    df, used = load_samples(args, raw=args.raw)
    known = ()
    if args.raw:  # the raw measurement columns that the adapter lists
        name = resolve(args.data, args.adapter)[1]
        if name != "canonical":
            try:
                known = measurements(get_adapter(name))
            except (KeyError, ImportError, AttributeError) as e:
                raise InputError(f"adapter '{name}': {e}") from e
    out = {"adapter": used, **profile(df, known)}
    text = [f"rows {out['rows']} (adapter: {used})"]
    for c in out["columns"]:
        rng = (
            f"[{fmt_number(c['min'])} .. {fmt_number(c['max'])}] p50 "
            f"{fmt_number(c['p50'])}"
            if "min" in c
            else c.get("ranges", f"true {c.get('true_fraction', 0):.3f}")
        )
        text.append(
            f"  {c['name']:24s} {c['dtype']:16s} missing "
            f"{c['missing_fraction']:.4f}  {rng}"
        )
    text.append(f"sampling: {out['sampling']}")
    return EXIT_OK, out, "\n".join(text)


def cmd_check(args):
    from .checks import check

    df, used = load_samples(args)
    out = {"adapter": used, **check(df)}
    text = [f"adapter: {used}"]
    for p in out["schema"]:
        text.append(f"  schema {p['level']}: {p['column']}: {p['message']}")
    for t in out["tests"]:
        v = "-" if t["value"] is None else f"{t['value']:.4g}"
        line = (
            f"  {t['status']:4s} {t['test']} [{t['column']}] value {v}, expected "
            f"{t['expected']}"
        )
        if t["message"]:
            line += f"\n       -> {t['message']}"
        text.append(line)
    fl = out.get("flags")
    if fl:
        text.append(
            f"configuration flags, share of {fl['definition']} ({fl['samples']}):"
        )
        rows = [("all", fl["all"])] + list(fl["bands"].items())
        for name, r in rows:
            vals = "  ".join(
                f"{k} {'-' if r[k] is None else f'{r[k]:.4f}'}"
                for k in ("clean", "speedbrake", "gear_down", "anti_ice")
            )
            text.append(f"  {name:18s} {vals}")
    text.append("all tests passed" if out["ok"] else "CHECK FAILED")
    return (EXIT_OK if out["ok"] else EXIT_FAIL), out, "\n".join(text)


def cmd_prep(args):
    from .prep import PrepConfig, check_config, prepare
    from .privacy import dumps
    from .schema import SchemaError

    cfg = PrepConfig()
    set_fields(cfg, args.set)
    try:
        check_config(cfg)
    except ValueError as e:
        raise InputError(str(e)) from e
    df, used = load_samples(args)
    try:
        blocks, windows, summary = prepare(df, cfg)
    except SchemaError as e:
        raise InputError(str(e), e.problems[0].column) from e
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks.to_parquet(out_dir / "blocks.parquet")
    windows.to_parquet(out_dir / "windows.parquet")
    summary = {"adapter": used, **summary}
    (out_dir / "prep.json").write_text(dumps(summary))
    text = [
        f"wrote {out_dir}/blocks.parquet ({len(blocks)} blocks) and windows.parquet "
        f"({len(windows)} windows)"
    ]
    for sp, c in summary["blocks"].items():
        text.append(
            f"  {sp}: {c['blocks']} blocks, {c['flights']} flights, "
            f"{summary['windows'][sp]} windows"
        )
    return EXIT_OK, {"out": str(out_dir), **summary}, "\n".join(text)


def train_config(args, seed=None, size=None):
    """TrainConfig from the common training arguments."""
    from .aircraft import SIZES, aircraft
    from .train import TrainConfig, check_config

    for name in ("steps", "patience_steps", "epochs", "patience", "threads"):
        positive("--" + name.replace("_", "-"), getattr(args, name))
    cfg = TrainConfig(
        seed=args.seed if seed is None else seed,
        device=args.device,
        threads=args.threads,
    )
    cfg.hidden, cfg.layers = SIZES[size or args.size]
    cfg.aircraft = aircraft(args.aircraft or "a320", args.engine or "")
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.patience_steps is not None:
        cfg.patience_steps = args.patience_steps
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.patience is not None:
        if args.epochs is None:
            raise InputError("--patience (passes) needs --epochs; use --patience-steps")
        cfg.patience = args.patience
    set_fields(cfg, args.set)
    try:
        cfg.aircraft.check()
        check_config(cfg)
    except ValueError as e:
        raise InputError(str(e)) from e
    return cfg


def train_log(args):
    stream = sys.stderr if args.json else sys.stdout

    def log(s):
        print(s, file=stream, flush=True)

    return log


def cmd_train(args):
    from .train import train, write_run

    blocks, windows, prep_cfg = load_prepared(args.prep)
    notes = []
    if args.init and any(s.startswith("aircraft.") for s in args.set or []):
        raise InputError(
            "--set aircraft.* is not possible with --init: the aircraft settings "
            "come from the given model"
        )
    if args.init and (args.aircraft or args.engine):
        notes.append(
            "--aircraft and --engine are ignored with --init: the aircraft settings "
            "come from the given model"
        )
        args.aircraft = args.engine = None
    cfg = train_config(args)
    if args.no_protect:
        cfg.loss.teacher = 0.0
    log = train_log(args)
    init = load_weights(args.init) if args.init else None
    try:
        model, rec = train(
            blocks, windows, cfg, init=init, log=log, prep_config=prep_cfg
        )
    except ValueError as e:  # for example a non-finite loss or an empty envelope
        raise InputError(str(e)) from e
    mp, rp = write_run(model, rec, args.out)
    if rec.get("drag_rise_note"):
        notes.append(f"drag-rise term P7 off: {rec['drag_rise_note']}")
    out = {k: v for k, v in rec.items() if k != "history"}
    out.update(weights=str(mp), run_record=str(rp), notes=notes)
    text = (
        f"wrote {mp} and {rp}: val loss {rec['val_loss']:.6f} at step "
        f"{rec['best_step']} of {rec['steps_run']}, {rec['run_time_s']:.0f} s, size "
        f"{rec['size']}, temperature rule {rec['temperature']}"
    )
    for n in notes:
        text += f"\nnote: {n}"
    if rec["protection"].get("warning"):
        text += f"\nwarning: {rec['protection']['warning']}"
    return EXIT_OK, out, text


def _release_run(blocks, windows, cfg, prep_cfg):
    """One training of 'release' in a worker process (no log lines)."""
    from .train import train

    return train(blocks, windows, cfg, log=lambda *a: None, prep_config=prep_cfg)


def cmd_release(args):
    """Train both sizes (best seed by validation loss), validate and export."""
    from .aircraft import SIZES, model_name
    from .inference import export_onnx
    from .model import save
    from .privacy import dumps
    from .train import train
    from .validate import validate

    blocks, windows, prep_cfg = load_prepared(args.prep)
    try:
        seeds = [int(x) for x in args.seeds.split(",")]
    except ValueError as e:
        raise InputError(
            f"--seeds: comma-separated integers, got '{args.seeds}'"
        ) from e
    sizes = args.sizes.split(",")
    for size in sizes:
        if size not in SIZES:
            raise InputError(f"unknown size '{size}'; known: {', '.join(SIZES)}")
    positive("--jobs", args.jobs)
    for kv in args.set or []:
        if kv.split("=", 1)[0] in ("hidden", "layers"):
            raise InputError(
                "release takes the network size from --sizes; --set hidden and --set "
                "layers are not allowed"
            )
    tasks = [(size, seed) for size in sizes for seed in seeds]
    cfgs = [train_config(args, seed=seed, size=size) for size, seed in tasks]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = train_log(args)
    try:
        if args.jobs == 1:
            done = []
            for (size, seed), cfg in zip(tasks, cfgs, strict=True):
                log(f"size {size}, seed {seed}")
                done.append(train(blocks, windows, cfg, log=log, prep_config=prep_cfg))
        else:  # independent trainings: one process for each, args.jobs at a time
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            log(f"{len(tasks)} trainings in {min(args.jobs, len(tasks))} processes")
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(args.jobs, mp_context=ctx) as ex:
                futs = [
                    ex.submit(_release_run, blocks, windows, c, prep_cfg) for c in cfgs
                ]
                done = [f.result() for f in futs]
    except ValueError as e:
        raise InputError(str(e)) from e
    res, text, notes = {}, [], []
    for size in sizes:
        runs = [r for (sz, _), r in zip(tasks, done, strict=True) if sz == size]
        model, rec = min(runs, key=lambda r: r[1]["val_loss"])
        rec["seed_val_losses"] = {
            str(r[1]["config"]["seed"]): r[1]["val_loss"] for r in runs
        }
        if model.size != size:  # the file name and the size label follow the network
            raise InputError(
                f"the trained network ({model.layers} x {model.hidden}) is not the "
                f"size '{size}'"
            )
        ac = model.aircraft
        name = model_name(ac.type, ac.engine, model.size)
        files = {
            "weights": out_dir / f"{name}.pt",
            "run_record": out_dir / f"{name}.run.json",
            "report": out_dir / f"{name}.validation.json",
            "onnx": out_dir / f"{name}.onnx",
        }
        save(model, files["weights"], meta={"weights_sha256": rec["weights_sha256"]})
        files["run_record"].write_text(dumps(rec))
        rep = validate(model, blocks, windows, splits=["test"], prep_config=prep_cfg)
        files["report"].write_text(dumps(rep))
        info = export_onnx(model, files["onnx"])
        res[size] = {
            "name": name,
            "seed": rec["config"]["seed"],
            "val_loss": rec["val_loss"],
            "seed_val_losses": rec["seed_val_losses"],
            "n_params": rec["n_params"],
            "test_MAPE_all_pct": rep["accuracy"]["all"]["MAPE_pct"],
            "test_MAPE_cruise_pct": rep["accuracy"]["cruise"]["MAPE_pct"],
            "onnx_max_rel_diff_ff": info["max_rel_diff_ff"],
            "files": {k: str(v) for k, v in files.items()},
        }
        if rec.get("drag_rise_note"):
            notes.append(f"{name}: drag-rise term P7 off: {rec['drag_rise_note']}")
        text.append(
            f"{name}: seed {res[size]['seed']}, val loss {rec['val_loss']:.6f}, test "
            f"MAPE {res[size]['test_MAPE_all_pct']:.2f} %, ONNX max rel. diff "
            f"{info['max_rel_diff_ff']:.1e}"
        )
    text += [f"note: {n}" for n in notes]
    ok = all(r["onnx_max_rel_diff_ff"] < 1e-4 for r in res.values())
    return (
        (EXIT_OK if ok else EXIT_FAIL),
        {"models": res, "notes": notes},
        "\n".join(text),
    )


def cmd_validate(args):
    import torch

    from .aircraft import aircraft
    from .privacy import dumps
    from .validate import validate

    positive("--threads", args.threads)
    torch.set_num_threads(args.threads)
    blocks, windows, prep_cfg = load_prepared(args.prep)
    model = load_weights(args.model)
    ac = aircraft(args.aircraft, args.engine or "") if args.aircraft else None
    if model.aircraft is None and ac is None:
        raise InputError(
            "the model file has no aircraft settings; give --aircraft (and --engine)"
        )
    samples = load_samples(args)[0] if args.surveillance else None
    try:
        rep = validate(
            model,
            blocks,
            windows,
            splits=args.splits.split(","),
            aircraft=ac,
            samples=samples,
            prep_config=prep_cfg,
        )
    except ValueError as e:
        raise InputError(str(e)) from e
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(rep))
    a, ms = rep["accuracy"], rep["mach_sensitivity"]
    text = [f"wrote {out}"]
    for k, v in a.items():
        if v["n"]:
            text.append(
                f"  MAPE {k:8s} {v['MAPE_pct']:.2f} %  bias {v['bias_pct']:+.2f} %  "
                f"(n {v['n']})"
            )
    tf = rep["trip_fuel"]
    text.append(
        f"  trip fuel median |error| {tf['median_abs_error_pct']:.2f} % over "
        f"{tf['flights']} flights"
    )
    if ms["mean_s_supported"] is not None:
        text.append(
            f"  mean d lnF/dM over {ms['supported_cells']} supported cells "
            f"{ms['mean_s_supported']:.3f}"
        )
    tt = rep["temperature_test"]
    if tt["rows"]:
        text.append(
            f"  Mach of maximum specific air range at {tt['alt_ft']:.0f} ft, "
            f"{tt['mass_kg']:.0f} kg, ISA {tt['rows'][0]['dT_isa_k']:+g} to "
            f"{tt['rows'][-1]['dT_isa_k']:+g} K: shift {tt['shift_mach_max_sr']:+.3f}"
        )
    sv = rep.get("surveillance")
    if sv:
        text.append(
            f"  surveillance inputs ({sv['blocks']} blocks): MAPE "
            f"{sv['surveillance_inputs']['all']['MAPE_pct']:.2f} % against "
            f"{sv['recorded_inputs']['all']['MAPE_pct']:.2f} % with recorded inputs"
        )
    return EXIT_OK, {"report": str(out), **rep}, "\n".join(text)


def cmd_export(args):
    from .inference import export_onnx

    model = load_weights(args.model)
    try:
        info = export_onnx(model, args.out, casadi=args.casadi)
    except ValueError as e:  # a model that overflows in the --casadi form
        raise InputError(str(e)) from e
    ok = info["max_rel_diff_ff"] < 1e-4
    text = (
        f"wrote {info['path']} (ops {', '.join(info['ops'])}); ONNX against PyTorch on "
        f"{info['n_check']} inputs: max relative fuel-flow difference "
        f"{info['max_rel_diff_ff']:.2e}"
    )
    return (EXIT_OK if ok else EXIT_FAIL), {"ok": ok, **info}, text


def cmd_contribute(args):
    from .aggregate import run_record_problems
    from .inference import export_onnx
    from .losses import CONSTRAINT_SETTINGS, DRAG_RISE
    from .model import load, read_meta, save, state_hash
    from .privacy import NUM, allowed, dumps, find_identifiers
    from .support import ENVELOPE
    from .train import RUN_RECORD
    from .validate import REPORT

    run = Path(args.run)
    src = {
        "model.pt": run / "model.pt",
        "run.json": run / "run.json",
        "validation.json": Path(args.report),
    }
    for name, p in src.items():
        if not p.exists():
            raise InputError(f"{name}: file not found: {p}")
    out = Path(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise InputError(
            f"{out}: the destination exists and is not empty; give a new or empty "
            "folder"
        )
    rec = read_json(src["run.json"], "run.json")
    rep = read_json(src["validation.json"], "validation.json")
    hits = find_identifiers(rec, path="run.json") + find_identifiers(
        rep, path="validation.json"
    )
    # the shared records hold only the keys of their formats
    rec, dropped = allowed(rec, RUN_RECORD, "run.json")
    rep, d = allowed(rep, REPORT, "validation.json")
    dropped += d
    if hits:
        return (
            EXIT_FAIL,
            {"ok": False, "identifying_fields": hits},
            ("privacy test failed; fields: " + ", ".join(hits)),
        )
    if rep.get("evaluated_splits") != ["test"]:
        raise InputError(
            "validation.json must be a report of the test split only (validate "
            f"--splits test); it evaluated {rep.get('evaluated_splits')}"
        )
    model = load_weights(src["model.pt"])
    h = state_hash(model)
    if (rep.get("model") or {}).get("weights_sha256") != h:
        raise InputError(
            "validation.json was computed with other weights than model.pt"
        )
    # the shared model file holds only the typed fields of the model
    if model.envelope is not None:
        model.envelope, d = allowed(model.envelope, ENVELOPE, "model.pt.envelope")
        dropped += d
    if model.constraints is not None:
        spec = dict.fromkeys(CONSTRAINT_SETTINGS, NUM) | {"drag_rise": DRAG_RISE}
        model.constraints, d = allowed(model.constraints, spec, "model.pt.constraints")
        dropped += d
    meta = read_meta(src["model.pt"])
    dropped += [f"model.pt.meta.{k}" for k in meta if k not in HASH_KEYS]
    problems = run_record_problems(
        model,
        rec,
        meta={k: v for k, v in meta.items() if k in HASH_KEYS},
        need_base=False,
    )
    if problems:
        raise InputError("run.json: " + "; ".join(problems))
    try:
        summary = {
            "test_MAPE_all_pct": rep["accuracy"]["all"]["MAPE_pct"],
            "test_MAPE_cruise_pct": rep["accuracy"]["cruise"]["MAPE_pct"],
            "trip_fuel_median_abs_error_pct": rep["trip_fuel"]["median_abs_error_pct"],
            "mean_dlnF_dM_supported": rep["mach_sensitivity"]["mean_s_supported"],
            "train_blocks": rec["n_train_blocks"],
        }
    except (KeyError, TypeError) as e:
        raise InputError(f"incomplete run.json or validation.json: {e}") from e
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        init = rec.get("init_weights_sha256")
        meta = {"weights_sha256": h, "init_weights_sha256": init}
        save(model, stage / "model.pt", meta=meta)
        if state_hash(load(stage / "model.pt")) != h:
            raise RuntimeError("the written model.pt has other weights")
        info = export_onnx(model, stage / "model.onnx")  # from the verified model
        (stage / "run.json").write_text(dumps(rec))
        (stage / "validation.json").write_text(dumps(rep))
        names = ("model.pt", "model.onnx", "run.json", "validation.json")
        manifest = {
            "neuralfuel_version": __version__,
            "weights_sha256": h,
            "files": {name: sha256(stage / name) for name in names},
            "evaluated_splits": rep["evaluated_splits"],
            "summary": summary,
            "onnx_max_rel_diff_ff": info["max_rel_diff_ff"],
            "dropped_fields": dropped,
        }
        (stage / "MANIFEST.json").write_text(dumps(manifest))
        if info["max_rel_diff_ff"] >= 1e-4:
            return (
                EXIT_FAIL,
                {"ok": False, **manifest},
                f"the ONNX file differs from PyTorch ({info['max_rel_diff_ff']:.1e})",
            )
        if out.exists():
            out.rmdir()  # empty, checked above
        stage.rename(out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    text = (
        f"wrote {out} ({', '.join(sorted(names))}, MANIFEST.json); attach this folder "
        "to a pull request"
    )
    if dropped:
        text += "\ndropped fields (not in the record formats): " + ", ".join(dropped)
    return EXIT_OK, {"ok": True, "out": str(out), **manifest}, text


HASH_KEYS = ("weights_sha256", "init_weights_sha256")  # the meta of a shared model


def contribution_files(path):
    """(weights file, run record dict or None) of a contribution: a folder with
    model.pt and run.json, or a weights file (its run record: run.json next to a
    file named model.pt, else <stem>.run.json)."""
    p = Path(path)
    if p.is_dir():
        w, r = p / "model.pt", p / "run.json"
    else:
        w = p
        r = (
            p.with_name("run.json")
            if p.name == "model.pt"
            else p.with_suffix(".run.json")
        )
    rec = read_json(r) if r.exists() else None
    return w, rec


def cmd_merge(args):
    from .aggregate import (
        average,
        combine,
        differences,
        duplicates,
        finite_state,
        run_record_problems,
        save_merged,
    )
    from .model import read_meta, state_hash
    from .quality import check_model
    from .support import union_envelope

    positive("--threads", args.threads)
    base = load_weights(args.base)
    h_base = state_hash(base)
    models, recs = [], []

    def refused(c, reasons):
        return (
            EXIT_FAIL,
            {"ok": False, "contribution": str(c), "differences": reasons},
            f"refused: {c}: " + "; ".join(reasons),
        )

    for c in args.contributions:
        w, rec = contribution_files(c)
        m = load_weights(w)
        if rec is None:
            raise InputError(
                f"{c}: no run record (run.json); merge needs the run record of every "
                "contribution"
            )
        d = differences(base, m)
        if d:
            return refused(c, d)
        if state_hash(m) == h_base:
            return refused(c, ["the contribution has the same weights as the base"])
        d = run_record_problems(m, rec, base, read_meta(w))
        if d:
            return refused(c, d)
        if not finite_state(m):
            return refused(c, ["parameters with NaN or infinite values"])
        models.append(m)
        recs.append(rec)
    d = duplicates(base, models)
    if d:
        return refused(", ".join(args.contributions), d)
    regions = [r["local_envelope"] for r in recs]  # checked by run_record_problems
    if args.method == "average":
        weights = None
        if args.weights == "blocks":
            weights = [r["n_train_blocks"] for r in recs]
        elif args.weights not in (None, "equal"):
            try:
                weights = [float(v) for v in args.weights.split(",")]
            except ValueError as e:
                raise InputError(f"--weights: {e}") from e
        try:
            model, rec = average(base, models, regions, weights)
        except ValueError as e:
            raise InputError(str(e)) from e
    else:
        cfg = train_config(args)
        try:
            model, rec = combine(base, models, regions, cfg, log=train_log(args))
        except ValueError as e:  # for example envelopes that cannot be joined
            raise InputError(str(e)) from e
    try:
        quality = check_model(model, base, declared=union_envelope(*regions))
    except ValueError as e:  # for example a base without an envelope
        quality = {"ok": None, "error": str(e), "checks": [], "warnings": 0}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _, mr = save_merged(model, rec, out, quality)
    text = (
        f"wrote {out} and {mr}: {rec['method']} of {len(models)} contributions, "
        "weights " + ", ".join(f"{c['weight']:.3f}" for c in rec["contributions"])
    )
    text += "\nquality check of the merged model against the base: " + (
        f"not run: {quality['error']}"
        if quality["ok"] is None
        else "passed"
        + (f" with {quality['warnings']} warnings" if quality.get("warnings") else "")
        if quality["ok"]
        else "FAILED (see quality_check in the merge record)"
    )
    res = {"ok": True, "out": str(out), "merge_record": str(mr), **rec}
    res["quality_check"] = quality
    return EXIT_OK, res, text


def cmd_local_fit(args):
    from .local import fit_offsets, write_offsets

    blocks, _, prep_cfg = load_prepared(args.prep)
    model = load_weights(args.model)
    try:
        off = fit_offsets(model, blocks, split=args.split, level_fpm=prep_cfg.level_fpm)
    except ValueError as e:
        raise InputError(str(e)) from e
    write_offsets(off, args.out)
    text = [f"wrote {args.out} (private: do not share)"]
    for a, e in off["aircraft"].items():
        v = ", ".join(f"{p} {e['offsets'][p]:+.4f}" for p in e["offsets"])
        text.append(f"  {a}: {v}")
    return EXIT_OK, {"out": str(args.out), **off}, "\n".join(text)


def cmd_check_model(args):
    from .model import read_meta
    from .quality import check_model

    contrib = load_weights(args.model)
    base = load_weights(args.base)
    if args.run:
        if not Path(args.run).exists():
            raise InputError(f"run record not found: {args.run}")
        rec = read_json(args.run)
    else:
        rec = contribution_files(args.model)[1]
        if rec is None:
            raise InputError(
                "no run record: give --run run.json (the run record of the "
                "contribution)"
            )
    try:
        res = check_model(contrib, base, rec, meta=read_meta(args.model))
    except ValueError as e:
        raise InputError(str(e)) from e

    def num(v):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if float(v).is_integer():
            return str(int(v))
        return f"{v:.4g}"

    text = [f"{'check':22s} {'value':>10s} {'warn':>8s} {'fail':>8s}  status"]
    for r in res["checks"]:
        line = (
            f"{r['check']:22s} {num(r['value']):>10s} {num(r['warn']):>8s} "
            f"{num(r['fail']):>8s}  {r['status']}"
        )
        if r["note"]:
            line += f"  ({r['note']})"
        text.append(line)
    text.append(
        "quality check passed"
        + (f" with {res['warnings']} warnings" if res.get("warnings") else "")
        if res["ok"]
        else "QUALITY CHECK FAILED"
    )
    return (EXIT_OK if res["ok"] else EXIT_FAIL), res, "\n".join(text)


def reference_for(spec, model):
    """(name, callable, ratios only) of a reference given as openap[:TYPE],
    bada:NAME or the path of a neuralfuel model."""
    from . import reference as R

    kind, _, arg = spec.partition(":")
    try:
        if kind == "openap":
            t = arg or model.aircraft.type
            return f"openap:{t.upper()}", R.openap_reference(t), False
        if kind == "bada":
            if not arg:
                raise InputError("bada needs a BADA model name, for example bada:NAME")
            return f"bada:{arg}", R.bada_reference(arg), True
    except (ImportError, FileNotFoundError) as e:
        raise InputError(str(e)) from e
    from .model import state_hash

    m = load_weights(spec)
    name = f"model:{Path(spec).stem}-{state_hash(m)[:8]}"  # file stem and weights hash
    return name, R.model_reference(m), False


def cmd_compare(args):
    from .reference import compare

    model = load_weights(args.model)
    refs = list(args.reference or ["openap"]) + list(args.against or [])
    out, text = {"references": {}}, []
    for spec in refs:
        name, fn, ratios_only = reference_for(spec, model)
        if name in out["references"]:
            raise InputError(f"the reference {name} is given twice")
        try:
            res = compare(model, fn, ratios_only=ratios_only)
        except ValueError as e:
            raise InputError(str(e)) from e
        out["references"][name] = res
        text.append(f"model / {name}" + (" (ratios only)" if ratios_only else ""))
        text.append(f"  {'phase':8s} {'n':>6s} {'median':>8s} {'p05':>8s} {'p95':>8s}")
        for ph, q in res["phases"].items():
            text.append(
                f"  {ph:8s} {q['n']:6d} {q['median']:8.3f} {q['p05']:8.3f} "
                f"{q['p95']:8.3f}"
            )
        ms = res.get("cruise_mach_sensitivity")
        if ms:
            line = (
                f"  cruise Mach sensitivity d lnF/dM ({ms['states']} states): median "
                "of the pointwise ratios model / reference "
                f"{ms['ratio_model_over_reference']['median']:.3f}"
            )
            if "model" in ms:
                line += (
                    f"; median model {ms['model']['median']:.3f}, median reference "
                    f"{ms['reference']['median']:.3f}"
                )
            text.append(line)
        marks = set()
        for r in res.get("best_range_mach", {}).get("rows", []):
            line = (
                f"  best-range Mach at {r['alt_ft']:.0f} ft, {r['mass_kg']:.0f} kg: "
                f"ratio {r['ratio']:.3f}"
            )
            mark = {}
            for k in ("model", "reference"):
                mark[k] = (
                    ("*" if r[f"{k}_at_upper_bound"] else "")
                    + ("_" if r[f"{k}_at_lower_bound"] else "")
                    + ("!" if r[f"{k}_outside_envelope"] else "")
                )
                marks |= set(mark[k])
            if "model" in r:
                line += (
                    f" (model {r['model']:.3f}{mark['model']}, reference "
                    f"{r['reference']:.3f}{mark['reference']})"
                )
            elif mark["model"] or mark["reference"]:
                line += (
                    f" (model {mark['model'] or '-'}, reference "
                    f"{mark['reference'] or '-'})"
                )
            text.append(line)
        legend = {
            "*": "at the upper bound of the search (the MMO)",
            "_": "at the lower bound of the search",
            "!": "outside the valid envelope of the model",
        }
        text += [f"  {k} {v}" for k, v in legend.items() if k in marks]
    return EXIT_OK, out, "\n".join(text)


# ---------------------------------------------------------------- parser
def parser():
    p = Parser(prog="neuralfuel", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        s = sub.add_parser(name, help=help_, description=help_)
        s.add_argument("--json", action="store_true", help="print one JSON object")
        s.set_defaults(fn=fn)
        return s

    def data_args(s):
        s.add_argument(
            "data", nargs="?", help="raw file (default: NEURALFUEL_DATA or local.toml)"
        )
        s.add_argument(
            "--adapter",
            help="adapter name or module:function (default: local.toml or canonical)",
        )

    s = add(
        "profile",
        cmd_profile,
        "summarize columns, ranges and sampling (no rows, no identifiers)",
    )
    data_args(s)
    s.add_argument(
        "--raw", action="store_true", help="profile the raw file, without the adapter"
    )
    s = add(
        "check", cmd_check, "schema and physical plausibility tests after the adapter"
    )
    data_args(s)
    s = add("prep", cmd_prep, "build 10 s blocks, steady windows and the split")
    data_args(s)
    s.add_argument("--out", required=True, help="output folder")
    s.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="preparation settings, for example exclude_anti_ice=true or block_s=10",
    )

    def train_args(s):
        s.add_argument("prep", help="folder written by 'prep'")
        s.add_argument(
            "--aircraft",
            help="aircraft type (default a320); ignored with --init",
        )
        s.add_argument("--engine", help="engine family, for example cfm56_5b")
        s.add_argument(
            "--steps", type=int, help="budget in update steps (default 19500)"
        )
        s.add_argument(
            "--patience-steps",
            type=int,
            help=(
                "early stopping: steps without a better validation loss (default 3900)"
            ),
        )
        s.add_argument(
            "--epochs",
            type=int,
            help="budget in passes over the data instead of steps (older runs)",
        )
        s.add_argument(
            "--patience", type=int, help="with --epochs: patience in passes (30)"
        )
        s.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
        s.add_argument("--threads", type=int, default=4, help="CPU threads")
        s.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="other settings, for example loss.window=0.1, lr=0.001, "
            "temperature=free or aircraft.mmo=0.86",
        )

    s = add("train", cmd_train, "train a model on prepared data")
    train_args(s)
    s.add_argument("--out", required=True, help="run folder for model.pt and run.json")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--size",
        default="standard",
        choices=["standard", "xs"],
        help="standard: 3 hidden layers of 64; xs: 3 hidden layers of 32",
    )
    s.add_argument("--init", help="weights to continue from (model.pt)")
    s.add_argument(
        "--no-protect",
        action="store_true",
        help="with --init: no protection against forgetting (loss.teacher=0)",
    )
    s = add(
        "release",
        cmd_release,
        "train both sizes (best seed by validation loss), validate and export ONNX",
    )
    train_args(s)
    s.add_argument("--out", required=True, help="folder for the nf-*.pt/.onnx files")
    s.add_argument("--seeds", default="0", help="comma-separated seeds (default 0)")
    s.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="trainings that run at the same time, one process each (default 1); "
        "the number of CPU cores in use is about jobs x threads",
    )
    s.add_argument(
        "--sizes", default="standard,xs", help="comma-separated sizes (standard,xs)"
    )
    s = add(
        "validate", cmd_validate, "validation report (JSON, aggregate numbers only)"
    )
    s.add_argument("prep", help="folder written by 'prep'")
    s.add_argument("--model", required=True, help="model.pt")
    s.add_argument("--out", required=True, help="report file (validation.json)")
    s.add_argument(
        "--aircraft",
        help="aircraft type, only for model files without aircraft settings",
    )
    s.add_argument("--engine", help="engine family, with --aircraft")
    s.add_argument(
        "--splits", default="test", help="comma-separated splits to evaluate"
    )
    s.add_argument("--threads", type=int, default=4, help="CPU threads")
    s.add_argument(
        "--surveillance",
        action="store_true",
        help="also run the surveillance-input test; it reads the 1 Hz samples "
        "through the adapter (--data, --adapter, NEURALFUEL_DATA or local.toml)",
    )
    s.add_argument("--data", help="raw file for --surveillance (default: local.toml)")
    s.add_argument("--adapter", help="adapter for --surveillance")
    s = add("export", cmd_export, "export the network to ONNX and compare with PyTorch")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True, help="model.onnx")
    s.add_argument(
        "--casadi", action="store_true", help="write Softplus as log(1 + exp(x))"
    )
    s = add(
        "contribute",
        cmd_contribute,
        "assemble weights, ONNX, run record and report in one folder",
    )
    s.add_argument("--run", required=True, help="run folder with model.pt and run.json")
    s.add_argument("--report", required=True, help="validation.json of the test split")
    s.add_argument("--out", required=True, help="new or empty output folder")
    s = add(
        "merge",
        cmd_merge,
        "aggregate contributions that continued from the same base model",
    )
    s.add_argument(
        "contributions",
        nargs="+",
        help="contribution folders (model.pt, run.json) or weights files",
    )
    s.add_argument("--base", required=True, help="the shared model they continued from")
    s.add_argument("--out", required=True, help="merged weights file")
    s.add_argument(
        "--method",
        default="average",
        choices=["average", "combine"],
        help="average: weight averaging; combine: output combination (a student "
        "trained on the outputs of the contributions in their declared regions)",
    )
    s.add_argument(
        "--weights",
        help="average: 'equal' (default), 'blocks' (training blocks from run.json) or "
        "comma-separated numbers",
    )
    s.add_argument("--aircraft", default="a320", help=argparse.SUPPRESS)
    s.add_argument("--engine", help=argparse.SUPPRESS)
    s.add_argument("--seed", type=int, default=0, help="combine: student seed")
    s.add_argument("--size", default="standard", help=argparse.SUPPRESS)
    s.add_argument("--steps", type=int, help="combine: update steps (default 19500)")
    s.add_argument("--patience-steps", type=int, help=argparse.SUPPRESS)
    s.add_argument("--epochs", type=int, help=argparse.SUPPRESS)
    s.add_argument("--patience", type=int, help=argparse.SUPPRESS)
    s.add_argument("--device", default="auto", help="combine: auto, cpu, cuda")
    s.add_argument("--threads", type=int, default=4, help="CPU threads")
    s.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help=argparse.SUPPRESS
    )
    s = add(
        "local-fit",
        cmd_local_fit,
        "fit the private offsets (climb, level, descent) per aircraft; never shared",
    )
    s.add_argument("prep", help="folder written by 'prep'")
    s.add_argument("--model", required=True, help="model.pt to correct")
    s.add_argument("--out", required=True, help="offsets file (JSON, keep private)")
    s.add_argument("--split", default="train", help="split to fit on (default train)")
    s = add(
        "check-model",
        cmd_check_model,
        "quality check of a contribution against its base model (no flight data)",
    )
    s.add_argument("model", help="the contribution (model.pt)")
    s.add_argument("--base", required=True, help="the shared model it continued from")
    s.add_argument(
        "--run",
        help="run record of the contribution (default: run.json next to model.pt)",
    )
    s = add(
        "compare",
        cmd_compare,
        "compare a model with reference fuel models on a grid inside its envelope",
    )
    s.add_argument("--model", required=True, help="model.pt")
    s.add_argument(
        "--reference",
        action="append",
        help="openap[:TYPE] (default) or bada:NAME (local licensed tables, ratios "
        "only); repeat for several",
    )
    s.add_argument(
        "--against",
        action="append",
        help="another neuralfuel model, for example the previous shared model",
    )
    return p


def main(argv=None):
    from .privacy import dumps

    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    try:
        args = p.parse_args(argv)
    except ArgumentError as e:
        if "--json" in argv:
            cmd = next((a for a in argv if not a.startswith("-")), None)
            out = {"command": cmd, "exit_code": EXIT_INPUT, "ok": False}
            print(dumps({**out, "error": str(e), "column": None}, indent=None))
        else:
            p.print_usage(sys.stderr)
            print(f"error: {e}", file=sys.stderr)
        return EXIT_INPUT
    try:
        code, out, text = args.fn(args)
    except InputError as e:
        code, out, text = (
            EXIT_INPUT,
            {"ok": False, "error": str(e), "column": e.column},
            f"error: {e}",
        )
    except Exception as e:  # noqa: BLE001
        code, out, text = (
            EXIT_ERROR,
            {"ok": False, "error": f"{type(e).__name__}: {e}"},
            f"error: {type(e).__name__}: {e}",
        )
    if args.json:
        out = {"command": args.cmd, "exit_code": code, "ok": code == EXIT_OK, **out}
        print(dumps(out, indent=None))
    else:
        print(text, file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
