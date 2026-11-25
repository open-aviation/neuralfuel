# Run: uv run examples/06_train_synthetic.py
"""The single-user cycle through the Python API, on synthetic data.

Steps: generate a synthetic raw file, read it with the 'synthetic' adapter, run the
plausibility checks, prepare blocks and steady windows, train a model from scratch,
validate it on the test split, export it to ONNX and print the main numbers of the
validation report. The same steps exist as commands (see docs/training.md).

The synthetic data is not real flight data. It comes from simple physics (see
neuralfuel.synthetic), so the numbers only show that the cycle works. --steps is a
short budget to test the workflow; the default budget of the library is 19,500
update steps. Never use a short budget for a contribution.

Run from the repository root:

    uv run examples/06_train_synthetic.py
    uv run examples/06_train_synthetic.py --steps 200 --out work/example06

Run time: about 40 seconds on a CPU with the default --steps 1000.
"""

import argparse
import json
import tempfile
from pathlib import Path

from neuralfuel import synthetic
from neuralfuel.adapters import synthetic as synthetic_adapter
from neuralfuel.checks import check
from neuralfuel.inference import export_onnx
from neuralfuel.prep import prepare
from neuralfuel.train import TrainConfig, train, write_run
from neuralfuel.validate import validate


def run(out, flights, steps, threads):
    out.mkdir(parents=True, exist_ok=True)

    # 1. raw data and adapter
    raw = out / "syn.csv"
    synthetic.generate(n_flights=flights, seed=0).to_csv(raw, index=False)
    samples = synthetic_adapter.load(raw)
    print(f"synthetic raw file: {len(samples)} samples at 1 Hz, {flights} flights")

    # 2. checks after the adapter
    c = check(samples)
    failed = [t["test"] for t in c["tests"] if t["status"] == "fail"]
    print(f"check: {len(c['tests'])} tests, failed: {failed or 'none'}")

    # 3. blocks, steady windows and the split by flight
    blocks, windows, summary = prepare(samples)
    for split, n in summary["blocks"].items():
        print(
            f"prep {split:5s}: {n['blocks']:6d} blocks, {n['flights']:3d} flights, "
            f"{summary['windows'][split]:4d} steady windows"
        )

    # 4. training from scratch
    cfg = TrainConfig(seed=0, max_steps=steps, device="cpu", threads=threads)
    model, rec = train(blocks, windows, cfg, log=lambda s: None)
    write_run(model, rec, out / "run")
    print(
        f"train: {rec['steps_run']} steps, best validation loss {rec['val_loss']:.5f} "
        f"at step {rec['best_step']}, {rec['n_params']} weights, "
        f"{rec['run_time_s']:.0f} s"
    )

    # 5. validation report on the test split
    rep = validate(model, blocks, windows, splits=["test"])
    (out / "run" / "validation.json").write_text(json.dumps(rep, indent=1))
    print("\nvalidation report (test split):")
    for phase, a in rep["accuracy"].items():
        if a["n"]:
            print(
                f"  {phase:8s} blocks {a['n']:5d}  MAPE {a['MAPE_pct']:5.2f} %  "
                f"rRMSE {a['rRMSE_pct']:5.2f} %  bias {a['bias_pct']:+5.2f} %"
            )
    tf = rep["trip_fuel"]
    print(
        f"  trip fuel: {tf['flights']} flights, median absolute error "
        f"{tf['median_abs_error_pct']:.2f} %"
    )
    ms = rep["mach_sensitivity"]
    if ms["mean_s_supported"] is not None:
        print(
            f"  Mach sensitivity d ln F / dM: mean {ms['mean_s_supported']:.3f} over "
            f"{ms['supported_cells']} supported cells"
        )
    viol = rep["constraints"]["all"]
    print(
        "  constraint violations on the grid (%): "
        + ", ".join(
            f"{k.removesuffix('_violation_pct')} {v:.2f}" for k, v in viol.items()
        )
    )
    print(
        f"  test blocks outside the valid envelope: "
        f"{rep['envelope']['evaluated_outside_pct']['all']:.1f} %"
    )

    # 6. ONNX export, checked against PyTorch
    info = export_onnx(model, out / "run" / "model.onnx")
    print(
        f"\nexport: {Path(info['path']).relative_to(out)}, largest relative difference "
        "ONNX / PyTorch "
        f"{info['max_rel_diff_ff']:.1e}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=1000, help="update steps")
    ap.add_argument("--flights", type=int, default=30, help="synthetic flights")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads")
    ap.add_argument("--out", help="keep the files in this folder (default: temporary)")
    args = ap.parse_args()
    if args.out:
        run(Path(args.out), args.flights, args.steps, args.threads)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            run(Path(tmp), args.flights, args.steps, args.threads)


if __name__ == "__main__":
    main()
