# Run: uv run examples/07_federated_round.py
"""One federated round through the Python API, on synthetic data.

A maintainer has a base model. Two users continue the training of the base on their
own data, with the protection against forgetting:

  user A  cruise blocks only (the user keeps only level flight above 25,000 ft)
  user B  complete flights (climb, cruise and descent)

The maintainer runs the quality check on each contribution against the base, then
averages the weights of both contributions into the merged model. The table shows
the error of the base, of each contribution and of the merged model on the test
flights of each user.

The synthetic data is not real flight data, and the step counts are short, so the
numbers only show how the round works. The default budget of the library is 19,500
update steps. Never use a short budget for a contribution.

Run from the repository root:

    uv run examples/07_federated_round.py
    uv run examples/07_federated_round.py --base-steps 200 --steps 100

Run time: about 1 minute on a CPU with the default step counts.
"""

import argparse
import copy
import tempfile
from pathlib import Path

import numpy as np

from neuralfuel import synthetic
from neuralfuel.adapters import synthetic as synthetic_adapter
from neuralfuel.aggregate import average
from neuralfuel.inference import predict
from neuralfuel.prep import FEATURES, prepare
from neuralfuel.quality import check_model
from neuralfuel.train import TrainConfig, train


def dataset(tmp, flights, seed):
    """Blocks and windows of a synthetic dataset."""
    raw = Path(tmp) / f"syn_{seed}.csv"
    synthetic.generate(n_flights=flights, seed=seed).to_csv(raw, index=False)
    blocks, windows, _ = prepare(synthetic_adapter.load(raw))
    return blocks, windows


def errors(model, blocks):
    """MAPE and bias (%) of fuel flow on the test blocks."""
    te = blocks[blocks.split == "test"]
    if len(te) == 0:
        return None
    e = predict(model, te[FEATURES]) / te.ff_kgh.to_numpy() - 1
    return 100 * np.abs(e).mean(), 100 * e.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-steps", type=int, default=600, help="steps of the base")
    ap.add_argument("--steps", type=int, default=400, help="steps of each user")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads")
    args = ap.parse_args()

    def cfg(steps, seed):
        return TrainConfig(
            seed=seed, max_steps=steps, device="cpu", threads=args.threads
        )

    with tempfile.TemporaryDirectory() as tmp:
        base_data = dataset(tmp, 12, seed=0)
        a_blocks, a_windows = dataset(tmp, 20, seed=1)
        user_a = (a_blocks[a_blocks.cruise].reset_index(drop=True), a_windows)
        user_b = dataset(tmp, 20, seed=2)

    # maintainer: the base model (from scratch)
    base, _ = train(*base_data, cfg(args.base_steps, 0), log=lambda s: None)

    # users: continue from the base with the protection against forgetting (default)
    contrib, records = {}, {}
    for name, data, seed in (("A", user_a, 1), ("B", user_b, 2)):
        m, rec = train(
            *data, cfg(args.steps, seed), init=copy.deepcopy(base), log=lambda s: None
        )
        contrib[name], records[name] = m, rec
        p = rec["protection"]
        print(
            f"user {name}: {rec['n_train_blocks']} training blocks, teacher weight "
            f"{p['teacher_weight']:g}, {100 * p['share_inside_local_envelope']:.1f} % "
            "of the protection points inside the declared region"
        )

    # maintainer: quality check of each contribution against the base
    print("\nquality check against the base (check_model):")
    for name, m in contrib.items():
        q = check_model(m, base, records[name])
        other = [
            f"{r['check']} {r['status']} ({r['value']:.3g})"
            for r in q["checks"]
            if r["status"] != "pass"
        ]
        print(
            f"  user {name}: {'passed' if q['ok'] else 'FAILED'}, "
            f"{len(q['checks'])} checks; not passed: {', '.join(other) or 'none'}"
        )

    # maintainer: weight average of the two contributions
    regions = [records[n]["local_envelope"] for n in contrib]
    merged, rec = average(base, list(contrib.values()), regions)
    print(
        f"\nmerged: {rec['method']}, weights "
        + ", ".join(f"{c['weight']:.2f}" for c in rec["contributions"])
    )

    # error on the test flights of each user
    b_blocks = user_b[0]
    b_phases = b_blocks[b_blocks.phase != "level"]
    columns = {
        "user A, cruise": user_a[0],
        "user B, all": b_blocks,
        "user B, climb+descent": b_phases,
    }
    models = {
        "base": base,
        "contribution A": contrib["A"],
        "contribution B": contrib["B"],
        "merged": merged,
    }
    print("\nMAPE / bias (%) of fuel flow on the test blocks of each user:")
    print(f"  {'model':16s}" + "".join(f"{c:>24s}" for c in columns))
    for name, m in models.items():
        cells = []
        for b in columns.values():
            e = errors(m, b)
            cells.append("-" if e is None else f"{e[0]:.2f} / {e[1]:+.2f}")
        print(f"  {name:16s}" + "".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    main()
