# Run: uv run examples/08_local_offsets.py
"""Private local offsets: a per-aircraft, per-phase correction of a model.

The script applies the shared A320 model to synthetic data of two aircraft, A and B.
Aircraft B burns 3 % more fuel than A (the synthetic fuel flow of B is multiplied by
1.03 in this script). The script fits three offsets in ln fuel flow per aircraft
(climb, level, descent) on the training split, applies them in a prediction and
shows the bias on the test split before and after.

The synthetic data is not real flight data, and its physics differs from the A320,
so the bias of the shared model here says nothing about real aircraft. The offsets
are private: they go to their own file and never into a model, a run record, a
contribution or a validation report.

Run from the repository root:

    uv run examples/08_local_offsets.py

Run time: about 7 seconds.
"""

import tempfile
from pathlib import Path

import numpy as np

from neuralfuel import synthetic
from neuralfuel.adapters import synthetic as synthetic_adapter
from neuralfuel.inference import predict
from neuralfuel.local import (
    fit_offsets,
    offsets_for,
    phases,
    read_offsets,
    write_offsets,
)
from neuralfuel.model import load
from neuralfuel.prep import FEATURES, prepare

MODEL = "models/a320/nf-a320-cfm56_5b.pt"
FUEL_B = 1.03  # aircraft B burns 3 % more fuel (a choice of this example)


def main():
    model = load(MODEL)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "syn.csv"
        synthetic.generate(n_flights=30, seed=3).to_csv(raw, index=False)
        s = synthetic_adapter.load(raw)
        # neutral aircraft labels: alternate flights belong to A and B
        codes = s.flight_id.factorize(sort=True)[0]
        s["tail"] = np.where(codes % 2 == 0, "A", "B")
        b = s["tail"] == "B"
        for c in ("ff_kgh", "ff1_kgh", "ff2_kgh"):
            s.loc[b, c] *= FUEL_B
        blocks, _, _ = prepare(s)

        offsets = fit_offsets(model, blocks, split="train")
        path = write_offsets(offsets, Path(tmp) / "private" / "offsets.json")
        offsets = read_offsets(path)

    print(f"offsets in ln fuel flow, fitted on the train split ({MODEL}):")
    for label, e in offsets["aircraft"].items():
        vals = ", ".join(
            f"{p} {e['offsets'][p]:+.4f} ({e['blocks'][p]} blocks)"
            for p in e["offsets"]
        )
        print(f"  {label}: {vals}")

    te = blocks[blocks.split == "test"]
    print("\nbias of fuel flow on the test split (%), before -> after the offsets:")
    print(f"  {'aircraft':8s} {'phase':8s} {'blocks':>6s} {'before':>8s} {'after':>8s}")
    for label in sorted(te["tail"].unique()):
        t = te[te["tail"] == label]
        x = t[FEATURES].to_numpy()
        obs = t.ff_kgh.to_numpy()
        before = predict(model, x) / obs - 1
        after = predict(model, x, offsets=offsets_for(offsets, label)) / obs - 1
        ph = phases(x)
        for p in ("all", "climb", "level", "descent"):
            sel = np.ones(len(x), bool) if p == "all" else ph == p
            if sel.any():
                print(
                    f"  {label:8s} {p:8s} {sel.sum():6d} "
                    f"{100 * before[sel].mean():+8.2f} "
                    f"{100 * after[sel].mean():+8.2f}"
                )
    print(
        "\nthe offsets make a step in fuel flow at the phase borders: use them for "
        "fuel estimation, not inside a trajectory optimizer"
    )


if __name__ == "__main__":
    main()
