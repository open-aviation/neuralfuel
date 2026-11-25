# Run: uv run --group reference examples/09_compare_openap.py
"""Compare the shared A320 model with the OpenAP fuel model on a small grid.

neuralfuel.reference.openap_reference gives the OpenAP fuel flow of the A320 at the
same pressure, static temperature, Mach and flight-path angle as the model state.
The script prints both fuel flows and their ratio at a few cruise, climb and descent
states, and marks the states outside the valid envelope of the model. OpenAP is a
reference, not the truth.

For the full comparison on the grid of the review of contributions, run
`uv run --group reference neuralfuel compare --model models/a320/nf-a320-cfm56_5b.pt`.

Run from the repository root:

    uv run --group reference examples/09_compare_openap.py

Run time: a few seconds.
"""

import numpy as np

from neuralfuel.inference import predict
from neuralfuel.model import load
from neuralfuel.reference import model_inputs, openap_reference

MODEL = "models/a320/nf-a320-cfm56_5b.pt"
DT_ISA = 10.0  # K


def main():
    model = load(MODEL)
    ref = openap_reference("A320")
    rows = []  # (phase, Mach, altitude ft, mass kg, vertical speed ft/min)
    for alt in (31000.0, 35000.0, 37000.0):
        for mass in (56000.0, 62000.0, 68000.0):
            rows.append(("cruise", 0.78, alt, mass, 0.0))
    for alt, mach in ((15000.0, 0.55), (25000.0, 0.70), (33000.0, 0.76)):
        rows.append(("climb", mach, alt, 64000.0, 1500.0))
        rows.append(("descent", mach, alt, 60000.0, -2000.0))
    phase, mach, alt, mass, vs = (np.array(c) for c in zip(*rows, strict=True))
    dT = np.full(len(rows), DT_ISA)

    x = model_inputs(mach, alt, mass, dT, vs)  # still air, no acceleration
    ff, outside = predict(model, x, outside=True)
    ff_ref = ref(mach, alt, mass, dT, vs)

    print(f"{MODEL} against OpenAP A320, ISA {DT_ISA:+.0f} K, still air")
    print(
        f"  {'phase':8s} {'Mach':>5s} {'alt ft':>7s} {'mass kg':>8s} {'vs ft/min':>9s} "
        f"{'model':>6s} {'OpenAP':>6s} {'ratio':>6s}  envelope"
    )
    for i, r in enumerate(rows):
        where = "outside" if outside[i] else "inside"
        print(
            f"  {r[0]:8s} {r[1]:5.2f} {r[2]:7.0f} {r[3]:8.0f} {r[4]:9.0f} "
            f"{ff[i]:6.0f} {ff_ref[i]:6.0f} {ff[i] / ff_ref[i]:6.3f}  {where}"
        )
    print("  fuel flow in kg/h; ratio = model / OpenAP")
    for p in ("cruise", "climb", "descent"):
        sel = (phase == p) & ~outside
        if sel.any():
            r = ff[sel] / ff_ref[sel]
            print(
                f"  {p:8s} median ratio inside the envelope {np.median(r):.3f} "
                f"({sel.sum()} states)"
            )


if __name__ == "__main__":
    main()
