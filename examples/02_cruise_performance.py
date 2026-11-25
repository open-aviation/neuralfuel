# Run: uv run examples/02_cruise_performance.py
# Figures: uv run --group examples examples/02_cruise_performance.py --plot
"""Specific air range of the shared A320 model in steady cruise.

Specific air range (SAR) is the distance flown through the air per fuel mass:
true airspeed / fuel flow, here in NM per 1000 kg. The script computes SAR against
Mach at several masses and altitudes, the Mach of maximum SAR (best-range Mach), and
the altitude of maximum SAR against mass. Only states inside the valid envelope of
the model count for the best altitude.

With the fixed temperature rule of the shared model, fuel flow and true airspeed
both scale with the square root of theta at a fixed Mach, so SAR and the best-range
Mach do not depend on the temperature. The script uses ISA +10 K, inside the
temperature range of the envelope.

Run from the repository root:

    uv run examples/02_cruise_performance.py
    uv run --group examples examples/02_cruise_performance.py --plot --out-dir DIR

--plot writes sar_mach.png and best_altitude.png (needs matplotlib, dependency group
'examples'). Run time: a few seconds.
"""

import argparse
from pathlib import Path

import numpy as np

from neuralfuel import atmos
from neuralfuel.inference import predict
from neuralfuel.model import load

MODEL = "models/a320/nf-a320-cfm56_5b.pt"
DT_ISA = 10.0  # K
MACH = np.round(np.arange(0.60, 0.8201, 0.002), 3)  # up to the MMO of the A320
ALTS = (31000.0, 35000.0, 37000.0)
MASSES = (56000.0, 62000.0, 68000.0)


def cruise(model, mach, alt_ft, mass_kg):
    """Fuel flow (kg/h), SAR (NM per 1000 kg) and the outside flag of level flight."""
    mach, alt_ft, mass_kg = (
        np.ravel(v) for v in np.broadcast_arrays(mach, alt_ft, mass_kg)
    )
    sat = atmos.isa_temperature(alt_ft) + DT_ISA
    zero = np.zeros(len(mach))
    x = np.column_stack(
        [
            mach,
            np.log(atmos.isa_delta(alt_ft)),
            np.log(mass_kg),
            zero,  # sin_gamma: level flight
            zero,  # acc_in: constant speed
            np.log(sat / atmos.T0),
        ]
    )
    ff, outside = predict(model, x, outside=True)
    tas_kt = mach * atmos.sound_speed(sat) / atmos.KT
    return ff, 1000.0 * tas_kt / ff, outside


def sar_curves(model):
    """SAR against Mach for each altitude and mass."""
    out = {}
    for a in ALTS:
        for w in MASSES:
            out[a, w] = cruise(model, MACH, a, w)
    return out


def best_altitude(model, masses, alts, inside_only=True):
    """For each mass: (mass, altitude, Mach, SAR, W/delta in t) of the largest SAR,
    with inside_only only over the states inside the valid envelope."""
    rows = []
    for w in masses:
        M, A = np.meshgrid(MACH, alts, indexing="ij")
        _, sar, outside = cruise(model, M, A, w)
        if inside_only:
            sar = np.where(outside, -np.inf, sar)
        sar = sar.reshape(M.shape)
        if not np.isfinite(sar).any():
            rows.append((w, None, None, None, None))
            continue
        i, j = np.unravel_index(np.argmax(sar), sar.shape)
        wd = w / float(atmos.isa_delta(alts[j])) / 1000.0
        rows.append((w, alts[j], MACH[i], sar[i, j], wd))
    return rows


def plot(curves, rows, envelope, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#1b6ca8", "#d1495b", "#2e8b57")
    fig, axes = plt.subplots(1, len(ALTS), figsize=(10, 3.6), sharey=True)
    for ax, a in zip(axes, ALTS, strict=True):
        for w, c in zip(MASSES, colors, strict=True):
            _, sar, outside = curves[a, w]
            ins = np.where(outside, np.nan, sar)
            ax.plot(MACH, sar, color=c, lw=1, ls=":")
            ax.plot(MACH, ins, color=c, lw=2, label=f"{w / 1000:.0f} t")
            i = int(np.argmax(sar))
            ax.plot(MACH[i], sar[i], "o", color=c, ms=5, mfc="white")
        ax.set_xlabel("Mach")
        ax.text(0.03, 0.04, f"{a:,.0f} ft", transform=ax.transAxes)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("specific air range (NM per 1000 kg)")
    axes[0].legend(title="mass", loc="upper left", fontsize=8, title_fontsize=8)
    fig.text(
        0.5,
        0.005,
        f"solid: inside the valid envelope; dotted: outside; circle: largest SAR "
        f"on the curve. ISA {DT_ISA:+.0f} K, level flight.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p1 = out_dir / "sar_mach.png"
    fig.savefig(p1, dpi=100, facecolor="white")
    plt.close(fig)

    ok = [r for r in rows if r[1] is not None]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(
        [r[0] / 1000 for r in ok],
        [r[1] for r in ok],
        "o-",
        color=colors[0],
        label="inside the valid envelope",
    )
    lv = envelope["phases"]["level"]
    w = np.linspace(min(r[0] for r in rows), max(r[0] for r in rows), 50)
    ax.plot(
        w / 1000,
        atmos.alt_from_delta(w / (1000.0 * lv["wd_t"][1])),
        "--",
        color=colors[1],
        label=f"W/delta = {lv['wd_t'][1]:.1f} t (envelope limit)",
    )
    ax.axhline(
        lv["alt_ft"][1],
        ls=":",
        color=colors[2],
        label=f"{lv['alt_ft'][1]:,.0f} ft (envelope limit)",
    )
    ax.set_xlabel("mass (1000 kg)")
    ax.set_ylabel("pressure altitude of largest SAR (ft)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(top=lv["alt_ft"][1] + 400)
    fig.tight_layout()
    p2 = out_dir / "best_altitude.png"
    fig.savefig(p2, dpi=100, facecolor="white")
    plt.close(fig)
    return p1, p2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plot", action="store_true", help="write two PNG figures")
    ap.add_argument("--out-dir", default="examples/output", help="folder of the PNGs")
    args = ap.parse_args()

    model = load(MODEL)
    curves = sar_curves(model)
    print(f"best-range Mach, level flight, ISA {DT_ISA:+.0f} K ({MODEL})")
    print(f"  {'altitude ft':>11s} {'mass kg':>8s} {'Mach':>6s} {'SAR':>6s}  envelope")
    for (a, w), (_, sar, outside) in curves.items():
        i = int(np.argmax(sar))
        where = "outside" if outside[i] else "inside"
        print(f"  {a:11.0f} {w:8.0f} {MACH[i]:6.3f} {sar[i]:6.1f}  {where}")
    print("  SAR in NM per 1000 kg")

    masses = np.arange(52000.0, 68001.0, 2000.0)
    top = model.aircraft.alt_ft[1]  # highest altitude of the aircraft settings
    alts = np.arange(25000.0, top + 1.0, 500.0)
    rows = best_altitude(model, masses, alts)
    free = best_altitude(model, masses, alts, inside_only=False)  # extrapolation
    print(
        "\nbest altitude against mass: largest SAR over Mach and altitude, inside "
        "the envelope and without the envelope"
    )
    print(
        f"  {'mass kg':>8s} | {'altitude ft':>11s} {'Mach':>6s} {'SAR':>6s} "
        f"{'W/delta t':>9s} | {'altitude ft':>11s} {'Mach':>6s} {'SAR':>6s}"
    )
    for r, f in zip(rows, free, strict=True):
        left = (
            "no state inside the envelope".center(37)
            if r[1] is None
            else f"{r[1]:11.0f} {r[2]:6.3f} {r[3]:6.1f} {r[4]:9.1f}"
        )
        print(f"  {r[0]:8.0f} | {left} | {f[1]:11.0f} {f[2]:6.3f} {f[3]:6.1f}")
    print(
        f"  without the envelope the search grid ends at {alts[-1]:.0f} ft and Mach "
        f"{MACH[-1]}"
    )
    lv = model.envelope["phases"]["level"]
    print(
        f"  envelope of level flight: Mach up to {lv['mach'][1]:.3f}, altitude up to "
        f"{lv['alt_ft'][1]:.0f} ft, W/delta up to {lv['wd_t'][1]:.1f} t; inside the "
        "envelope the best state can lie on these limits (altitude step 500 ft)"
    )

    if args.plot:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("\n--plot needs matplotlib: uv run --group examples ...")
            return
        for p in plot(curves, rows, model.envelope, args.out_dir):
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
