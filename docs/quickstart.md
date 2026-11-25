# Quick start

## Install

neuralfuel needs Python 3.12 or later and [uv](https://docs.astral.sh/uv/). Run every command from the repository root.

The shared models (`models/`), the examples (`examples/`) and this documentation (`docs/`) are in the repository, not in the installed package. The package is not on PyPI yet. An install from the repository without a clone (`uv add git+https://github.com/open-aviation/neuralfuel`) gives the library and the command `neuralfuel` only. The first example of the README and the examples need a clone of the repository:

```sh
git clone https://github.com/open-aviation/neuralfuel
cd neuralfuel
```

```sh
uv sync
uv run neuralfuel --help
```

Optional dependency groups:

| group | content | install |
|---|---|---|
| `casadi` | CasADi, for use in an optimizer | `uv sync --group casadi` |
| `reference` | OpenAP, for the comparison with reference models | `uv sync --group reference` |
| `examples` | matplotlib, for the figures of the examples | `uv sync --group examples` |
| `dev` | pytest | installed by `uv sync` |

`uv run --group NAME ...` installs a group for one command.

## Quick start with synthetic data

The package contains a generator for an artificial dataset. **The synthetic data is not real flight data.** It comes from a parabolic drag polar with a compressibility term, a thrust-specific fuel consumption law, an idle floor and measurement noise (`src/neuralfuel/synthetic.py`). It uses a raw format that differs from the schema on purpose, so the cycle also runs the `synthetic` adapter (see [data.md](data.md)).

```sh
uv run python -m neuralfuel.synthetic syn.csv --flights 30
uv run neuralfuel profile syn.csv --raw
uv run neuralfuel check syn.csv --adapter synthetic
uv run neuralfuel prep syn.csv --adapter synthetic --out prep
uv run neuralfuel train prep --out run --device cpu --steps 2000
uv run neuralfuel validate prep --model run/model.pt --out run/validation.json
uv run neuralfuel export --model run/model.pt --out run/model.onnx
uv run neuralfuel contribute --run run --report run/validation.json --out share
```

The steps:

1. `neuralfuel.synthetic` writes 30 synthetic flights at 1 Hz.
2. `profile --raw` summarizes the raw columns without rows or identifiers.
3. `check` converts the file with the adapter and tests it against physics.
4. `prep` builds the 10 s blocks, the steady cruise windows and the split by flight.
5. `train` trains a model from scratch.
6. `validate` writes the validation report of the test split ([validation.md](validation.md)).
7. `export` writes the ONNX file and compares it with PyTorch.
8. `contribute` assembles the files of a contribution in the new folder `share/`, with an ONNX file made from the weights.

The numbers from the synthetic data only show that the cycle works. They say nothing about real aircraft. The quick start uses 2,000 update steps (about 1 minute on a CPU) to test the workflow; the default is 19,500. Never use a short step count for a contribution.

To train and validate both sizes in one call and write the files under their release names:

```sh
uv run neuralfuel release prep --out work/models --aircraft a320 --engine cfm56_5b --seeds 0 --steps 2000 --device cpu
```

It writes `nf-a320-cfm56_5b.pt`, `.onnx`, `.run.json` and `.validation.json`, and the same four files for `nf-a320-cfm56_5b-xs` (see [training.md](training.md)). Do not write to `models/`: that folder holds the shared models.

The same cycle through the Python API is [`examples/06_train_synthetic.py`](../examples/06_train_synthetic.py). The next steps with your own data are in [data.md](data.md) and [training.md](training.md).

## Run times

Measured on 16 CPU cores.

| step | time |
|---|---|
| `uv run --group casadi --group reference --group examples pytest -q` (full test suite) | about 7 minutes; about 4 minutes with `-n 4` (4 test processes) |
| `train` with the default 19,500 steps, one model | about 10 minutes on a CPU, about 3 minutes on a GPU |
| `train --steps 2000` | about 1 minute on a CPU |
| `merge --method combine` | about 10 minutes on a CPU |
| `release --seeds 0,1,2` | 6 trainings (3 seeds, 2 sizes) plus validation |
| each example script | see [`examples/README.md`](../examples/README.md) |

## Tests

```sh
uv run pytest
```

Without the optional groups, the tests of CasADi, OpenAP and the examples that need them are skipped. The full suite with all groups:

```sh
uv run --group casadi --group reference --group examples pytest -q
```

Tests use only the synthetic dataset and the shared models, never real data.
