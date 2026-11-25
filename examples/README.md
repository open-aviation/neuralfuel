# Examples

Run each script from the repository root. The first comment line of each script gives the command. The examples use only the shared models in [`../models/`](../models/README.md) and the synthetic data generator (`neuralfuel.synthetic`). The synthetic data is not real flight data, so the numbers of examples 6 to 8 only show how the steps work.

| script | shows | dependency group | run time |
|---|---|---|---|
| [`01_predict.py`](01_predict.py) | fuel flow at one flight state from altitude, Mach, mass and temperature; climb and descent with `sin_gamma` and `acc_in`; a state inside and outside the valid envelope | none | about 3 s |
| [`02_cruise_performance.py`](02_cruise_performance.py) | specific air range against Mach at several masses and altitudes, the best-range Mach, the best altitude against mass; `--plot` writes two figures | none; `examples` for `--plot` | about 4 s |
| [`03_trajectory_fuel.py`](03_trajectory_fuel.py) | fuel flow along a climb-cruise-descent profile from surveillance-type states, mass update along the flight, trip fuel | none | about 3 s |
| [`04_onnx_without_torch.py`](04_onnx_without_torch.py) | the ONNX file with only `onnxruntime` and `numpy`; aircraft settings and envelope from the ONNX metadata | none | about 1 s |
| [`05_casadi_optimize.py`](05_casadi_optimize.py) | the ONNX model in CasADi: check against PyTorch, then Mach and altitude of maximum specific air range with IPOPT inside the envelope | `casadi` | about 5 s |
| [`06_train_synthetic.py`](06_train_synthetic.py) | the single-user cycle in Python: generate, adapter, check, prepare, train, validate, export | none | about 50 s |
| [`07_federated_round.py`](07_federated_round.py) | one federated round: base model, two users with different coverage, protection, quality check, merge, error table | none | about 80 s |
| [`08_local_offsets.py`](08_local_offsets.py) | private per-aircraft phase offsets: fit, apply, bias before and after | none | about 7 s |
| [`09_compare_openap.py`](09_compare_openap.py) | the shared model against OpenAP on a small grid | `reference` | about 3 s |

Run times on 16 CPU cores with the default options. A script with a dependency group runs with `uv run --group NAME examples/SCRIPT.py`, for example:

```sh
uv run examples/01_predict.py
uv run --group examples examples/02_cruise_performance.py --plot
uv run --group casadi examples/05_casadi_optimize.py
uv run --group reference examples/09_compare_openap.py
```

`--plot` writes the PNG files into `examples/output/` (option `--out-dir`), which is in `.gitignore`. The figures in [`../docs/img/`](../docs/img/) come from `uv run --group examples examples/02_cruise_performance.py --plot --out-dir docs/img`.

The training examples take `--steps` (07 also `--base-steps`) for a shorter run. The test `tests/test_examples.py` runs every example with short step counts.

The documentation of each topic is in [`../docs/`](../docs/index.md).
