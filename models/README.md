# Shared models

One folder per aircraft type. The folder name is the ICAO type code in lower case.

| Folder | Aircraft type | Models |
|---|---|---|
| [`a320/`](a320/README.md) | Airbus A320 | `nf-a320-cfm56_5b`, `nf-a320-cfm56_5b-xs` |

Each folder has a model card (`README.md`) and, for each model, these files:

| File | Content |
|---|---|
| `.pt` | The model: weights, standardization, temperature rule, aircraft settings, valid envelope. Training continues from this file. |
| `.onnx` | The same network for use without PyTorch (onnxruntime, CasADi). Made from the `.pt` with `neuralfuel export`. |
| `.run.json` | Training record (a model trained directly, for example with `neuralfuel release`). |
| `.validation.json` | Validation report (a model trained directly). |
| `.merge.json` | Merge record (a merged model, written by `neuralfuel merge`): the method, the hashes of the base and of the contributions, the union envelope and the quality check against the base. |

A model has one of two layouts:

- trained directly: `.pt`, `.onnx`, `.run.json`, `.validation.json`;
- merged: `.pt`, `.onnx`, `.merge.json`. A merged model has no validation report from data; the users validate it on their own data.

`tests/test_shared_models.py` checks that the weights hash of each `.pt` equals `weights_sha256` of its `.run.json` or, without one, of its `.merge.json`, and fails when a model has neither.

File names: `nf-<type>-<engine family>.pt` for the standard size, and the same name with `-xs` for the small size. A model for all engines of a type has no engine part: `nf-<type>.pt`.

The file names and the two sizes are explained in [`docs/model.md`](../docs/model.md), the use of the files in [`docs/use.md`](../docs/use.md). The maintainers publish a new shared model as described in [`docs/federated.md`](../docs/federated.md), section "Workflow for a maintainer".
