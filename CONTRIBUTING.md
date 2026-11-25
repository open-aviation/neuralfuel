# Contributing

You contribute a model, not data. You train on your own flight data on your own computer, and you send the trained model and its aggregate records in a pull request. The method is in [`docs/model.md`](docs/model.md), the federated workflow in [`docs/federated.md`](docs/federated.md). An AI coding agent can do the steps with [`AGENTS.md`](AGENTS.md).

## Contribute a model

1. Install and test the package.

   ```sh
   uv sync
   uv run pytest -q
   ```

2. Write an adapter for your data in the folder `local/` and put the data path in `local.toml`. Both are in `.gitignore`. See "Adapters" and "A private adapter" in [`docs/data.md`](docs/data.md).

3. Check and prepare the data.

   ```sh
   uv run neuralfuel check
   uv run neuralfuel prep --out work/prep
   ```

4. Validate the current shared model on your data. A merged shared model has no validation report from data, because the maintainers have no flight data. This report is its validation on your data.

   ```sh
   uv run neuralfuel validate work/prep --model models/a320/nf-a320-cfm56_5b.pt --out work/shared_validation.json
   ```

5. Continue the training from the shared model in `models/<type code>/`. Use the model of your aircraft type and engine family. Keep the protection against forgetting on: do not add `--no-protect`. Keep the default budget of 19,500 update steps. If `train` stops because the drag table of the shared model cannot be reproduced, install the aerox version of the shared model that the message names.

   ```sh
   uv run neuralfuel train work/prep --init models/a320/nf-a320-cfm56_5b.pt --out work/run
   ```

6. Validate on your test split and export to ONNX.

   ```sh
   uv run neuralfuel validate work/prep --model work/run/model.pt --out work/run/validation.json
   uv run neuralfuel export --model work/run/model.pt --out work/run/model.onnx
   ```

7. Run the quality check against the shared model. It reads the run record `work/run/run.json`. Exit code 1 means a check failed; see "When check-model fails".

   ```sh
   uv run neuralfuel check-model work/run/model.pt --base models/a320/nf-a320-cfm56_5b.pt
   ```

8. Assemble the contribution.

   ```sh
   uv run neuralfuel contribute --run work/run --report work/run/validation.json --out work/share
   ```

   `work/share` must be a new or empty folder. The command writes the ONNX file itself, from the verified weights. It copies only the keys of the record formats into `run.json` and `validation.json` and lists every key it drops (`dropped_fields`). The command stops with exit code 1 if the run record or the validation report contains a field that looks like an identifier, a date or a position. It stops with exit code 2 if the folder is not empty, if the report is not of the test split only, or if the report or the run record belongs to other weights.

9. Open a pull request. Put the files of `work/share` in a new folder `contributions/<short neutral name>/`. The `.gitignore` keeps `.pt` and `.onnx` files in `contributions/` tracked. Fill in the checklist of the pull request template.

### Files to include

- `model.pt`: the weights.
- `model.onnx`: the ONNX export, made by `contribute` from `model.pt`.
- `run.json`: the run record. It names the base model by its weights hash (`init_weights_sha256`), holds the hash of the weights (`weights_sha256`), the number of aircraft (`n_aircraft`) and the envelope of your training data (`local_envelope`, the declared region).
- `validation.json`: the validation report of the test split.
- `MANIFEST.json`: the checksums and a short summary, written by `contribute`.

In the pull request text, give the aircraft type, the engine family, the number of flights per split and the number of aircraft (`n_aircraft` in `run.json`).

### Never include

- flight data in any form: raw files, prepared blocks and windows (`work/prep`), samples, plots of single flights;
- flight identifiers, aircraft registrations, dates, times, positions, airports or routes;
- the name of the data owner or of an airline;
- data paths, `local.toml`, your adapter in `local/`;
- the private local offsets of `neuralfuel local-fit`;
- BADA values or coefficients.

If you are not sure about a file, leave it out and ask in the pull request.

### When check-model fails

Each check gives a value, a warning limit, a failure limit and a status. The table in "Quality check of a contribution" in [`docs/federated.md`](docs/federated.md) gives the unit and the details of each check. In short:

- `finite` (yes or no): all parameters and outputs are finite.
- `compatible` (yes or no): same architecture, inputs, temperature rule, aircraft settings and standardization as the base.
- `base` (yes or no): the run record names the base model.
- `protection` (teacher weight): the protection against forgetting was on; 0 fails.
- `drag_table` (yes or no): the installed aerox rebuilds the drag table of P7 of the base and of your model. Only for a model with P7 settings.
- `constraint_delta` (percentage points): increase of the violation rates of the physical constraints.
- `fuel_level` (%): median fuel-flow difference to the base in steady cruise inside your data region.
- `region_type` (fraction): share of your data region with a mass or Mach outside the range of the type.
- `change_outside_mean`, `change_outside_max` (%): change of fuel flow outside your data region.
- `outside_states` (number): test states outside your data region; few states make the two checks above weak and give a warning.
- `mach_mean`, `mach_max` (1 per Mach): change of the cruise Mach sensitivity d ln F / dM.
- `best_mach` (Mach): change of the Mach of maximum specific air range.
- `smoothness` (ratio): roughness of the contribution relative to the base.

What to do:

- `finite`, `compatible`, `base`, `protection`: train again from the shared model with `--init`, without `--no-protect` and without `--set temperature=...` or aircraft settings.
- `drag_table`: install the aerox version of the shared model that the note names and run the check again.
- `fuel_level`, `region_type`: check the units and scale factors of your adapter with `neuralfuel check`. Mass in pounds and fuel flow in another unit are the usual causes.
- the other checks: the training moved the model far from the base. Check the protection and the step budget.

Do not submit a contribution that fails. If you think that a failure is wrong, open the pull request as a draft and explain it. A model trained on the synthetic data of the quick start is expected to fail `fuel_level` and the Mach checks against the shared A320 model, because the physics of the synthetic data differs from the real aircraft.

## Review by the maintainers

The maintainers need no flight data for the review.

1. Quality check against the shared model that the contribution continued from. The run record is required. Exit code 1 means a check failed; exit code 2 means the run record is missing or belongs to other weights.

   ```sh
   uv run neuralfuel check-model contributions/NAME/model.pt --base models/a320/nf-a320-cfm56_5b.pt --run contributions/NAME/run.json
   ```

2. Comparison with OpenAP and with the current shared model.

   ```sh
   uv run --group reference neuralfuel compare --model contributions/NAME/model.pt --against models/a320/nf-a320-cfm56_5b.pt
   ```

   A maintainer with a BADA licence and a local BADA 4 folder (`NEURALFUEL_BADA`, or `bada_path` in the `[reference]` table of `local.toml`) also compares with BADA 4 (`TYPE` is the BADA name of the aircraft type). The output then holds ratios only.

   ```sh
   uv run --group reference neuralfuel compare --model contributions/NAME/model.pt --reference bada:TYPE
   ```

3. Review the run record and the validation report: the base model hash, the protection, the accuracy per phase, the declared region and the coverage.

4. Merge the accepted contributions into the next shared model. Write it to a new folder first, not over the base.

   ```sh
   uv run neuralfuel merge contributions/NAME1 contributions/NAME2 --base models/a320/nf-a320-cfm56_5b.pt --out next/nf-a320-cfm56_5b.pt
   ```

   `merge` writes the weights `next/nf-a320-cfm56_5b.pt` and the merge record `next/nf-a320-cfm56_5b.merge.json`: the method, the hashes of the base and of the contributions, the weights, the union envelope and the result of the quality check of the merged model against the base. Exit code 1 means a contribution was refused; the output names the reason. `--method combine` trains a student instead of averaging the weights (about 10 minutes on a CPU).

5. Export the merged model to ONNX.

   ```sh
   uv run neuralfuel export --model next/nf-a320-cfm56_5b.pt --out next/nf-a320-cfm56_5b.onnx
   ```

6. Release. Copy `nf-a320-cfm56_5b.pt`, `.onnx` and `.merge.json` into `models/a320/`, replacing the previous shared model, and update the model card `models/a320/README.md`. The previous `.run.json` and `.validation.json` belong to the previous weights: remove them from `models/`. `tests/test_shared_models.py` accepts both layouts: a model trained directly has `.run.json` (and `.validation.json`), a merged model has `.merge.json`; the test checks the weights hash against the record that exists and fails without one (see [`models/README.md`](models/README.md)). A merged model has no validation report from data: the validation reports of the contributions stay with the contributions in `contributions/`, and the users validate the new shared model on their own data in the next round (step 4 of "Contribute a model").

The quality check finds honest mistakes, such as wrong units, missing protection or diverged training. It is not a security barrier. The review of each pull request and the comparison with reference models are also necessary.

## Code changes

Code changes are welcome in separate pull requests. Follow "Code changes" in [`AGENTS.md`](AGENTS.md): format with `uvx ruff@0.17.6 format` and lint with `uvx ruff@0.17.6 check`, add a test in `tests/`, use only the synthetic dataset and the shared models in tests, and keep `docs/` and `examples/` in step with the code. Run `uv run --group casadi --group reference --group examples pytest -q` before you open the pull request.

Run `uvx pre-commit install` once after you clone the repository. The hook then runs the ruff lint with fixes and the ruff format on each commit.
