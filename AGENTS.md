# Instructions for AI coding agents

This file tells an AI coding agent how to help a user train and contribute a model with neuralfuel. It is complete: follow it step by step. For details, the documentation is in `docs/` (start at `docs/index.md`): the method in `docs/model.md`, the data and adapters in `docs/data.md`, training in `docs/training.md`, the validation report in `docs/validation.md`, the federated workflow and the quality check in `docs/federated.md`, all commands and exit codes in `docs/cli.md`.

## Privacy rule

The raw data of the user must not leave the machine and must not enter your context.

- Work only from the output of `neuralfuel profile` and `neuralfuel check`. These print column names, types, missing shares, ranges of recognized measurement columns and test results, never rows. `profile` gives value ranges (and the share of True of a flag) only for the numeric and flag columns of the schema and, with `--raw`, for the raw columns that the adapter lists in `MEASUREMENTS`. Every other column gives only its type and its share of missing values.
- Do not print raw rows, `head()`, `sample()`, unique values of text columns, flight identifiers, aircraft registrations, dates or positions. Do not open the data file in a viewer.
- Do not write the data path, the name of the data owner or airline, aircraft registrations or flight identifiers into code, comments, tests, file names or commit messages. The data path goes only into `local.toml` or the environment variable `NEURALFUEL_DATA`.
- Share only the folder that `neuralfuel contribute` writes into a new or empty folder. It holds the weights, the ONNX file, the run record, the validation report and the manifest, and nothing else.

## Run times

Measured on 16 CPU cores. Set the timeout of a command above these values.

| step | time |
|---|---|
| `uv run --group casadi --group reference --group examples pytest -q` (full test suite) | about 7 minutes; about 4 minutes with `-n 4` (4 test processes) |
| `train` with the default 19,500 steps, one model | about 10 minutes on a CPU, about 3 minutes on a GPU |
| `train --steps 2000` | about 1 minute on a CPU |
| `merge --method combine` | about 10 minutes on a CPU |
| `release --seeds 0,1,2` | 6 trainings (3 seeds, 2 sizes) plus validation; `--jobs 6` runs them at the same time |

A short step count (`--steps 2000`) is allowed only to test the workflow, for example on the synthetic data. Never use it for a contribution or a release.

## Workflow

Run every command from the repository root with `uv run`. Add `--json` when you parse the output.

1. Set up the environment (the tests take about 7 minutes).

   ```sh
   uv sync
   uv run --group casadi --group reference --group examples pytest -q
   ```

2. Tell the user to create `local.toml` (it is in `.gitignore`) with the data path. Do not create it with a real path yourself unless the user gives you the path for that purpose.

   ```toml
   [data]
   path = "/path/to/flights.parquet"
   adapter = "canonical"
   ```

3. Profile the raw file.

   ```sh
   uv run neuralfuel profile --raw --json
   ```

   The raw profile gives ranges only for raw columns with schema names; every other raw column gives only its type and its share of missing values. Use the column names and types to choose the raw columns of the schema quantities (table in `docs/data.md`, section "Schema", and `src/neuralfuel/schema.py`). Then list the chosen raw measurement columns in `MEASUREMENTS` of the adapter (step 4) and run `uv run neuralfuel profile --raw --json` again: it then gives their ranges, from which you get the units. List only measurements, never a column that can identify a flight, an aircraft, a time or a position. The raw profile has a sampling rate only if the raw time column is a datetime column; for a text time column, `sampling` has no rate. Then get the rate from `profile` after the adapter (step 5).

4. Write an adapter. Copy `src/neuralfuel/adapters/example_a320.py` (a 1 Hz table) to a new file in the folder `local/` (it is in `.gitignore`), for example `local/my_fleet.py`, with a neutral name. The example uses absolute imports, so the copy works unchanged. If you add imports from the package, make them absolute too: `from neuralfuel.adapters import read_table`, `from neuralfuel.atmos import FT, KT`; a relative import such as `from . import read_table` or `from ..atmos import FT` fails in `local/`. For recorder data with parameters at different rates, resample every parameter to 1 Hz inside each continuous stretch of a recording; do not interpolate across a recording boundary or a time gap. Convert every column to the schema unit (kt, ft, kg, kg/h, K). Set `clean` to True only when flaps and slats are both retracted; a flap position alone often reads 0 with slats extended. If the data has them, also set `speedbrake` (speed brake extended in flight), `gear_down` (gear not up and locked) and `anti_ice` (engine anti-ice on); `prep` removes speed-brake and gear-down samples. Give aircraft only neutral labels in `tail` (A, B, ...) and give flights neutral codes. Set `adapter = "local/my_fleet.py"` in `local.toml`; the path starts from the folder of `local.toml`. The commands then load the function `load` of that file. Put the list `MEASUREMENTS` of the raw measurement columns in the adapter file, as in the example. A flag column (`clean` and the optional flags) must not have missing values. Details: `docs/data.md`, section "Adapters".

5. Check the adapter output.

   ```sh
   uv run neuralfuel profile --json
   uv run neuralfuel check --json
   ```

   Exit code 1 means a test failed. Each failed test names the suspect column and the likely cause (for example a temperature scale factor or fuel flow in kg/s). The tests and their limits are listed in `docs/data.md`, section "Checks". Fix the adapter and repeat until `check` passes. Exit code 2 means bad input; the `column` field names the column. If the adapter raises an exception, the `error` field gives only the operation, the class of the exception and the function and line in the adapter (for example `adapter 'local/my_fleet.py' failed: ValueError in load, line 31 of the adapter (...)`), never the message of the exception, because it can hold a value of the data. Read that line of the adapter to find the cause. Do not run the adapter outside the tool to see the message, and do not ask the user to paste it.

   Read the `flags` section of the `check` output: the share of samples in flight that are clean, with speed brake, with gear down and with anti-ice on, overall and by altitude band. Above 25,000 ft nearly all samples are clean with gear up, the speed brake is rare and mostly in descent, and the gear is down only below about 10,000 ft. Other shares point to a wrong decoding (for example an inverted flag).

6. Prepare the blocks and windows.

   ```sh
   uv run neuralfuel prep --out work/prep --json
   ```

   Report the counts per split from the output. They are counts only.

7. Train. There are two paths. Use path A unless the user tells you otherwise.

   **Path A, contribution (default): continue from the shared model.** Use the model in `models/<type code>/` of the aircraft type and engine family of the user. First validate the shared model on the data of the user; a merged shared model has no other validation from data.

   ```sh
   uv run neuralfuel validate work/prep --model models/a320/nf-a320-cfm56_5b.pt --out work/shared_validation.json --json
   uv run neuralfuel train work/prep --init models/a320/nf-a320-cfm56_5b.pt --out work/run --seed 0 --device auto --json
   ```

   The standardization constants, the temperature rule, the aircraft settings and the constraint settings of the shared model are kept, the settings of the drag-rise term P7 included. If the training stops because the drag table of the shared model cannot be reproduced, tell the user the two aerox versions of the message; the fix is to install the aerox version of the model. `--aircraft` and `--engine` are ignored with `--init`. The protection against forgetting is on; do not add `--no-protect` and do not change `--set temperature`. Keep the default budget of 19,500 update steps (about 10 minutes on a CPU, 3 minutes on a GPU). The first progress line names the base model hash and the protection setting. If the run record has a `protection.warning`, tell the user: the given model file has no valid envelope.

   **Path B, model from scratch: a new aircraft type, or a first model of a type.** Give the aircraft type and, for an engine-specific model, the engine family.

   ```sh
   uv run neuralfuel train work/prep --out work/run --seed 0 --aircraft a320 --engine cfm56_5b --device auto --json
   ```

   The budget is 19,500 update steps for every dataset size; do not lower it for a small dataset. `--size xs` trains the small size (3 layers of 32). The default aircraft settings come from the aerox library (MMO, VMO, OEW to MTOW, 10,000 ft to the ceiling, wing area). For a type that aerox does not have, or a value that aerox does not have (`train` stops with "envelope values missing"), ask the user for the missing values (maximum operating Mach and CAS, mass and altitude ranges, reference wing area) and give them with `--set aircraft.mmo=...` and so on; a value given with `--set` overrides the aerox value (`aircraft.wing_area_m2` is optional; without it the envelope has no lift coefficient). The drag-rise term P7 is on by default for a type that aerox has with a drag model with wave drag. For another type the training runs without P7 and the output has the note `drag-rise term P7 off: <reason>` (also in `drag_rise_note` of the run record); this is not an error. Tell the user about the note. Do not set `loss.drag_rise`. To train and validate both sizes in one call, with the release file names (`nf-<type>[-<engine family>][-xs]`):

   ```sh
   uv run neuralfuel release work/prep --out work/models --aircraft a320 --engine cfm56_5b --seeds 0,1,2 --json
   ```

   A model from scratch has no base model, so `check-model` does not apply to it (its `base` check fails).

8. Validate on the test split.

   ```sh
   uv run neuralfuel validate work/prep --model work/run/model.pt --out work/run/validation.json --json
   ```

   Add `--surveillance` for the optional surveillance-input test. It reads the 1 Hz samples through the adapter of `local.toml`. Each value of the report is explained in `docs/validation.md`.

9. Export to ONNX (optional; `contribute` makes the ONNX file of the contribution itself).

   ```sh
   uv run neuralfuel export --model work/run/model.pt --out work/run/model.onnx --json
   ```

10. Path A only: run the quality check against the shared model. It reads `work/run/run.json`.

    ```sh
    uv run neuralfuel check-model work/run/model.pt --base models/a320/nf-a320-cfm56_5b.pt --json
    ```

    Exit code 1 means a check failed; read the `checks` list and "When check-model fails" in `CONTRIBUTING.md` (each check is explained in `docs/federated.md`, section "Quality check of a contribution"). Tell the user which checks failed. If `drag_table` fails because a table is not reproduced, the installed aerox does not reproduce the drag table of P7; tell the user the aerox versions of its note, and install the aerox version of the shared model. If it fails because the P7 settings differ from the base, train again with `--init` from the shared model. Do not prepare a contribution that fails. A model trained on the synthetic data is expected to fail `fuel_level` and the Mach checks against the shared A320 model.

11. Assemble the contribution.

    ```sh
    uv run neuralfuel contribute --run work/run --report work/run/validation.json --out work/share --json
    ```

    `work/share` must be a new or empty folder. The command writes `model.pt`, `model.onnx` (made from the verified weights), `run.json`, `validation.json` and `MANIFEST.json`. It keeps only the keys of the record formats with values of the declared shapes (numbers, bounded lists of numbers, strings of a fixed set or a strict pattern) and lists the path of every dropped field in `dropped_fields` of its output; tell the user about them. The list is only in the output, not in the folder (`MANIFEST.json` has only their number). The report must be of the test split only (the default of `validate`). The command stops with exit code 1 if the report or the run record contains a field that looks like an identifier, a date or a position, and with exit code 2 if the folder is not empty, the report is not of the test split, or the files belong to other weights. It runs no git or network command. The user puts the files of `work/share` in a new folder `contributions/<short neutral name>/` and opens a pull request.

12. Optional, private: fit the local offsets of each aircraft against the model the user applies.

    ```sh
    uv run neuralfuel local-fit work/prep --model work/run/model.pt --out work/private/offsets.json --json
    ```

    The offsets file stays with the user. Never copy it into `work/share` or into a pull request. The offsets make a step at the phase borders: use them for fuel estimation, not inside a trajectory optimizer. Details: `docs/federated.md`, section "Private local correction".

## Maintainer workflow

A maintainer reviews contributions and builds the next shared model. No flight data is needed.

1. Quality check of each contribution against the shared model it continued from. The run record is required. Exit code 1 means a check failed; read the `checks` list. Exit code 2 means the run record is missing or belongs to other weights.

   ```sh
   uv run neuralfuel check-model contributions/NAME/model.pt --base models/a320/nf-a320-cfm56_5b.pt --run contributions/NAME/run.json --json
   ```

   The run record is checked as in `contribute` and `merge`: the base hash of the run record and of the model file, the declared region (`local_envelope`, required) and the envelope of the model. The check finds honest mistakes (wrong units, missing protection, diverged training). It is not a security barrier: it cannot detect inaccurate airspeed in the training data or a false data region.

2. Comparison with reference models and with the current shared model.

   ```sh
   uv sync --group reference
   uv run --group reference neuralfuel compare --model contributions/NAME/model.pt --against models/a320/nf-a320-cfm56_5b.pt --json
   ```

   A best-range Mach with `model_outside_envelope` or `reference_outside_envelope`, or at a search bound, is an extrapolation or a limit of the search, not an optimum inside the envelope.

   For a BADA 4 comparison the user of the tool needs a BADA licence and a local BADA 4 folder (`NEURALFUEL_BADA` or `[reference] bada_path` in `local.toml`); add `--reference bada:NAME`. The output then holds ratios only. Never write BADA values or coefficients to a tracked file.

3. Merge the accepted contributions, then export the merged model.

   ```sh
   uv run neuralfuel merge contributions/NAME1 contributions/NAME2 --base models/a320/nf-a320-cfm56_5b.pt --out next/nf-a320-cfm56_5b.pt --json
   uv run neuralfuel export --model next/nf-a320-cfm56_5b.pt --out next/nf-a320-cfm56_5b.onnx --json
   ```

   `merge` writes the weights and the merge record `next/nf-a320-cfm56_5b.merge.json` with the quality check of the merged model against the base. Exit code 1 means a contribution was refused, or the quality check of the merged model could not run; then the files are written, but the merged model is unchecked: do not publish it. `--method combine` trains a student on the outputs of the contributions inside their declared regions instead of averaging the weights (about 10 minutes on a CPU; it uses the GPU if there is one). The release files are the `.pt`, the `.onnx` and the `.merge.json`; see "Review by the maintainers" in `CONTRIBUTING.md`. `tests/test_shared_models.py` checks a merged model against its `.merge.json` and a model trained directly against its `.run.json`. The aggregation methods and the merge record are explained in `docs/federated.md`.

## Code changes

- Keep the package small. Use functions and dataclasses; do not add frameworks or plugin systems.
- Format and lint every Python and Markdown file you change: `uvx ruff@0.17.6 format <file>` and `uvx ruff@0.17.6 check`. CI runs `uvx ruff@0.17.6 format --check` and `uvx ruff@0.17.6 check`.
- Run `uvx pre-commit install` once after you clone the repository. The hook then runs the ruff lint with fixes and the ruff format on each commit.
- Add a test in `tests/` for every change. Tests use only the synthetic dataset (`neuralfuel.synthetic`) and the shared models, never real data.
- Keep the documentation in `docs/` and the examples in `examples/` in step with the code. `tests/test_examples.py` runs every example.
- A new loss term gets a weight in `LossConfig` (`src/neuralfuel/losses.py`), a function in the same file, and two lines in the training step (`src/neuralfuel/train.py`).
- Values that depend on the aircraft type (maximum operating Mach and CAS, mass and altitude ranges, wing area) belong in `Aircraft` (`src/neuralfuel/aircraft.py`), not in the loss or validation code.
