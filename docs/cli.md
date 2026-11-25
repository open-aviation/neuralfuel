# Command reference

Run every command from the repository root with `uv run neuralfuel COMMAND`. `uv run neuralfuel COMMAND --help` prints the options. `uv run neuralfuel --version` prints the version.

## Common rules

- Every command accepts `--json` and then prints exactly one JSON object on stdout, with the fields `command`, `exit_code` and `ok`. This is also true for an error in the command-line arguments (exit code 2, the message in `error`). `train`, `release` and `merge --method combine` then write their progress lines to stderr. The JSON output and every JSON file that the commands write are standard JSON: a missing or non-finite value is `null`, never `NaN` or `Infinity`.
- The commands check the options and the `--set` settings before they start any work: a name that is not a setting, a value of the wrong type (for example text for a number) and a value out of range (for example `--steps 0`) give exit code 2.
- Without `--json`, a command prints text: on stdout for exit code 0, on stderr otherwise.
- The data commands (`profile`, `check`, `prep`) take the data file as the optional argument `DATA`. Without it, the path comes from the environment variable `NEURALFUEL_DATA` or from `local.toml`. `--adapter` gives the adapter; without it, the adapter comes from `NEURALFUEL_ADAPTER`, from `local.toml` or is `canonical` (see [data.md](data.md)).

Exit codes:

| code | meaning |
|---|---|
| 0 | success |
| 1 | a check failed (including the schema validation in `check`), the privacy test of `contribute` failed, `merge` refused a contribution, the ONNX file of `export` or `release` differs from PyTorch, or a quality check of `check-model` failed |
| 2 | bad input: a missing file, a file that is not a model file, an unknown adapter, a schema error in `prep`, a missing `prep.json`, a missing run record, a run record of other weights, an unknown setting, a value of the wrong type or out of range, an error in the command-line arguments; the `column` field of the JSON output names the column when there is one |
| 3 | unexpected error |

## Commands

| command | does |
|---|---|
| [`profile`](#profile) | summarize columns, ranges and sampling (no rows, no identifiers) |
| [`check`](#check) | schema and physical plausibility tests after the adapter |
| [`prep`](#prep) | build 10 s blocks, steady windows and the split |
| [`train`](#train) | train a model on prepared data |
| [`release`](#release) | train both sizes (best seed by validation loss), validate and export ONNX |
| [`validate`](#validate) | validation report (JSON, aggregate numbers only) |
| [`export`](#export) | export the network to ONNX and compare with PyTorch |
| [`contribute`](#contribute) | assemble weights, ONNX, run record and report in one folder |
| [`merge`](#merge) | aggregate contributions that continued from the same base model |
| [`local-fit`](#local-fit) | fit the private offsets (climb, level, descent) per aircraft; never shared |
| [`check-model`](#check-model) | quality check of a contribution against its base model (no flight data) |
| [`compare`](#compare) | compare a model with reference fuel models on a grid inside its envelope |

### profile

`neuralfuel profile [DATA] [--adapter ADAPTER] [--raw] [--json]`

Column names, types, missing fractions, value ranges and the sampling rate. No rows, identifiers, dates or positions. Value ranges (and the share of True of a flag) are given only for the columns that the tool recognizes as measurements: the numeric and flag columns of the schema, and with `--raw` also the raw columns that the adapter lists in `MEASUREMENTS`. Every other column gives only its type and its share of missing values. The sampling rate needs a datetime time column: with `--raw` there is no rate when the raw time column is text.

| option | meaning |
|---|---|
| `DATA` | raw file (default: `NEURALFUEL_DATA` or `local.toml`) |
| `--adapter` | adapter name, `module:function` or file path; with `--raw`, only its list `MEASUREMENTS` is used |
| `--raw` | profile the raw file, without the adapter |

Exit codes: 0, 2 (no data path, missing file, unknown adapter, adapter error).

### check

`neuralfuel check [DATA] [--adapter ADAPTER] [--json]`

Schema validation and plausibility tests (Mach range, temperature against ISA, fuel flow per mass, mass decrease against integrated fuel flow, groundspeed against TAS, sampling rate, gear and speed-brake flags at high altitude); the share of samples in flight (CAS >= 100 kt) that are clean, with speed brake, with gear down and with anti-ice on, overall and by altitude band. The tests are listed in [data.md](data.md).

Options: `DATA`, `--adapter` as for `profile`.

Exit codes: 0 all tests passed, 1 a test or the schema validation failed, 2 bad input.

### prep

`neuralfuel prep [DATA] [--adapter ADAPTER] --out DIR [--set KEY=VALUE ...] [--json]`

Writes `blocks.parquet`, `windows.parquet` and `prep.json` into `DIR` (see "Preparation" in [data.md](data.md)). `prep.json` holds counts and the preparation settings (`config`). `train`, `release`, `validate` and `local-fit` read these settings, so the folder must have its `prep.json`.

| option | meaning |
|---|---|
| `DATA`, `--adapter` | as for `profile` |
| `--out` | output folder (required) |
| `--set KEY=VALUE` | a field of `PrepConfig`, for example `exclude_anti_ice=true` or `block_s=10`; repeat for several |

Exit codes: 0, 2 (bad input, a schema error, an unknown setting or a setting out of range, for example `block_s=0`).

### train

`neuralfuel train PREP --out RUN [options] [--json]`

Trains from scratch, or from `--init model.pt`. Writes `model.pt` and `run.json` into `RUN`. The phase settings of the envelope come from `PREP/prep.json`, and the run record keeps all preparation settings (`prep`). With `--init`, the temperature rule, the exponent, the aircraft settings and the constraint settings come from that file, `--aircraft` and `--engine` are ignored, and the protection against forgetting is on. With `--init`, the training rebuilds the drag-slope table of P7 of the given model with the installed aerox and stops with exit code 2 if the table does not have the stored SHA-256. The training stops with exit code 2 at the first non-finite loss. See [training.md](training.md).

| option | default | meaning |
|---|---|---|
| `PREP` | | folder written by `prep` |
| `--out` | | run folder for `model.pt` and `run.json` (required) |
| `--seed` | 0 | random seed |
| `--size` | `standard` | `standard` (3 hidden layers of 64) or `xs` (3 hidden layers of 32) |
| `--init` | | weights to continue from (`model.pt`) |
| `--no-protect` | off | with `--init`: no protection against forgetting (`loss.teacher=0`) |
| `--aircraft` | `a320` | aircraft type; ignored with `--init` |
| `--engine` | | engine family, for example `cfm56_5b` |
| `--steps` | 19500 | budget in update steps |
| `--patience-steps` | 3900 | early stopping: steps without a better validation loss |
| `--epochs` | | budget in passes over the data instead of steps (older runs) |
| `--patience` | 30 | with `--epochs`: patience in passes |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `cuda:N` |
| `--threads` | 4 | CPU threads |
| `--set KEY=VALUE` | | other settings, for example `loss.window=0.1`, `lr=0.001`, `temperature=free` or `aircraft.mmo=0.86`; repeat for several. `aircraft.*` is refused with `--init`, because the aircraft settings come from the given model. The drag-rise term P7 is on by default where aerox has a drag model with wave drag for the type; elsewhere the training runs without it and `notes` says so. `loss.drag_rise=0` switches it off (settings: [training.md](training.md)) |

`--steps`, `--patience-steps`, `--epochs`, `--patience` and `--threads` must be 1 or more.

Exit codes: 0, 2 (no prepared data or no `prep.json`, a missing or invalid weights file, `--patience` without `--epochs`, missing aircraft settings, an unknown setting, a value of the wrong type or out of range, a non-finite loss, a P7 table of the `--init` model that the installed aerox does not reproduce).

### release

`neuralfuel release PREP --out DIR [options] [--json]`

Trains the sizes of `--sizes`, one model per seed, keeps the one with the lowest validation loss per size, validates it on the test split (with the preparation settings of `PREP/prep.json`) and exports it to ONNX. Writes `nf-<type>[-<engine>][-xs]` `.pt`, `.onnx`, `.run.json` and `.validation.json` into `DIR`. The file name and the size label come from the trained network. Do not write to `models/`. The output has `notes`, with one note for each kept model that was trained without the drag-rise term P7 (with the reason).

| option | default | meaning |
|---|---|---|
| `PREP` | | folder written by `prep` |
| `--out` | | folder of the files (required) |
| `--seeds` | `0` | comma-separated seeds |
| `--jobs` | `1` | trainings that run at the same time, one process each; the number of CPU cores in use is about `--jobs` x `--threads`; the models are the same as with `--jobs 1` |
| `--sizes` | `standard,xs` | comma-separated sizes; the only way to set the network size |
| `--aircraft` | `a320` | aircraft type |
| `--engine` | | engine family, for example `cfm56_5b` |
| `--steps` | 19500 | budget in update steps |
| `--patience-steps` | 3900 | early stopping: steps without a better validation loss |
| `--epochs` | | budget in passes over the data instead of steps (older runs) |
| `--patience` | 30 | with `--epochs`: patience in passes |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `cuda:N` |
| `--threads` | 4 | CPU threads |
| `--set KEY=VALUE` | | other settings as for `train`; `--set hidden=...` and `--set layers=...` are refused |

The help text of `release` describes `--aircraft` as "ignored with --init"; `release` has no `--init`.

Exit codes: 0, 1 (an ONNX file differs from PyTorch by 1e-4 or more in relative fuel flow), 2 (bad input, an unknown size, `--set hidden` or `--set layers`).

### validate

`neuralfuel validate PREP --model M --out FILE [options] [--json]`

Writes the validation report (see [validation.md](validation.md)). The block length, the window length and the phase settings come from `PREP/prep.json`; the constraint settings and the aircraft settings come from the model file.

| option | default | meaning |
|---|---|---|
| `PREP` | | folder written by `prep` |
| `--model` | | `model.pt` (required) |
| `--out` | | report file, for example `validation.json` (required) |
| `--splits` | `test` | comma-separated splits to evaluate: `train`, `val`, `test` |
| `--aircraft` | | aircraft type, only for model files without aircraft settings |
| `--engine` | | engine family, with `--aircraft` |
| `--threads` | 4 | CPU threads |
| `--surveillance` | off | also run the surveillance-input test; it reads the 1 Hz samples through the adapter |
| `--data` | `local.toml` | raw file for `--surveillance` |
| `--adapter` | | adapter for `--surveillance` |

Exit codes: 0, 2 (no prepared data or no `prep.json`, a missing or invalid weights file, a model without aircraft settings and no `--aircraft`, an unknown split name, no blocks in the splits, P7 settings without `table_sha256`).

### export

`neuralfuel export --model M --out FILE [--casadi] [--json]`

ONNX export, checked against PyTorch on 1,000 random inputs. For every temperature setting the graph takes the six inputs `x` (n, 6) in the order of the inputs, ln theta included, applies the rule inside and returns `ln_fuel_flow` (n, 1). The aircraft settings are in the metadata entry `neuralfuel_aircraft`, the valid envelope in `neuralfuel_envelope`.

| option | meaning |
|---|---|
| `--model` | weights file (required) |
| `--out` | ONNX file (required) |
| `--casadi` | write Softplus as log(1 + exp(x)), for ONNX importers without Softplus (`casadi_function` does this itself, so the option is not needed for it) |

With `--casadi`, exp overflows in float32 above about 88.7. The export is refused (exit code 2, no file written) when the largest hidden pre-activation at 4,096 states inside the envelope and at the check inputs is above 80 (`MAX_PREACTIVATION` in `neuralfuel.inference`). Export without `--casadi` then.

Exit codes: 0, 1 (the largest relative fuel-flow difference is 1e-4 or more), 2 (a missing or invalid weights file, a `--casadi` export that can overflow).

### contribute

`neuralfuel contribute --run RUN --report FILE --out DIR [--json]`

Assembles a contribution in a new folder: the weights `model.pt`, the ONNX file `model.onnx`, the run record `run.json`, the validation report `validation.json` and `MANIFEST.json` with checksums and a short summary. Runs no git and no network command.

- `DIR` must not exist or must be an empty folder. The files are written into a temporary folder next to `DIR`, which is renamed to `DIR` at the end, so `DIR` holds only these five files.
- `model.pt` is a new file, written from the typed fields of the given weights; its meta entry holds only `weights_sha256` and `init_weights_sha256`. `model.onnx` is exported from it and checked against PyTorch.
- `run.json` and `validation.json` are copies with only the keys of their current formats (`RUN_RECORD` in `neuralfuel.train`, `REPORT` in `neuralfuel.validate`). An unknown key, and a value of the wrong kind, is dropped and listed in `dropped_fields` of the output and of `MANIFEST.json`.
- The report must be of the test split only (`validate --splits test`, the default).
- The run record is checked as in `check-model` and `merge` (see "Quality check of a contribution" in [federated.md](federated.md)): the weights hash, the base hash of the run record and of the meta entry, the declared region of a continued model and the envelope of the weights.

| option | meaning |
|---|---|
| `--run` | run folder with `model.pt` and `run.json` (required) |
| `--report` | `validation.json` of the test split (required) |
| `--out` | new or empty output folder (required) |

Exit codes: 0, 1 (the run record or the report contains a field that looks like an identifier, a date or a position; the ONNX file differs from PyTorch), 2 (a missing file; `DIR` exists and is not empty; a report that is not of the test split only; the report or the run record belongs to other weights; the run record and the weights file name different base models; a continued model without a declared region; an envelope that does not agree with the run record).

### merge

`neuralfuel merge DIR... --base S --out FILE [options] [--json]`

Aggregates contributions that continued from the same base model (see [federated.md](federated.md)). Needs the run record of every contribution. Writes `FILE` and the merge record `<name>.merge.json` next to it. `--method combine` takes the phase rule and the constraint settings from the base, and checks that the envelopes can be joined before it trains.

| option | default | meaning |
|---|---|---|
| `DIR...` | | contribution folders (`model.pt`, `run.json`) or weights files (the run record is then `run.json` next to a file named `model.pt`, else `<stem>.run.json`) |
| `--base` | | the shared model they continued from (required) |
| `--out` | | merged weights file (required) |
| `--method` | `average` | `average`: weight averaging; `combine`: output combination |
| `--weights` | `equal` | `average`: `equal`, `blocks` (training blocks from `run.json`) or comma-separated numbers |
| `--seed` | 0 | `combine`: student seed |
| `--steps` | 19500 | `combine`: update steps |
| `--device` | `auto` | `combine`: `auto`, `cpu`, `cuda` |
| `--threads` | 4 | CPU threads |

Exit codes: 0; 1 (a contribution was refused: another architecture, temperature rule, aircraft settings or standardization constants, a run record of other weights, a run record that names another base model or no base model, a model file that names another base than its run record, a run record without `local_envelope`, counts or an envelope that do not agree with the declared region, non-finite parameters, the same contribution twice, the base itself); 2 (a missing run record, bad `--weights`, envelopes that cannot be joined, with `combine` a P7 table of the base that the installed aerox does not reproduce). The quality check of the merged model is written into the merge record; its failure does not change the exit code.

### local-fit

`neuralfuel local-fit PREP --model M --out FILE [--split S] [--json]`

Fits the private offsets (climb, level, descent) per aircraft and writes them to their own JSON file. The phase threshold `level_fpm` comes from `PREP/prep.json`. Never share this file (see [federated.md](federated.md)).

| option | default | meaning |
|---|---|---|
| `PREP` | | folder written by `prep` |
| `--model` | | `model.pt` to correct (required) |
| `--out` | | offsets file (required) |
| `--split` | `train` | split to fit on |

Exit codes: 0, 2 (no prepared data or no `prep.json`, a missing or invalid weights file, no blocks in the split).

### check-model

`neuralfuel check-model M --base S [--run run.json] [--json]`

Quality check of a contribution against its base model, without flight data (see [federated.md](federated.md)).

| option | meaning |
|---|---|
| `M` | the contribution (`model.pt`) |
| `--base` | the shared model it continued from (required) |
| `--run` | run record of the contribution (default: `run.json` next to `M`) |

Exit codes: 0 no check failed (warnings are allowed), 1 at least one check failed (also `base` for a model file that names another base than its run record, `declared_region` for a run record without `local_envelope`, and `drag_table` for a P7 table that the installed aerox does not reproduce), 2 (no run record, a run record of other weights, a base without aircraft settings or envelope, P7 settings without `table_sha256`).

### compare

`neuralfuel compare --model M [--reference openap|bada:NAME] [--against M2] [--json]`

Fuel-flow ratio, cruise Mach sensitivity and best-range Mach against reference models on a grid inside the envelope of the model (see [federated.md](federated.md)). OpenAP needs the dependency group `reference`: `uv run --group reference neuralfuel compare ...`. A neuralfuel model is named `model:<file stem>-<first 8 characters of its weights hash>` in the output; the output holds no file paths. A best-range Mach is marked `*` at the upper bound of the search (the MMO), `_` at the lower bound and `!` outside the valid envelope of the model.

| option | meaning |
|---|---|
| `--model` | `model.pt` (required) |
| `--reference` | `openap[:TYPE]` (default) or `bada:NAME` (local licensed tables, ratios only); repeat for several |
| `--against` | another neuralfuel model, for example the previous shared model; repeat for several |

Exit codes: 0, 2 (OpenAP not installed, no BADA folder, a model without envelope or aircraft settings, the same reference twice).
