# Training

The commands below assume that `local.toml` names the data file and the adapter (see [data.md](data.md)). With the synthetic data of the quick start, add the data file and `--adapter synthetic` to `prep`, as in [quickstart.md](quickstart.md). The examples use the folders `work/prep` and `work/run`; `work/` is in `.gitignore`.

## Prepare

```sh
uv run neuralfuel check
uv run neuralfuel prep --out work/prep
```

`prep` writes the 10 s blocks, the steady windows and the split by flight (see "Preparation" in [data.md](data.md)). The output gives the counts per split. `prep.json` records the preparation settings; `train`, `release`, `validate` and `local-fit` read them, so keep it in the folder.

## Train from scratch

For a new aircraft type, or the first model of a type, give the aircraft type and, for an engine-specific model, the engine family:

```sh
uv run neuralfuel train work/prep --out work/run --seed 0 --aircraft a320 --engine cfm56_5b
```

`--size xs` trains the small size (3 hidden layers of 32). The default aircraft settings come from the aerox library. For a type that aerox does not have, or a value that aerox does not have, give the aircraft settings with `--set`; a value given with `--set` also overrides the aerox value (see "Aircraft-type settings" in [model.md](model.md)). `aircraft.wing_area_m2` is optional; without it the envelope has no lift coefficient.

A model from scratch has no base model, so `check-model` does not apply to it (its `base` check fails).

## Continue from a shared model

A contribution continues the training of the shared model of the aircraft type and engine family:

```sh
uv run neuralfuel train work/prep --init models/a320/nf-a320-cfm56_5b.pt --out work/run
```

The standardization constants, the temperature rule, the aircraft settings and the constraint settings of the shared model are kept, so the models of different users stay compatible. `--aircraft` and `--engine` are ignored with `--init`. The training protects the shared model against forgetting (see [federated.md](federated.md)). The first output line gives the hash of the base model and the protection setting. `--no-protect` switches the protection off; do not use it for a contribution. If the run record has a `protection.warning`, the given model file has no valid envelope. The training rebuilds the drag-slope table of P7 from the stored settings with the installed aerox. If the rebuilt table does not have the stored SHA-256, the drag table of the shared model cannot be reproduced: the training stops with exit code 2, and the message names the aerox version of the model and the installed aerox version. Install the aerox version of the model in that case. A model with P7 settings and no stored SHA-256 also stops the training with exit code 2.

The valid envelope of a continued model is the union of the envelope of the shared model and the envelope of the local training data. The run record also holds the local envelope alone (`local_envelope`, the declared region). A model continued from a file without an envelope gets only the local envelope.

## Training budget

The budget is a number of update steps (default 19,500, batch 4,096). This is the number that the full A320 dataset (875 training flights) gets in about 150 passes (128 steps per pass). A fixed number of passes undertrains a small dataset, because it gets few updates; with a fixed number of steps, a small dataset gets as many updates as a large one. Do not lower the budget for a small dataset.

- The optimizer is AdamW (learning rate 0.002, weight decay 1e-6).
- The learning rate follows a cosine from 0.002 to 0.00002 over the steps.
- The validation loss (mean squared error of ln fuel flow on the validation blocks) is computed every 130 steps (`eval_every`).
- Training stops early after 3,900 steps without a better validation loss (`patience_steps`, option `--patience-steps`).
- The model with the lowest validation loss is kept.

The older budget in passes over the data is available with `--epochs` (and `--patience` in passes, default 30); run records written with it have `budget` `epochs`.

A short step count (for example `--steps 2000`) is allowed only to test the workflow, for example on the synthetic data. Never use it for a contribution or a release. Run times are in [quickstart.md](quickstart.md).

## Settings

`train` and `release` take these options (all options: [cli.md](cli.md)):

| option | default | meaning |
|---|---|---|
| `--seed` | 0 | random seed (`release`: `--seeds`) |
| `--size` | `standard` | `standard` or `xs` (`release`: `--sizes`) |
| `--aircraft`, `--engine` | `a320`, none | aircraft type and engine family; ignored with `--init` |
| `--steps` | 19500 | budget in update steps |
| `--patience-steps` | 3900 | early stopping |
| `--device` | `auto` | `auto` uses the GPU if there is one; `cpu`, `cuda` or `cuda:N` |
| `--threads` | 4 | CPU threads |
| `--set KEY=VALUE` | | any field of `TrainConfig` (`src/neuralfuel/train.py`) or of its `loss` (`LossConfig`, `src/neuralfuel/losses.py`) and `aircraft` entries |

The drag-rise term P7 (see "Loss and physical constraints" in [model.md](model.md)) is on by default for a type with an aerox drag model with wave drag. Its settings:

| setting | default | meaning |
|---|---|---|
| `loss.drag_rise` | 0.01 | weight; 0 switches the term off |
| `loss.drag_rise_mach_min` | 0.70 | lowest Mach of the P7 states; the highest is the MMO |
| `loss.drag_rise_cl_factor` | 1.1 | upper lift coefficient of the P7 states, as a factor on the largest lift coefficient of the cruise windows |

For a type where P7 is not available (aerox does not have the type, its drag model has no wave drag, no wing area, or no cruise lift coefficient in the envelope), the training runs without P7 and says so: a log line, a note in the `notes` of the command output and `drag_rise: null` with the reason in `drag_rise_note` of the run record. An explicit `--set loss.drag_rise=<positive>` gives the same note and no error, because the default weight is already positive.

Examples of `--set`: `--set loss.window=0.1`, `--set lr=0.001`, `--set temperature=free`, `--set aircraft.mmo=0.86`. A list value is written as `[a,b]`. The value must have the type of the setting (a number for a number, `true` or `false` for a flag); an unknown name, a wrong type or a value out of range (for example `eval_every=0`) stops the command with exit code 2 before the training. With `--init`, do not change the temperature rule, the aircraft settings or the constraint settings: they come from the given model.

The training stops with exit code 2 at the first non-finite training or validation loss, with the step and the loss terms in the message. A non-finite loss comes from non-finite inputs or targets in the prepared data.

A constant input (for example a fixed mass in the data) gets the standardization SD 1, also when its SD is not exactly 0 because of rounding: an SD below 1e-9 times the magnitude of the input counts as 0 (`CONSTANT_RTOL` in `neuralfuel.model`).

## Run record

`train` writes `model.pt` and the run record `run.json` into the run folder. The run record holds:

- `config`: all training settings;
- `init` and `init_weights_sha256`: from scratch, or the weights hash of the base model;
- `protection`: the teacher weight, the number of protection points, the number inside the shared envelope, the share inside the local envelope, and a `warning` when the base has no envelope;
- `features`, `temperature`, `theta_exp`, `aircraft`, `size`, `n_params`;
- counts: `n_train_blocks`, `n_val_blocks`, `n_train_windows`, `n_val_windows`, and `n_aircraft` (distinct `tail` labels, 1 without a `tail` column);
- the budget: `budget`, `steps_per_epoch`, `best_step`, `steps_run`, `best_epoch`, `epochs_run`;
- losses: `val_loss`, `train_loss`, `val_window_loss`;
- `run_time_s`, `device` (`cpu` or `cuda`, without a product name or an index), `torch`, `python`, `neuralfuel_version`;
- `envelope` (the valid envelope of the model) and `local_envelope` (the envelope of the local training data);
- `prep`: the preparation settings of `prep.json` (block length, phase thresholds, window settings);
- `weights_sha256`: the hash of the weights (`neuralfuel.model.state_hash`);
- `drag_rise`: the P7 settings, the aerox version, the grid of the drag-slope table and the SHA-256 of its values; null when the training ran without P7, with the reason in `drag_rise_note`;
- `history`: the validation loss and the mean loss terms at each evaluation.

The run record has no flight identifiers, dates, positions or aircraft labels. Its format (the allowed keys) is `RUN_RECORD` in `src/neuralfuel/train.py`. `model.pt` holds the weights, the standardization, the temperature rule, the aircraft settings, the constraint settings (`constraints`: the collocation envelope and the limits of P1 to P6 of the training, and the P7 settings when the training used P7), the valid envelope and the weights hashes in its meta entry.

## Validate and export

```sh
uv run neuralfuel validate work/prep --model work/run/model.pt --out work/run/validation.json
uv run neuralfuel export --model work/run/model.pt --out work/run/model.onnx
```

The validation report is described in [validation.md](validation.md). `export` writes the ONNX file and compares it with PyTorch on 1,000 random inputs; it exits with code 1 if the largest relative difference of the fuel flow is 1e-4 or more, or not finite.

## Both sizes: release

To train and validate both sizes in one call and write the files under their release names:

```sh
uv run neuralfuel release work/prep --out work/models --aircraft a320 --engine cfm56_5b --seeds 0,1,2
```

For each size (`--sizes`, default `standard,xs`) it trains one model per seed, keeps the one with the lowest validation loss, validates it on the test split and exports it to ONNX. The size comes only from `--sizes`: `--set hidden=...` and `--set layers=...` are refused, and the file name and the size label come from the trained network. It writes `nf-a320-cfm56_5b.pt`, `.onnx`, `.run.json` and `.validation.json`, and the same four files for `nf-a320-cfm56_5b-xs`. The run record also holds the validation loss of each seed (`seed_val_losses`). Do not write to `models/`: that folder holds the shared models.

### Speed

One training is one process. On a CPU it gains from up to about 8 threads (`--threads`, default 4). Measured for 1,000 update steps on the A320 dataset: 64 s with 1 thread, 26 s with 4, 21 s with 8, 22 s with 16, and 10 s on a GPU.

The trainings of `release` are independent. `--jobs N` runs N of them at the same time, one process each. The number of CPU cores in use is about `--jobs` x `--threads`. The models are the same as with `--jobs 1`. Measured for 4 trainings of 1,000 steps: 96 s with `--jobs 1` and 50 s with `--jobs 4` on a CPU (4 threads each), and 27 s with `--jobs 4` on a GPU. These times include validation and export.

```sh
uv run neuralfuel release work/prep --out work/models --aircraft a320 --engine cfm56_5b --seeds 0,1,2 --jobs 6
```

## Python

The same steps in Python, on synthetic data: [`examples/06_train_synthetic.py`](../examples/06_train_synthetic.py). The functions are `neuralfuel.prep.prepare`, `neuralfuel.train.train` and `write_run`, `neuralfuel.validate.validate` and `neuralfuel.inference.export_onnx`.
