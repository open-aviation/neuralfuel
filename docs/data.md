# Data

## Suitable data

The training needs flight data with accurate airspeed, temperature, mass and fuel flow, at 1 Hz. Full-rate flight recorder data, such as quick access recorder (QAR) data, is one example. Any other source with these quantities can be used through an adapter.

Requirements:

- **Sampling**: 1 Hz samples, one flight per flight identifier. For recorder data with parameters at different rates, the adapter must resample every parameter to 1 Hz inside each continuous stretch of a recording. Do not interpolate across a recording boundary or a time gap.
- **Airspeed**: CAS, TAS or Mach from the air data of the aircraft. The model is only as good as the airspeed it is trained on. Do not train on airspeed from surveillance data (see "Rules of the method" in [model.md](model.md)).
- **Altitude**: pressure altitude (standard setting), in ft.
- **Temperature**: total or static air temperature.
- **Groundspeed**: for the energy-rate inputs.
- **Mass**: the gross mass of the aircraft. `check` tests that the mass decrease agrees with the integrated fuel flow.
- **Fuel flow**: the total fuel flow of all engines. Per-engine fuel flows are optional; they enable the engine-imbalance filter.
- **Configuration**: a flag for the clean configuration (flaps and slats retracted). Flags for the speed brake, the landing gear and the engine anti-ice are optional.

## Schema

An adapter returns one row per sample with these columns (`src/neuralfuel/schema.py`):

| column | unit | required |
|---|---|---|
| `flight_id` | text | yes |
| `time` | datetime, UTC | yes |
| `alt_ft` | ft, pressure altitude | yes |
| `cas_kt`, `tas_kt` or `mach` | kt, kt, - | at least one |
| `gs_kt` | kt | yes |
| `mass_kg` | kg | yes |
| `ff_kgh` | kg/h, all engines | yes |
| `tat_k` or `sat_k` | K | at least one |
| `clean` | bool, flaps and slats retracted | yes |
| `speedbrake` | bool, speed brake extended in flight | no (excluded when present) |
| `gear_down` | bool, landing gear not up and locked | no (excluded when present) |
| `anti_ice` | bool, engine anti-ice on | no (carried to the blocks) |
| `tail` | text, neutral aircraft label such as A or B | no (carried to the blocks) |
| `ff1_kgh`, `ff2_kgh` | kg/h per engine | no (engine imbalance filter) |
| `roll_deg` | deg | no (steady-window filter) |

If more than one airspeed is given, the pipeline uses CAS first, then Mach, then TAS. If both temperatures are given, it uses TAT. Mach comes from CAS and pressure altitude, SAT from TAT and Mach (recovery factor 1).

`neuralfuel.schema.validate` reports missing columns, wrong types, missing values in a text or flag column, and values outside physical bounds, with the likely unit error. A column with more than 0.1 % of its values outside its bounds is an error. A flag column (`clean`, `speedbrake`, `gear_down`, `anti_ice`) with a missing value is an error: set the flag in the adapter, for example `clean` False where the flap position is unknown.

## Adapters

An adapter is a function `load(path) -> pandas.DataFrame` that returns the schema. There are three ways to give an adapter to the commands (`--adapter`, the environment variable `NEURALFUEL_ADAPTER` or `adapter` in `local.toml`):

- a registered name: add the function to `ADAPTERS` in `src/neuralfuel/adapters/__init__.py`;
- `package.module:function` for a module on the Python path;
- the path of a Python file: `my_adapter.py` uses its function `load`, `my_adapter.py:function` another function. A relative path on the command line or in `NEURALFUEL_ADAPTER` starts from the working directory; a relative path in `local.toml` starts from the folder of that `local.toml`.

Without an adapter setting, the commands use `canonical`: a parquet or CSV file that already follows the schema.

The two included examples use absolute imports, so a copy of `src/neuralfuel/adapters/example_a320.py` or `src/neuralfuel/adapters/synthetic.py` in `local/` works unchanged:

- `example_a320`: a processed 1 Hz table of the A320 dataset (raw TAT = 4 x degC with an invalid floor of -127, a flap code for the clean configuration). Its flap code reads 0 in CONF 1 and it has no speed-brake or gear flags, so its `clean` flag includes approach segments in CONF 1 and speed-brake use. The shared A320 models were not trained with this adapter: their adapter decoded the CONF 1 bit and the speed-brake and gear flags from the raw recorder parameters (see the model card `models/a320/README.md`, section "Training data").
- `synthetic`: the raw format of the synthetic dataset (metres, m/s, tonnes, kg/s, degC).

### A private adapter

Keep an adapter for your own data out of the package. Put it in the folder `local/`, which is listed in `.gitignore`, and put the data path and the adapter in `local.toml` (also in `.gitignore`):

```toml
[data]
path = "/path/to/flights.parquet"
adapter = "local/my_adapter.py"
```

The data path can also come from the environment variable `NEURALFUEL_DATA`. Never put it in tracked files. The order is: command-line argument, then `NEURALFUEL_DATA` and `NEURALFUEL_ADAPTER`, then `local.toml` in the working directory (or the file named by `NEURALFUEL_CONFIG`). `local.example.toml` is a template.

A file adapter imports from neuralfuel with absolute imports, for example `from neuralfuel.adapters import read_table` and `from neuralfuel.atmos import FT, KT`. A relative import such as `from . import read_table` fails in `local/`.

Notes for an adapter:

- Convert every column to the schema unit (kt, ft, kg, kg/h, K).
- Set `clean` to True only when flaps and slats are both retracted. A flap position alone often reads 0 with slats extended.
- If the data has them, set `speedbrake`, `gear_down` and `anti_ice`.
- Give aircraft only neutral labels in `tail` (A, B, ...) and give flights neutral codes.

With `local.toml` in place, run `profile`, `check`, `prep`, `train`, `validate` and `contribute` as in the quick start, without the data argument.

If the adapter raises an exception, the commands give exit code 2 and a short message: the operation, the class of the exception and the function and line in the adapter file, for example `adapter 'local/my_adapter.py' failed: ValueError in load, line 31 of the adapter (...)`. The message of the exception is not shown, because it can hold a value of the data (for example a time or a flight identifier that a parser could not read). A `KeyError` for a missing column names the column if the name is a schema column or is written in the adapter file. Run the adapter on your computer, outside the tool, to see the full message; do not give that message to another person or an AI agent.

## Checks

`neuralfuel profile` reports column names, types, missing fractions and the sampling rate. It never prints rows, identifiers, dates or positions. Its guarantee:

- Value ranges (minimum, quantiles, maximum, number of distinct values) and the share of True of a flag are given only for the columns that the tool recognizes as measurements: the numeric and flag columns of the schema, and with `--raw` also the raw columns that the adapter lists in `MEASUREMENTS` (a list of column names in the module of the adapter function).
- Every other column gives only its type and its share of missing values: no range, no distinct values, no share of True.
- Text and datetime columns never give ranges. A recognized column with a name like lat, lon, time, date, reg, tail or flight, or with numbers that look like epoch times, gives no range either.

So `profile --raw` without an adapter shows the ranges of the raw columns that have schema names only. To see the ranges of other raw measurement columns (for example an altitude in metres), list them in `MEASUREMENTS` of the adapter and run `profile --raw --adapter ADAPTER`. List only measurements, never a column that can identify a flight, an aircraft, a time or a position. The sampling rate needs a datetime time column: `profile --raw` reports no sampling rate when the raw time column is text, for example ISO 8601 strings. Run `profile` after the adapter to get it.

`neuralfuel check` runs the schema validation and these tests on the adapter output (`src/neuralfuel/checks.py`). A failed test names the column it suspects and the likely cause. A test marked "warn" does not fail the check.

| test | column | expected |
|---|---|---|
| sampling rate | `time` | median step 1 s |
| regular 1 s steps (warn) | `time` | at least 0.9 of the steps are 1 s |
| cruise altitude | `alt_ft` | median of the flight maxima in [15,000, 47,000] ft |
| Mach range above 10,000 ft | airspeed | at least 0.99 of the clean samples with 0.2 < M < 0.92 |
| cruise Mach | airspeed | median in [0.55, 0.88] above 25,000 ft |
| temperature against ISA | temperature | median SAT - ISA in [-25, 25] K above 25,000 ft |
| temperature spread (warn) | temperature | SD of SAT - ISA at most 12 K above 25,000 ft |
| cruise fuel flow per mass | `ff_kgh` | median fuel flow / mass in [0.012, 0.09] per hour above 25,000 ft |
| groundspeed against TAS | `gs_kt` | median groundspeed / TAS in [0.7, 1.4] above 25,000 ft |
| mass decrease against integrated fuel flow | `ff_kgh`, `mass_kg` | median ratio in [0.8, 1.25] over continuous segments (same flight, no step above 10 s) of at least 30 min; without such a segment the test warns (insufficient coverage) |
| engine fuel flows add up | `ff1_kgh`, `ff2_kgh` | median \|ff - ff1 - ff2\| / ff at most 0.02 |
| clean configuration above 20,000 ft | `clean` | at least 0.95 of the samples |
| gear down above 20,000 ft | `gear_down` | at most 0.01 of the samples |
| speed brake above 30,000 ft (warn) | `speedbrake` | at most 0.02 of the samples |
| anti-ice share (warn) | `anti_ice` | at most 0.5 of the samples |

The tests of optional columns run only when the columns exist. `check` also reports the share of samples in flight (CAS >= 100 kt) that are clean, with speed brake, with gear down and with anti-ice on, overall and in three altitude bands (below 10,000 ft, 10,000 to 25,000 ft, above 25,000 ft). Above 25,000 ft nearly all samples are clean with gear up, the speed brake is rare and mostly in descent, and the gear is down only below about 10,000 ft. Other shares point to a wrong decoding, for example an inverted flag.

## Preparation

`neuralfuel prep` (Python: `neuralfuel.prep.prepare`) turns the 1 Hz samples into 10 s blocks and steady 300 s cruise windows, and splits the flights:

1. Derived quantities per sample: Mach, SAT, CAS, delta, true airspeed.
2. Savitzky-Golay derivatives (21 s, order 2) of pressure altitude and groundspeed inside each segment (same flight, 1 s steps), and the energy-rate inputs.
3. Sample filters: clean configuration; speed brake retracted and gear up and locked when the data has these flags; CAS >= 180 kt; fuel flow > 0; engine fuel-flow imbalance <= 15 %; finite values. Samples with engine anti-ice on are kept by default; the setting `exclude_anti_ice` removes them (`neuralfuel prep --set exclude_anti_ice=true`).
4. 10 s block means. A block is used only when all its 10 samples pass. The block carries `tail` and `anti_ice` (True if anti-ice is on in any of its samples) when the samples have them.
5. Phase labels and a split in time order by flight: the first 80 % of the flights for training, the next 10 % for validation (early stopping) and the last 10 % for the test.
6. Steady 300 s cruise windows above 25,000 ft: Mach SD < 0.003, altitude range < 150 ft, wings level (at least 95 % of the roll values below 5 deg, when roll is given), engines balanced (imbalance of the window means below 5 %, when the engine fuel flows are given), all samples passing the configuration filters and with fuel flow > 0, finite model inputs and target. Data without a steady window (for example climb and descent only) gives an empty window table with the same columns; `train` and `validate` work without windows.

The output folder holds `blocks.parquet`, `windows.parquet` and `prep.json`. `prep.json` holds counts and the preparation settings (`config`), no flight identifiers. The settings change the meaning of the blocks and windows (for example the block length and the phase thresholds), so `train`, `release`, `validate` and `local-fit` read them from `prep.json`: keep it with the blocks. The run record and the validation report keep them as `prep`. The blocks and windows hold flight identifiers for the local split and the trip-fuel test; they stay local. The settings are in `PrepConfig` (`src/neuralfuel/prep.py`) and can be changed with `--set KEY=VALUE`.

## Privacy rules

The raw data stays with the user. Only the model files, the run record and the validation report are shared.

- `profile` and `check` print summaries and test results, never rows.
- The validation report and the run record contain no flight identifiers, dates, positions or aircraft labels. `contribute` scans both for fields that look like an identifier, a date or a position and stops (exit code 1) if it finds one. It then keeps only the fields of the record formats with values of the declared shapes (see `contribute` in [cli.md](cli.md)): no free text enters the contribution.
- Never share flight data in any form: raw files, prepared blocks and windows (`work/prep`), samples, plots of single flights.
- Never write flight identifiers, aircraft registrations, dates, times, positions, airports, routes, the name of the data owner or of an airline, or data paths into tracked files, file names or commit messages.
- `local.toml`, `local/` and the private local offsets of `local-fit` stay on the computer of the user (see [federated.md](federated.md)).

The complete list for a pull request is in [`CONTRIBUTING.md`](../CONTRIBUTING.md), section "Never include".
