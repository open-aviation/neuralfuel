# Validation report

```sh
uv run neuralfuel validate work/prep --model work/run/model.pt --out work/run/validation.json
```

`validate` (Python: `neuralfuel.validate.validate`) writes one JSON file with aggregate numbers only. Unless stated otherwise, a test uses the blocks and steady windows of the evaluated splits (`--splits`, default: the test split). The settings of the tests are in `ValidationConfig` (`src/neuralfuel/validate.py`) and are copied into the report (`config`). `validate` takes the block length and the window length of the tests from the preparation settings in `prep.json` (copied into the report as `prep`), and the constraint settings from the model file (copied as `constraint_settings`; the defaults of `LossConfig` for a file without them). The split names are `train`, `val` and `test`; another name is bad input. The format of the report (the allowed keys) is `REPORT` in `src/neuralfuel/validate.py`. The validation reports of the shared models are in `models/a320/` and summarized in the model card [`models/a320/README.md`](../models/a320/README.md).

The report contains no flight identifiers, dates, positions or aircraft labels. `contribute` scans the report and the run record for such fields and stops if it finds one.

## Sections

| key | content |
|---|---|
| `model` | weights hash, number of weights, inputs, temperature rule and exponent, size, aircraft settings, file name |
| `evaluated_splits` | the splits of the blocks and windows below (`contribute` needs `["test"]`) |
| `accuracy` | accuracy per phase |
| `trip_fuel` | trip fuel error per flight |
| `steady_windows` | accuracy on the steady windows |
| `mach_sensitivity` | Mach sensitivity in cells of W/delta by Mach |
| `max_specific_range` | Mach of maximum specific air range per W/delta |
| `temperature_test` | best-range Mach against ISA deviation |
| `constraints` | violation rates of P1 to P3 and P6, and of P7 for a model with P7 settings |
| `roughness` | mean Hessian norm of ln F |
| `pair_test` | speed-change pairs of steady windows |
| `smoothness` | block-to-block variation and variation inside steady windows |
| `noise_sensitivity` | change of the prediction for input noise |
| `envelope` | the valid envelope and the share of evaluated blocks outside it |
| `coverage` | ranges and histograms of the evaluated and the training data |
| `surveillance` | optional: accuracy with Mode S quality inputs |
| `config`, `constraint_settings`, `prep` | the settings of the tests, of the constraints and of the preparation |
| `config` | the settings of the tests |

## Tests

**Accuracy** (`accuracy`). For all blocks, climb, level, descent and cruise blocks: the number of blocks `n`, the mean absolute percentage error `MAPE_pct`, the relative root mean square error `rRMSE_pct` and the mean bias `bias_pct` of the predicted fuel flow, in per cent of the measured value. The relative error of a block is predicted / measured - 1.

**Trip fuel** (`trip_fuel`). The fuel of each flight, summed over its kept blocks, predicted against measured: the median absolute error, the mean bias and the median bias, in per cent, and the number of flights. The kept blocks cover only clean flight above 180 kt CAS, so this is not the fuel of the full trip.

**Steady windows** (`steady_windows`). MAPE, rRMSE and bias on the steady 300 s cruise windows. Each window is the mean of 300 s of steady cruise.

**Mach sensitivity** (`mach_sensitivity`). s = d ln F / dM at eps = 0 (steady level flight), in cells of W/delta by Mach (see "Grids" below). The values are the number of supported cells, the mean s over them, the mean per W/delta and every cell. `cells_s_above_1_over_mach` counts the cells with s > 1 / M, where the specific air range decreases with Mach. The shared A320 models have a mean of about 1.2 over 20 cells (model card).

**Mach of maximum specific air range** (`max_specific_range`). For each W/delta, the Mach of maximum M / F at fixed temperature and eps = 0, at the conditions of its best-supported cell, with the mass set to the nominal W/delta. `at_upper_bound` is True when the maximum is at the MMO, the upper end of the search. `n_local_max` counts the local maxima on the search grid.

**Temperature test** (`temperature_test`). The Mach of maximum specific air range at one altitude and mass for ISA deviations of -20, -10, 0, +10 and +20 K, and the shift between -20 K and +20 K (`shift_mach_max_sr`). With the fixed temperature rule the shift is 0 by construction (see [model.md](model.md)).

**Constraints** (`constraints`). The violation rates (%) of P1, P2 and P3 on a regular grid in the envelope of the type, inside and outside the training data support, and of P6 at the MMO. The grid has 24 x 20 x 11 x 17 x 8 cells in Mach, altitude, mass, energy rate and ISA deviation (`grid_n`), limited to CAS from 180 kt to the VMO. A grid state is inside the support when its 4D cell (Mach, altitude, mass, energy rate) holds at least 20 training blocks (`grid_min_count`) and its ISA deviation lies inside the 1 % to 99 % range of the training blocks. A state violates P1 when f_MM + f_M² < -0.1 (only at powered states), P2 when f_ln m < -0.01 (only at powered states), P3 when f_eps < -0.05 (tolerances `tol_*`). The rates of P1 and P2 are shares of the powered grid states, eps >= `eps_powered` (-0.005) of the constraint settings of the model. P6 uses 10 altitudes from 30,000 ft to the maximum altitude, 11 masses and 8 ISA deviations at the MMO; it reports the share with M f_M < 1.1 and the smallest M f_M. For a model with P7 settings (`constraints.drag_rise` of the model file), `drag_rise` reports P7 on a grid of 13 Mach values from the lower Mach of P7 to the MMO, 20 altitudes, 11 masses and 8 ISA deviations in level flight, limited to the CAS range and to the lift-coefficient bound of P7: the share with f_M < g - 0.05, the smallest margin f_M - g, with g the drag slope of the stored table grid, and whether the drag-slope table rebuilt from the installed aerox has the stored SHA-256 (`table_sha256_match`). A model without P7 settings has no `drag_rise` entry and no P7 test. Rates outside the support show how the model extrapolates.

**Roughness** (`roughness`). The mean Frobenius norm of the Hessian of f = ln F on the same grid, inside and outside the support. The Hessian is taken in standardized state coordinates (Mach, ln delta, ln mass, eps, ln theta, each divided by its SD in the training blocks), so each entry is the change of ln F for one SD of two inputs. Lower is smoother.

**Speed-change pair test** (`pair_test`). Pairs of steady windows of one flight with at most 200 ft altitude difference and at least 0.015 Mach difference. For each pair, the predicted change of ln F is compared with the measured change: the slope through the origin, the correlation and the RMSE of the difference. A slope of 1 means the model has the measured speed dependence. The test is reported for the test split (`test`) and for the test and validation splits together (`test_and_val`), each with the number of pairs and flights and the field `flights_used`. Few pairs make the result uncertain.

**Smoothness at the block rate** (`smoothness`). The RMS first difference of the predicted and of the measured fuel flow between consecutive 10 s blocks of a segment, per phase, and their ratio; the median ratio of the predicted to the measured SD in 5-minute runs of cruise blocks (30 blocks); and, for every steady window with at least 25 complete blocks inside it, the SD of the predicted fuel flow of these blocks divided by the measured SD (median over the windows). A ratio below 1 means the prediction varies less than the measurement.

**Input-noise sensitivity** (`noise_sensitivity`). The RMS relative change (%) of the predicted fuel flow when Gaussian noise is added to one input at a time and to all inputs together (`joint`). The noise SD is 0.002 for Mach, 25 ft for pressure altitude (through ln delta), 0.5 % for mass, 5e-4 for the energy rate and 0.5 K for the static temperature. The energy-rate noise is split into independent noise of SD 5e-4 / sqrt(2) on sin_gamma and on acc_in, so that their sum has SD 5e-4. The random numbers have a fixed seed (`noise_seed`). The values show which input error matters most for a user of the model.

**Valid envelope** (`envelope`). The valid envelope of the model (see [use.md](use.md)), its source (the model file, or the training split of the validated data for a file without one) and the share of the evaluated blocks outside it, per phase (`evaluated_outside_pct`). The ranges are 1 % to 99 % quantiles, so several per cent of the training blocks themselves are outside.

**Coverage** (`coverage`). For the evaluated data and for the training data: the numbers of blocks, flights, windows, blocks per phase and cruise blocks, and the range, quantiles (0, 1, 50, 99, 100 %) and histogram counts of Mach, altitude, mass, ISA deviation and energy rate.

**Surveillance-input test** (`surveillance`, optional: `--surveillance`, or `samples=` in Python). The 1 Hz samples of the evaluated flights are rebuilt at Mode S quality: Mach rounded to 0.004, pressure altitude rounded to 25 ft, groundspeed rounded to 1 kt, temperature unchanged. The energy inputs come from the rounded altitude and groundspeed through the Savitzky-Golay filter of `surveillance_inputs`, and the inputs are averaged over the same 10 s blocks. The test reports the accuracy per phase with these inputs and with the recorded inputs on the same blocks. The schema has no wind, so an airspeed from groundspeed and wind (ADS-B) is not tested. `--surveillance` reads the 1 Hz samples through the adapter (`--data`, `--adapter`, `NEURALFUEL_DATA` or `local.toml`).

## Grids of the Mach and temperature tests

The grids come from the steady windows of the validated data (all splits, because they only set representative flight conditions) and from the aircraft settings, not from fixed values of one type:

- W/delta values: the 10 %, 35 %, 65 % and 90 % quantiles of the W/delta of the steady windows (`wd_quantiles`), rounded to 0.1 t;
- Mach cells: centres at multiples of 0.01 from the 1 % to the 99 % quantile of the window Mach, not above the MMO;
- a cell holds the windows within 0.01 Mach and within 4 % of its W/delta (`cell_dmach`, `cell_dwd_rel`). It is supported if it holds at least 30 windows (`cell_min_windows`); the Mach sensitivity is computed at the mean altitude, mass and temperature of the windows of each supported cell;
- the Mach of maximum specific air range M / F (fixed temperature, eps = 0) is searched from MMO - 0.22 to the MMO in steps of 0.001 (`sr_mach_span`, `sr_mach_step`), at the conditions of the best-supported cell of each W/delta, with the mass set to the nominal W/delta;
- the temperature test point is the median pressure altitude and the median mass of the steady windows (of the cruise blocks if there are no windows).

The support of the constraint grid and the roughness scale use the training blocks if there are any, else all blocks.

## How to read a report

1. Check `model.weights_sha256` against the run record: the report belongs to these weights.
2. Read `accuracy` per phase and `steady_windows`. Compare a contribution with the validation of the shared model on the same data (`work/shared_validation.json`, see [federated.md](federated.md)).
3. Check `envelope.evaluated_outside_pct`: a high share means the test data lie outside the region of the training data.
4. Read `mach_sensitivity`, `max_specific_range` and `pair_test` together: they show the speed dependence, which the trip fuel alone does not show.
5. Check `constraints` outside the support and `roughness`: high values mean the model extrapolates badly.
6. Read `coverage` to see which part of the flight envelope the numbers describe.

The numbers of a model trained on synthetic data say nothing about real aircraft.
