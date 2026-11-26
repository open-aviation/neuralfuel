# A320 CFM56-5B models

This folder holds the shared neuralfuel models of the A320 with CFM56-5B engines. All numbers below come from the run record (`.run.json`) and the validation report (`.validation.json`) of each model, unless stated otherwise. How to use the models: [`docs/use.md`](../../docs/use.md) and [`examples/`](../../examples/README.md). The inputs, the method and the tests are explained in [`docs/model.md`](../../docs/model.md) and [`docs/validation.md`](../../docs/validation.md).

## Files

| file | content | load |
|---|---|---|
| `nf-a320-cfm56_5b.pt` | standard size, PyTorch weights with the standardization, the temperature rule, the aircraft settings and the valid envelope | `neuralfuel.model.load("models/a320/nf-a320-cfm56_5b.pt")` |
| `nf-a320-cfm56_5b.onnx` | the same network as ONNX; the aircraft settings and the envelope are in the metadata | `onnxruntime.InferenceSession("models/a320/nf-a320-cfm56_5b.onnx")`, or CasADi: `neuralfuel.inference.casadi_function("models/a320/nf-a320-cfm56_5b.onnx")` |
| `nf-a320-cfm56_5b.run.json` | run record: training settings, data counts, envelope, weights hash | JSON |
| `nf-a320-cfm56_5b.validation.json` | validation report on the test split | JSON |
| `nf-a320-cfm56_5b-xs.*` | the same four files for the `xs` size | as above |

```python
from neuralfuel.model import load
from neuralfuel.inference import fuel_flow

model = load("models/a320/nf-a320-cfm56_5b.pt")
fuel_flow(model, mach=0.78, alt_ft=37000, mass_kg=62000, sat_k=225.0)  # kg/h
```

The ONNX file takes the six inputs below. At 37,000 ft pressure altitude the ISA pressure ratio delta is 0.2138 (`neuralfuel.atmos.isa_delta(37000)`); theta is SAT / 288.15 K.

```python
import numpy as np
import onnxruntime as ort

from neuralfuel import atmos

sess = ort.InferenceSession("models/a320/nf-a320-cfm56_5b.onnx")
delta = atmos.isa_delta(37000.0)  # 0.2138
x = np.array(
    [[0.78, np.log(delta), np.log(62000), 0.0, 0.0, np.log(225.0 / atmos.T0)]],
    np.float32,
)
ln_ff = sess.run(None, {"x": x})[0]  # ln of fuel flow in kg/h, shape (1, 1)
np.exp(ln_ff)  # kg/h
```

```python
import casadi as ca

from neuralfuel.inference import casadi_function

f = casadi_function("models/a320/nf-a320-cfm56_5b.onnx")  # needs uv sync --group casadi
xs = ca.MX.sym("x", 1, 6)
fuel = ca.Function("fuel", [xs], [ca.exp(f(xs))])  # kg/h
float(fuel(x))  # at the state x above
```

The SHA-256 of the weights (`neuralfuel.model.state_hash`) is in `weights_sha256` of the run record and in `model.weights_sha256` of the validation report:

| model | weights SHA-256 |
|---|---|
| `nf-a320-cfm56_5b` | `195512e0bd5ec50b2aa1dc92a65dfdc222981f24baa5037d91245665c5ad202e` |
| `nf-a320-cfm56_5b-xs` | `c55a238ac31bae26c669bc58abb4e4159b4df5f1eee55accebe0f573b66ee0ce` |

## Model

| | `nf-a320-cfm56_5b` | `nf-a320-cfm56_5b-xs` |
|---|---|---|
| aircraft type | A320 | A320 |
| engine family | CFM56-5B | CFM56-5B |
| network | 3 hidden layers of 64 softplus units | 3 hidden layers of 32 softplus units |
| weights | 8,769 | 2,337 |
| temperature rule | fixed, exponent 0.5 | fixed, exponent 0.5 |
| training | from scratch, 19,500 update steps, seeds 0, 1, 2; the seed with the lowest validation loss is kept: seed 2 (validation loss 0.001844) | same: seed 0 (validation loss 0.001944) |

Inputs, in this order (array `x` of shape (n, 6)):

| input | definition | unit |
|---|---|---|
| `mach` | Mach number | - |
| `ln_delta` | ln of the ISA pressure ratio at the pressure altitude | - |
| `ln_m` | ln of the mass | mass in kg |
| `sin_gamma` | geometric vertical speed / true airspeed | - |
| `acc_in` | (groundspeed / true airspeed) x (d groundspeed / dt) / g | - |
| `ln_theta` | ln of SAT / 288.15 K | - |

Output: `ln_fuel_flow`, ln of the total fuel flow of both engines in kg/h.

Physical constraints of the training (see "Loss and physical constraints" in [`docs/model.md`](../../docs/model.md)), with the violation rates of the validation report (%). P1 to P3: on the regular constraint grid of the envelope of the type, all states and inside the training data support; P1 and P2 only at powered states (energy rate eps >= -0.005). P6 and P7: on their own grids (P6 at the MMO; P7 in level flight, a violation is d ln F / dM below d ln D / dM - 0.05):

| constraint | weight | condition | standard: all / inside | xs: all / inside |
|---|---|---|---|---|
| P1 | 0.001 | fuel flow convex in Mach at powered states | 1.56 / 0.00 | 0.00 / 0.00 |
| P2 | 0.001 | fuel flow increases with mass at powered states | 5.80 / 0.00 | 0.01 / 0.00 |
| P3 | 0.001 | fuel flow increases with energy rate | 12.68 / 0.00 | 0.14 / 0.00 |
| P6 | 0.001 | M d ln F / dM >= 1.1 at the MMO, 30,000 ft to the ceiling | 0.00 | 0.00 |
| P7 | 0.01 | in level flight from Mach 0.70 to the MMO, d ln F / dM >= d ln D / dM of the aerox drag model (aerox 0.3, A320) at fixed W/delta, lift coefficient up to 1.1 x 0.678 | 1.35 | 1.80 |

P2 is tested and trained only at powered states. The required thrust is T = D + W eps, so below a limit descent angle (for the A320 at eps of about -0.03 to -0.05) it decreases with mass until the engines reach idle, and the fuel flow measured in descent can decrease with mass. The P2 violations of the earlier models inside the data support were all in this region.

P7 follows from F = TSFC x D in steady level flight: TSFC does not decrease with Mach, so the Mach slope of fuel flow is at least that of drag. The P7 settings, the aerox version and the SHA-256 of the drag-slope table are in the `.pt` file (`constraints.drag_rise`), in the run record (`drag_rise`) and in the validation report (`constraint_settings.drag_rise`). The smallest margin d ln F / dM - d ln D / dM on the P7 grid is -0.35 (standard) and -0.55 (xs); the smallest M d ln F / dM at the MMO is 1.61 and 1.44.

Aircraft settings: MMO 0.82, VMO 350 kt, mass 42,600 to 78,000 kg, altitude 10,000 to 39,800 ft, wing area 122.0 m². These are the aerox values for the A320 with CFM56-5B engines. They are in the `.pt` file (entry `aircraft`), in the ONNX metadata (`neuralfuel_aircraft`), in the run record (`aircraft`) and in the validation report (`model.aircraft`). The wing area is also in every copy of the envelope (`wing_area_m2`).

## Training data

- 1 Hz flight-recorder data of one A320 dataset with two aircraft.
- Flights split in time order: 875 for training, 109 for validation (early stopping), 110 for the test.
- Clean configuration only: flaps and slats retracted, speed brake retracted, landing gear up and locked, CAS of 180 kt or more. The samples were decoded from the raw recorder parameters with an adapter that is not part of the package: `clean` is True when the flap code is 0 and the CONF 1 bit (slats extended with flaps retracted) is 0, `speedbrake` is True when spoiler 2 is deflected more than 5 deg on both sides, and `gear_down` is True when the landing gear is not up and locked. `prep` removed the speed-brake and gear-down samples.
- The packaged example adapter `example_a320` does not apply these filters: its flap code reads 0 in CONF 1 and it has no speed-brake or gear flags (see [`docs/data.md`](../../docs/data.md), section "Adapters").
- 10 s blocks: 520,973 for training, 67,273 for validation, 69,933 for the test. Steady 300 s cruise windows: 5,866 for training, 874 for validation, 750 for the test.

The data stays with its owner. Only the model files and the aggregate records are shared.

## Valid envelope

The envelope is the same for both models. Each range is the 1 % to 99 % quantile of the training blocks of the phase. Phases: climb above +300 ft/min and descent below -300 ft/min geometric vertical speed, else level; cruise is level flight above 25,000 ft.

| variable | climb | level | descent |
|---|---|---|---|
| blocks | 106,995 | 338,505 | 75,473 |
| Mach | 0.339 to 0.780 | 0.445 to 0.786 | 0.322 to 0.779 |
| pressure altitude (ft) | 2,773 to 37,647 | 11,059 to 39,014 | 3,063 to 37,471 |
| mass (kg) | 52,189 to 71,160 | 51,982 to 68,807 | 50,294 to 65,590 |
| W/delta (t) | 65.9 to 300.9 | 87.6 to 317.4 | 61.7 to 280.0 |
| ISA deviation (K) | +6.8 to +18.7 | +2.8 to +16.6 | +7.3 to +18.5 |
| sin_gamma | 0.0084 to 0.1368 | -0.0023 to 0.0027 | -0.0947 to -0.0113 |
| acc_in | -0.0288 to 0.0744 | -0.0119 to 0.0119 | -0.0442 to 0.0313 |

Cruise maxima from the 5,866 training windows: W/delta 339.6 t, lift coefficient 0.678.

`predict(model, x, outside=True)` returns a flag for states outside the envelope. Of the test blocks, 8.0 % are outside (climb 6.8 %, level 8.6 %, descent 6.6 %), because the ranges are 1 % to 99 % quantiles.

## Accuracy on the test split

Fuel flow of the 10 s blocks, in per cent of the measured value:

| phase | blocks | MAPE standard | rRMSE standard | bias standard | MAPE xs | rRMSE xs | bias xs |
|---|---|---|---|---|---|---|---|
| all | 69,933 | 2.42 | 4.39 | +0.26 | 2.52 | 4.50 | +0.06 |
| climb | 14,678 | 1.26 | 2.02 | +0.17 | 1.60 | 2.40 | -0.24 |
| level | 45,967 | 1.83 | 2.71 | +0.01 | 1.84 | 2.71 | -0.14 |
| descent | 9,288 | 7.18 | 10.12 | +1.62 | 7.34 | 10.34 | +1.48 |
| cruise | 44,689 | 1.76 | 2.44 | +0.00 | 1.77 | 2.46 | -0.16 |

| test | standard | xs |
|---|---|---|
| trip fuel, 110 flights: median absolute error | 0.37 % | 0.40 % |
| trip fuel: mean bias / median bias | +0.07 % / +0.07 % | -0.17 % / -0.21 % |
| steady windows, 750: MAPE / rRMSE / bias | 0.77 % / 0.96 % / +0.10 % | 0.74 % / 0.93 % / -0.05 % |

With surveillance-quality inputs (Mach rounded to 0.004, pressure altitude to 25 ft, groundspeed to 1 kt), the MAPE of all test blocks is 2.53 % (standard) and 2.62 % (xs).

## Cruise Mach sensitivity

s = d ln F / dM at zero energy rate, mean over the cells of W/delta by Mach with at least 30 steady windows (20 cells), from the validation report. W/delta is the mass divided by the ISA pressure ratio, in tonnes. The comparison with OpenAP below uses another definition: the median of s over its grid of cruise states. The Mach of maximum specific air range is computed at the conditions of the best-supported cell of each W/delta.

| W/delta (t) | s standard | s xs | best-range Mach standard | best-range Mach xs |
|---|---|---|---|---|
| 239.9 | 1.461 | 1.671 | 0.745 | 0.728 |
| 270.9 | 1.353 | 1.499 | 0.767 | 0.757 |
| 288.5 | 1.236 | 1.397 | 0.774 | 0.765 |
| 303.2 | 1.242 | 1.379 | 0.778 | 0.769 |
| mean of supported cells | 1.334 | 1.501 | | |

Speed-change pairs (steady windows of one flight, at most 200 ft apart, at least 0.015 Mach apart): on the test and validation splits, 93 pairs of 11 flights, the slope of the predicted against the measured change of ln F is 0.924 (standard) and 0.965 (xs), with a correlation of 0.971 and 0.991. The test split alone has 37 pairs of 6 flights (slope 0.840 and 0.875).

## Comparison with OpenAP

Output of `uv run --group reference neuralfuel compare --model models/a320/<name>.pt` (OpenAP A320). The ratio is model / OpenAP fuel flow on a grid of states inside the envelope of the model. OpenAP sees the pressure, the static temperature, the Mach and the flight-path angle of each state (see "Comparison with reference models" in [`docs/federated.md`](../../docs/federated.md)). The grid uses the ISA deviation in the middle of each phase range of the envelope (+9.7 K in cruise).

| phase | states | median standard | 5 % to 95 % standard | median xs | 5 % to 95 % xs |
|---|---|---|---|---|---|
| cruise | 1,138 | 0.926 | 0.881 to 1.006 | 0.923 | 0.883 to 1.007 |
| climb | 1,782 | 0.820 | 0.647 to 1.208 | 0.881 | 0.667 to 1.206 |
| descent | 1,826 | 1.153 | 0.618 to 1.629 | 1.121 | 0.701 to 1.588 |

The cruise grid includes the states on the bounds of the envelope (the lowest and the highest mass of level flight).

Cruise Mach sensitivity d ln F / dM, median on the grid: 1.319 (standard) and 1.383 (xs), OpenAP 0.781. The median of the pointwise ratios model / OpenAP is 1.659 (standard) and 1.798 (xs).

Mach of maximum specific air range at ISA +9.7 K (search from MMO - 0.22 to the MMO; `*`: at the upper bound of the search, the MMO; `_`: at the lower bound of the search; `!`: outside the valid envelope of the model, which ends at Mach 0.786 in level flight). A marked Mach is a limit of the search or an extrapolation, not an optimum inside the envelope:

| altitude (ft) | mass (kg) | standard | xs | OpenAP |
|---|---|---|---|---|
| 26,000 | 52,000 | 0.600_ | 0.600_ | 0.656 |
| 26,000 | 60,400 | 0.626 | 0.634 | 0.710 |
| 26,000 | 68,800 | 0.668 | 0.664 | 0.762 |
| 30,000 | 52,000 | 0.634 | 0.646 | 0.716 |
| 30,000 | 60,400 | 0.692 | 0.684 | 0.776 |
| 30,000 | 68,800 | 0.726 | 0.708 | 0.820*! |
| 36,000 | 52,000 | 0.742 | 0.732 | 0.820*! |
| 36,000 | 60,400 | 0.766 | 0.754 | 0.820*! |
| 36,000 | 68,800 | 0.774 | 0.758 | 0.820*! |

Best-range Mach of the models in level flight at ISA +10 K (search MMO - 0.22 to the MMO, step 0.001; with the fixed temperature rule the result is the same at ISA). These values are not in the validation reports; they come from the model files. `!`: the state is outside the valid envelope (`neuralfuel.support.in_envelope`). On a grid of cruise states from 29,000 to 39,800 ft and 42.6 to 78 t (1,971 states with a lift coefficient at Mach 0.78 not above 1.1 x 0.678), no best-range Mach is above 0.80 for either model.

| mass (t) | altitude (ft) | standard | xs |
|---|---|---|---|
| 56 | 33,000 | 0.722 | 0.708 |
| 56 | 35,000 | 0.747 | 0.734 |
| 56 | 37,000 | 0.763 | 0.758 |
| 56 | 39,000 | 0.775 | 0.779 |
| 56 | 39,800 | 0.779! | 0.787! |
| 62 | 33,000 | 0.743 | 0.726 |
| 62 | 35,000 | 0.762 | 0.747 |
| 62 | 37,000 | 0.774 | 0.766 |
| 62 | 39,000 | 0.783! | 0.783! |
| 62 | 39,800 | 0.786! | 0.789! |
| 68 | 33,000 | 0.755 | 0.735 |
| 68 | 35,000 | 0.769 | 0.751 |
| 68 | 37,000 | 0.779! | 0.766! |
| 68 | 39,000 | 0.786! | 0.779! |
| 68 | 39,800 | 0.788! | 0.784! |

OpenAP is a reference, not the truth. A BADA 4 comparison needs a licensed local BADA copy and is not included here.

## Limits

- Clean configuration only: flaps and slats retracted, speed brake retracted, landing gear up. The models do not cover take-off, approach and landing configurations.
- Descent error is about 7 % (MAPE), much higher than in climb and level flight.
- The temperature exponent 0.5 is provisional. It is fixed, not learned. The temperature test shows no change of the best-range Mach between ISA -20 K and +20 K by construction, and the valid envelope covers only ISA deviations of about +3 K to +19 K.
- The models come from one dataset with two aircraft. Other aircraft, engines and operations can differ.
- The high-Mach behaviour above the data Mach limit (0.786 in level flight) comes from the aerox drag model and the assumption that TSFC does not decrease with Mach (constraint P7), not from data. The same holds for the condition P6 at the MMO.
- The models are not valid outside the stored envelope. Check states with `predict(model, x, outside=True)` or `neuralfuel.support.in_envelope`.
