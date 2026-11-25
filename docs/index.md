# neuralfuel documentation

neuralfuel trains, validates and shares small neural-network models of aircraft fuel flow. A model gives the total fuel flow of one aircraft type from Mach, pressure altitude, mass, two energy-rate inputs and the static air temperature. It is trained on flight data with accurate airspeed, mass and fuel flow, with physical constraints. Quick access recorder (QAR) data is one example of suitable data; any source with these quantities at 1 Hz can be used through an adapter.

The training is federated. A user continues the training of a public shared model on local data and shares only the model and aggregate records. A maintainer combines the contributions of several users into the next shared model without flight data. The shared models are in [`models/`](../models/README.md).

## Who it is for

- Users of a shared model: fuel estimation, trajectory optimization and performance studies with the files in `models/`. No flight data is needed.
- Users with flight data: train a model, or improve a shared model with local data and contribute it.
- Maintainers: review contributions and publish the next shared model.

## Pages

| page | content |
|---|---|
| [quickstart.md](quickstart.md) | install, the synthetic quick start, run times |
| [model.md](model.md) | inputs and output, network and sizes, file names, temperature rule, loss and constraints, the rules of the method, limits |
| [data.md](data.md) | suitable data, the schema, adapters, the checks, privacy rules |
| [training.md](training.md) | prepare, train from scratch and continued, settings, run record, both sizes, `release` |
| [validation.md](validation.md) | every test of the validation report and how to read it |
| [federated.md](federated.md) | user and maintainer workflow, protection against forgetting, aggregation, quality check, comparison with reference models, private local correction |
| [use.md](use.md) | Python, ONNX, CasADi, inputs from ADS-B or Mode S data, the valid envelope |
| [cli.md](cli.md) | every command with its options and exit codes |
| [../examples/README.md](../examples/README.md) | runnable example scripts |

## Reading order

A user of a shared model:

1. [quickstart.md](quickstart.md), section "Install".
2. [use.md](use.md) and the examples 1 to 5 and 9 in [`examples/`](../examples/README.md).
3. [model.md](model.md) for the inputs and the limits, and the model card [`models/a320/README.md`](../models/a320/README.md).

A user who trains and contributes:

1. [quickstart.md](quickstart.md): the full cycle on synthetic data.
2. [data.md](data.md): the adapter for your data.
3. [training.md](training.md) and [validation.md](validation.md).
4. [federated.md](federated.md), section "Workflow for a user", and [`CONTRIBUTING.md`](../CONTRIBUTING.md).
5. [model.md](model.md) for the method.
6. An AI coding agent can do these steps with [`AGENTS.md`](../AGENTS.md).

A maintainer:

1. [federated.md](federated.md), all sections.
2. [validation.md](validation.md) and [cli.md](cli.md).
3. [`CONTRIBUTING.md`](../CONTRIBUTING.md), section "Review by the maintainers".
4. [model.md](model.md) and [training.md](training.md) for code changes. Run the tests with `uv run pytest` (see [quickstart.md](quickstart.md)).
