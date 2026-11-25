<!-- For a model contribution. For a code change, delete this text and describe the change, the tests and the updated pages in docs/. See CONTRIBUTING.md. -->

## Model contribution

- Aircraft type:
- Engine family:
- Base model (file in `models/<type code>/`):
- Base model weights hash (`init_weights_sha256` in `run.json`):
- Flights per split (train / validation / test):
- Number of aircraft (`n_aircraft` in `run.json`):

## Checklist

- [ ] The training continued from the base model above (`train --init`), and `init_weights_sha256` in `run.json` equals the hash of that model.
- [ ] The protection against forgetting was on (no `--no-protect`), and `run.json` has no `protection.warning`.
- [ ] The validation report `validation.json` of the test split is attached and was computed with the attached weights.
- [ ] The folder was written by `neuralfuel contribute` into a new folder and holds only `model.pt`, `model.onnx`, `run.json`, `validation.json` and `MANIFEST.json`. `dropped_fields` in `MANIFEST.json` is empty, or the pull request explains it.
- [ ] `neuralfuel check-model` with this `run.json` against the base model passes (the checks are explained in `docs/federated.md`). Output:
- [ ] The pull request contains no flight data, no flight identifiers, no aircraft registrations, no dates, no positions, no data paths, no names of the data owner or an airline, and no private local offsets.
