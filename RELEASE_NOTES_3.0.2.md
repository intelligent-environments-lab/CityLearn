# CityLearn 3.0.2 release notes

CityLearn 3.0.2 retires selected datasets from the official public registry.

## Changed

- Removed all 19 `ALADI` datasets.
- Removed the five `rec_2023_*` datasets.
- Removed `EC_Ermesinde` and its repository-bound integration test.
- Retained the simulator capabilities used by those datasets.

Dataset discovery against the matching release tag now returns 32 datasets.

## Validation

- Full local test suite: 457 passed.
- Critical lint checks: passed.
- Wheel and source archive build and metadata checks: passed.
- Dataset-retirement regression test: passed.

## Installation

```console
pip install citylearn==3.0.2
```

The distribution and Python import names remain `citylearn`.
