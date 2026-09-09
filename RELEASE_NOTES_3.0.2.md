# CityLearn 3.0.2 release notes

CityLearn 3.0.2 retires selected datasets from the official public registry.

## Changed

- Removed all 19 `ALADI` datasets.
- Removed the five `rec_2023_*` datasets.
- Removed `EC_Ermesinde` and its repository-bound integration test.
- Retained the simulator capabilities used by those datasets.

Dataset discovery against the matching release tag now returns 32 datasets.

## Installation

```console
pip install citylearn==3.0.2
```

The distribution and Python import names remain `citylearn`.
