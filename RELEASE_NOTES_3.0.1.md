# CityLearn 3.0.1 release notes

CityLearn 3.0.1 is the installation and dataset-registry follow-up to the
Version 3 launch.

## Fixed

- The default dataset source now resolves from the official
  `citylearn-project/CityLearn` repository and the matching `v3.0.1` tag.
- The PyPI publish job installs `twine` before uploading the built artifacts.

## Installation

```console
pip install citylearn==3.0.1
```

The distribution and Python import names remain `citylearn`.
