# CityLearn

`citylearn` is an open-source simulation framework for energy-community studies
with reinforcement learning. Version 3 integrates the extended development line
maintained by the Soft-CPS Research Group.

It is a fork-based project used for research and experimentation on:

- electric vehicles (EVs) and chargers,
- stationary batteries (BESS),
- photovoltaic generation (PV),
- electrical-service constraints (single/three-phase),
- local community market settlement and KPI analysis.
- entity-mode RL observation contracts with derived forecasts, physical deadlines,
  feasible action capacity and requested/limited/applied action feedback.
- dynamic member and asset topology with repeatable multi-episode reset semantics.

Current source release: `3.0.0`.

## Project Positioning

This is the official CityLearn package, including the Soft-CPS REC extensions.

## Installation

```bash
pip install citylearn
```

## Python Usage

For compatibility with existing ecosystems, the Python module path currently remains:

```python
from citylearn.citylearn import CityLearnEnv
```

## Source and Documentation

- Source: https://github.com/citylearn-project/CityLearn
- Documentation: https://www.citylearn.net/
