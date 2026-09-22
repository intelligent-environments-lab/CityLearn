# Moving an existing project to CityLearn v3

Both the distribution and import name are `citylearn`. Install it from PyPI, or
use the [developer guide](developer_guide.md) for a repository checkout.

## Keep the controller loop explicit

```python
observations, info = env.reset(seed=0)
observations, rewards, terminated, truncated, info = env.step(actions)
```

Stop at `terminated or truncated`. `info` is the last return value, not the third.
The flat interface retains one observation/action vector per building, or one
combined vector with `central_agent=True`. Read action and observation order
from the environment rather than hard-coding positions.

## Choose the interface and timestep

Start fixed-membership controllers with `interface="flat"`. Use
`interface="entity"` for identified entity tables, relational edges and dynamic
membership. Entity observations are an input representation; a GNN or Transformer
policy must still be supplied by the user.

Match `seconds_per_time_step` to the data cadence. Energy columns use `kWh/step`;
power limits use `kW`. See the [unit contract](data_unit_contract.md).

## Choose the evaluation API

`evaluate()` retains legacy cost-function names. `evaluate_v2()` returns the
structured KPI names used by the current exports and scorecard. The `v2` in the
method name denotes the KPI interface, not CityLearn version 2.

The [scorecard](scorecard.md) distinguishes the conventional counterfactual
baseline from the native business-as-usual controller and explains service,
target accuracy and requested/applied electrical-limit indicators.

## Resolve public and local data separately

Public dataset names are resolved against the installed package's release tag.
Use `DataSet().get_dataset_names()` to see that version's catalogue (import
`DataSet` from `citylearn.data`). Supply your own
local schema and files for data that are not in the public catalogue; a
repository-relative path is not created by `pip install`.

The [release history](releases.md) retains version-specific changes and the
development-line history.
