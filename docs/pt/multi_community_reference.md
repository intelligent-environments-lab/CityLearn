# Referencia Multi-Comunidade

`MultiCommunityEnv` orquestra varias instancias independentes de `CityLearnEnv`, uma por comunidade. A v1 e apenas sincronizacao e reporting de portfolio: nao ha fisica global, mercado entre comunidades, coordenacao DSO/TSO global ou mistura de buildings entre comunidades.

## Quickstart

```python
import numpy as np
from citylearn.multi_community import MultiCommunityEnv

env = MultiCommunityEnv(
    communities=[
        {
            "community_id": "community_a",
            "schema": "data/datasets/community_a/schema.json",
            "env_kwargs": {"interface": "entity", "episode_time_steps": 48},
            "weight": 1.0,
        },
        {
            "community_id": "community_b",
            "schema": "data/datasets/community_b/schema.json",
            "env_kwargs": {"interface": "entity", "episode_time_steps": 48},
            "weight": 2.0,
        },
    ],
    render_directory="outputs/runs",
    render_session_name="portfolio_001",
)

observations, info = env.reset(seed=0)
terminated = truncated = False

while not (terminated or truncated):
    actions = {
        community_id: {
            "tables": {
                table: np.zeros(space.shape, dtype="float32")
                for table, space in child.action_space["tables"].items()
            }
        }
        for community_id, child in env.envs.items()
    }
    observations, rewards, terminated, truncated, info = env.step(actions)

kpis = env.evaluate_v2(include_business_as_usual=False)
```

## Constructor

```python
MultiCommunityEnv(
    communities=[
        {
            "community_id": "community_a",
            "schema": "path/or/dataset/name/schema.json",
            "env_kwargs": {"interface": "entity", "episode_time_steps": 48},
            "weight": 1.0,
        }
    ],
    render_directory=None,
    render_session_name=None,
)
```

`community_id` tem de ser unico, nao vazio e seguro como componente de path relativo. Pode conter letras, numeros, `_`, `-` e `.`. `weight` tem default `1.0`, tem de ser finito e nao negativo, e pelo menos uma comunidade tem de ter peso positivo.

`env_kwargs` sao passados para o `CityLearnEnv` filho. O wrapper poe os exports de cada filho em `<render_session_name>/<community_id>` para separar ficheiros por comunidade.

## Reset E Step

`reset()` devolve observacoes por comunidade e info do wrapper:

```python
observations = {"community_a": child_observations, "community_b": child_observations}
info = {"communities": {...}, "time_step": 0}
```

`step(actions)` espera uma action payload por comunidade:

```python
actions = {"community_a": child_actions, "community_b": child_actions}
```

Devolve:

```python
observations_by_community,
rewards_by_community,
terminated,
truncated,
info
```

`info` inclui:

| Key | Significado |
|---|---|
| `community_rewards_scalar` | Soma finita do reward de cada comunidade. |
| `reward_total` | `sum(weight * community_reward_scalar)`. |
| `reward_mean_weighted` | `reward_total / sum(weights)`. |
| `terminated_by_community` | Flags de terminacao por comunidade. |
| `truncated_by_community` | Flags de truncation por comunidade. |

Se uma comunidade terminar ou truncar, o wrapper tambem termina ou trunca. Isto evita dessicronizacao temporal entre comunidades.

## Contrato Temporal

Todas as comunidades filhas tem de usar:

| Requisito | Motivo |
|---|---|
| Mesmo `seconds_per_time_step` | Mantem o tempo de cada step alinhado. |
| Mesmo episode length efetivo | Mantem batches de treino retangulares. |
| Mesmo `interface` | Mantem estavel o contrato action/observation. |
| Mesmo modo `central_agent` | Evita misturar semanticas centralizadas e descentralizadas. |

O wrapper valida isto no constructor e depois de cada reset.

## Demand Response

Demand response continua local a cada comunidade. Se o schema filho tiver `demand_response.enabled=true`, esse filho le o seu proprio `requests_file`, expoe o seu proprio pedido nas observations entity e calcula settlement/KPIs locais. A v1 do `MultiCommunityEnv` nao coordena pedidos DSO/TSO entre comunidades.

## KPIs

`evaluate_v2()` concatena as tabelas KPI dos filhos e adiciona a coluna `community_id`. Tambem cria linhas de portfolio:

```text
level = "portfolio"
name = "Portfolio"
community_id = "__portfolio__"
```

So KPIs district sao agregados:

| Sufixo KPI | Regra no portfolio |
|---|---|
| `_kwh`, `_eur`, `_kgco2`, `_count` | Soma entre comunidades. |
| `_ratio`, `_percent` | Media ponderada por `weight`. |
| `_kw`, `_c`, `_hours` e outros point metrics | Nao agregados na v1. |

Os nomes de portfolio substituem o prefixo `district_` por `portfolio_`, por exemplo `district_demand_response_revenue_total_eur` vira `portfolio_demand_response_revenue_total_eur`.

## Export

`export_final_kpis()` escreve os exports locais em subpastas e um CSV global na pasta da sessao do wrapper:

```text
<render_directory>/<render_session_name>/
  exported_kpis_multi_community.csv
  community_a/exported_kpis.csv
  community_b/exported_kpis.csv
```

O CSV global fica no formato longo de `evaluate_v2()`, com linhas locais e linhas portfolio.
