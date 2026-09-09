# Simulator Features

Esta pagina resume o que o simulador tem de bom, incluindo funcionalidades que nao sao obvias quando se olha so para o README.

## Core

| Feature | O que oferece |
|---|---|
| Gymnasium environment | API `reset/step`, action/observation spaces e compatibilidade RL. |
| Multi-building district | Simula comunidade de buildings com DERs. |
| Multi-agent ou central-agent | Pode devolver um vetor por building ou um vetor centralizado. |
| HVAC e thermal storage | Cooling, heating e DHW com storage termico. |
| BESS | Storage eletrico com SOC, eficiencia, power curves e degradacao. |
| PV | Geracao em modo perfil por kW ou absoluto `kWh/step`. |
| EVs e chargers | Schedules por charger, EV connected/incoming, V2G e eficiencia. |
| Deferrable appliances | Ciclos normalizados, janelas de flexibilidade e KPIs de servico. |
| Demand response | Pedidos DSO/TSO orientados por dataset com observacoes entity, settlement e KPIs. |
| Multi-comunidade | Orquestra varias comunidades `CityLearnEnv` sincronizadas e reporta KPIs de portfolio. |
| Eventos de robustez | Degradacao opcional e orientada por dataset para observations, forecasts, actions e disponibilidade logica de assets. |

## Fisica e Unidades

| Feature | O que resolve |
|---|---|
| Sub-hourly support | 1h, 15min, 5min, 1min, 15s e outras resolucoes via `seconds_per_time_step`. |
| Contrato kWh/step vs kW | Datasets energeticos em `kWh/step`; limites e potencias em `kW`. |
| Conversoes por step | Acoes de potencia sao convertidas para energia com a duracao do step. |
| PV absolute | Permite usar dados reais de geracao diretamente. |
| Physics invariant checks | Checks opcionais de sanidade durante runtime. |
| Audit scripts | `audit_physics.py` e `audit_entity_contract.py --strict`. |

## Entity Interface

| Feature | Porque importa |
|---|---|
| Tabelas por entidade | `district`, `building`, `charger`, `ev`, `storage`, `pv`, `deferrable_appliance`. |
| Edges relacionais | Pronto para GNN/GraphRL. |
| IDs estaveis | Permite memoria por entidade e topology dynamic. |
| Bundles | Permite controlar custo/riqueza das observacoes. |
| Bundle demand response | Adiciona pedido ativo, baseline, precos e delivery/shortfall anterior a tabela district. |
| Bundle robustez | Adiciona estado de robustez ativo e contadores de corrupcao do step anterior a tabela district. |
| Specs machine-readable | `env.entity_specs` descreve ids, features, unidades e bundles. |
| Copias nas observacoes | Evita mutacao acidental do estado interno pelo agente. |

## Dynamic Topology

| Feature | O que faz |
|---|---|
| `add_member`/`remove_member` | Buildings podem entrar/sair da comunidade. |
| `add_asset`/`remove_asset` | Chargers, PV, BESS e deferrables podem mudar durante o episodio. |
| Refresh de spaces/specs | Action/observation spaces acompanham topology version. |
| Lifecycle | `born_at`, `removed_at`, `active` em `entity_specs`. |
| Remocao segura de deferrables | Cancela ciclos futuros sem deixar consumo pendente. |

## Three Phase e Electrical Service

| Feature | O que mede/controla |
|---|---|
| Phase connection | Chargers e storage podem ter `L1`, `L2`, `L3`, `all_phases`. |
| Headroom | Observacoes de headroom import/export. |
| Phase power | Potencia atual por fase no bundle core. |
| Violations | KPIs de violacao total e contagem. |
| Phase imbalance | KPI de desequilibrio medio. |
| Phase peaks | Picos import/export por fase. |

## Community Market e KPIs

| Feature | O que oferece |
|---|---|
| Local settlement | Matching local entre exportadores/importadores. |
| Weighted imports | Reparticao por pesos de membros. |
| Savings | Compara settled cost contra counterfactual. |
| Self-consumption community | Mede share local/import. |
| KPI v2 naming tree | Nomes estruturados por level/family/subfamily/metric/unit. |
| Golden tests | Testes com valores calculados a mao para KPIs criticos. |

## Multi-Comunidade

| Feature | O que oferece |
|---|---|
| `MultiCommunityEnv` | Wrapper publico em `citylearn.multi_community` para varias comunidades independentes. |
| Step sincronizado | Todas as comunidades tem de partilhar `seconds_per_time_step`, episode length, interface e modo `central_agent`. |
| DR independente | Cada comunidade mantem os seus pedidos, observations, settlement e KPIs de demand response. |
| KPIs portfolio | `evaluate_v2()` adiciona linhas `level="portfolio"` com somas e ratios ponderados. |
| Layout de export | KPIs locais em subpastas por comunidade e CSV global `exported_kpis_multi_community.csv`. |

## Robustez

| Feature | O que oferece |
|---|---|
| Eventos por dataset | Ficheiros CSV/Parquet configuram perturbacoes deterministicas. |
| Switches modulares | Observations, forecasts, actions e assets podem ser ligados separadamente. |
| Separacao do estado real | Corrupcao de observation/forecast nao altera rewards, settlement DR nem KPIs fisicos. |
| Canal de acao imperfeito | Dropout, noise, bias, stuck, delay e clip podem alterar a acao aplicada. |
| Outage logico de asset | Assets ficam na topologia, mas telemetria e/ou controlo podem ficar indisponiveis. |
| KPIs de robustez | Contam eventos, corrupcoes, observations missing e action dropouts. |

## Dataset e Performance

| Feature | O que melhora |
|---|---|
| CSV e Parquet | Mesmo contrato de dados com storage eficiente. |
| Windowed loader | Carrega apenas a janela necessaria. |
| Shared time-series cache | Reusa weather/pricing/carbon comuns. |
| 15s fixtures | Testes especificos para energia pequena e arredondamentos. |
| Render modes | `none`, `during`, `end` para equilibrar performance e debug. |

## UI e Export

| Feature | O que permite |
|---|---|
| Render/export CSV | Inspecao posterior de episodios. |
| KPI export | Integracao com dashboards. |
| CityLearn UI compatibility | Consumo de `evaluate_v2()` por workflows visuais. |
| Start date | Timestamps humanos em outputs. |

## Testing e Producao

| Area | Cobertura |
|---|---|
| Sub-hourly | 1min/5min/15min/1h/15s. |
| KPIs | Golden e sanity tests. |
| Entity contract | Strict audit e snapshots. |
| Dynamic topology | Add/remove member/assets e refresh de action space. |
| Deferrables | Parser, actions, KPIs, topology add/remove. |
| Market/phases | Conservacao e coerencia de import/export/fases. |
