# Dataset Reference

Esta pagina descreve como construir datasets compativeis com o simulador, incluindo CSV, Parquet, dados reais em kW, EVs, PV absoluto, deferrable appliances e pedidos demand response.

## Formatos Aceites

| Formato | Extensoes | Leitura | Quando usar |
|---|---|---|---|
| CSV | `.csv` | `pandas.read_csv` com `skiprows/nrows` para janelas | Datasets pequenos/medios, facil de inspecionar. |
| Parquet | `.parquet`, `.pq`, `.parq` | `pyarrow.parquet` por batches para janelas | Datasets grandes, 15s, muitos anos/assets. |

CSV e Parquet sao intercambiaveis se:

1. O schema aponta para o novo ficheiro.
2. As colunas tem os mesmos nomes.
3. As unidades sao as mesmas.
4. Os tipos conseguem ser convertidos para os construtores do simulador.

## Suite REC anual canonica de 2023

O repositorio inclui uma escada anual hibrida e reprodutivel de cenarios para a tese e
para as experiencias com algoritmos. Todas as familias usam o ano completo de
2023 em UTC, 35 040 passos de 15 minutos, atributos de calendario local de
Lisboa e ficheiros Parquet autocontidos. O antigo cenario horario de 17
edificios continua disponivel para reproducao historica, mas nao pertence a
esta suite.

Os horarios das rotinas sao construidos primeiro em hora civil
`Europe/Lisbon` e so depois convertidos para UTC. A hora inexistente da mudanca
de primavera e avancada uma hora e, na hora repetida de outono, e escolhida de
forma deterministica a primeira ocorrencia. Assim, as rotinas locais nao
sofrem um desvio de uma hora nos dias de transicao, mantendo-se uma timeline
fisica UTC sem ambiguidades.

| Familia | Membros | PV | BESS | Membros com charging | Chargers | Deferrables | Variantes |
|---|---:|---:|---:|---:|---:|---:|---|
| `rec_2023_micro_4_q` | 4 | 2 | 1 | 2 | 2 | 1 | `MICRO-4-Q` |
| `rec_2023_core_15_stripped` | 15 | 7 | 3 | 6 | 7 | 4 | `CORE-15-STRIPPED` |
| `rec_2023_core_30` | 30 | 15 | 6 | 12 | 15 | 8 | Nominal, Safety, Health, Dynamic e Combined |
| `rec_2023_premium_100` | 100 | 55 | 22 | 45 | 80 | 30 | Clean e AllIn |

Cada familia tem um `schema.json` por defeito; as variantes alternativas estao
em `schemas/`. As variantes apontam para os mesmos ficheiros fisicos, o que
torna as comparacoes Safety, Health, Dynamic e AllIn em clean twins. Os
catalogos em `catalogs/` separam as identidades de membro, charger fisico, EV e
sessao. Assim, um membro pode ter varios chargers e um charger partilhado pode
receber diferentes EVs sem confundir infraestrutura e mobilidade.

O contrato `2023-q15-v1.9` distribui ainda os assets instalados de forma
estratificada, em vez de concentrar toda a flexibilidade nos membros PV com os
indices mais baixos. Em Core-15, Core-30 e Premium existem membros com charging
com e sem PV; o Micro inclui igualmente um caso de cada tipo. As baterias
continuam associadas a PV, mas distribuem-se por
rotinas e indices distintos. Mantem-se a composicao declarada e reduz-se o
confounding entre PV e EV. Os schemas das variantes usam uma raiz relativa ao
diretorio pai e podem ser abertos diretamente a partir de `schemas/`.

Os perfis de procura sao sinteticos, mas os totais anuais sao ancorados nas
classes portuguesas de consumo por consumidor de 2023: domestico, nao domestico
e industrial. Cinco arquetipos de rotina estratificados por categoria completa
alteram horarios de funcionamento, picos, ocupacao, dias de atividade reduzida,
janelas EV e pedidos de cargas adiaveis. A capacidade PV e associada a procura
anual do membro e varia tambem com orientacao e derating. O armazenamento
estacionario usa classes realistas de capacidade/potencia dimensionadas por
autonomia diaria estratificada, com um limite inferior associado ao PV; SOC,
eficiencia e parametros de degradacao variam por membro. As previsoes
meteorologicas incluem erro reprodutivel dependente do horizonte. Os forecasts
derivados de carga e PV usam persistencia causal do dia anterior, com cold start
no step atual durante o primeiro dia; os horizontes de preco OMIE respeitam a
fronteira de publicacao day-ahead, usando valores realizados apenas depois do
cut-off declarado e persistencia diaria causal antes dessa hora. Estas sao
distribuicoes controladas de cenario, nao uma amostra estatisticamente ajustada
de CER portuguesas.

Cada sessao EV normal e individualmente realizavel com a potencia do charger e
95% de eficiencia declaradas. Os pedidos residenciais usam no maximo 85% da
energia disponivel na janela; os de servicos partilhados usam no maximo 72%,
reservando pelo menos um passo de controlo mesmo nas sessoes normais mais
curtas. Safety e AllIn podem ainda tornar o servico
conjunto inviavel devido a procura simultanea, limites totais/por fase,
deadlines e falhas. As variantes dinamicas exercitam separadamente o ciclo de
vida de membros e a remocao/reposicao independente de chargers, PV, baterias
estacionarias e cargas adiaveis.
Um membro que entra nao herda falhas de servico relativas a pedidos que
expiraram antes do inicio da sua participacao. As saidas sao permanentes no ano
do benchmark, evitando um reset implicito do estado acumulado do participante.
Os chargers residenciais ligados a uma fase individual seguem uma rotacao
deterministica por familia, com contagens L1/L2/L3 que diferem no maximo por um.

Um asset com estado que seja reinstalado passa a ser uma nova instancia de
runtime, inicializada no estado de fronteira declarado; nao herda de forma
oculta o SoC nem os servicos pendentes da instancia removida. A instancia
anterior continua disponivel para agregacao de KPIs. A energia que entra na REC
nessa fronteira de ativacao e uma condicao inicial exogena, nao energia de rede
criada dentro de um passo de controlo.

`catalogs/electrical_services.parquet` regista, por membro, o surrogate de
ligacao BTN/BTE portuguesa, a atribuicao mono/trifasica e os limites totais e
por fase de importacao/exportacao. O kVA e mapeado para kW com fator de potencia
unitario explicito; testa-se headroom da ligacao, nao fluxo de potencia ou
protecao da rede. Antes de aplicar assets flexiveis, o gerador verifica o
envelope nativo de procura nao deslocavel e PV e, quando necessario, seleciona
o nivel BTE declarado seguinte com 2% de margem nativa. Assim, uma violacao
normal de Safety resulta de concorrencia controlavel ou de um evento rotulado,
nao de um baseline exogeno impossivel. `file_checksums.sha256` fixa todos os
ficheiros gerados.

Os precos sao os valores oficiais do mercado diario OMIE para Portugal. Cada
periodo horario fisico de 2023 e repetido nos quatro passos de simulacao, sem
interpolacao, e convertido de EUR/MWh para EUR/kWh. Os URLs diarios, revisoes e
hashes SHA-256 ficam em `sources/`. O settlement faz matching local no mesmo
passo, usa um preco local igual a 80% do OMIE, pesos iguais, exportacao residual
sem remuneracao e um contrafactual grid-only.

```console
.venv/bin/python scripts/generate_annual_rec_suite.py --families all
.venv/bin/python scripts/audit/audit_annual_rec_suite.py
.venv/bin/python scripts/audit/audit_annual_rec_suite_diversity.py
.venv/bin/python scripts/audit/audit_annual_rec_suite_scientific.py
.venv/bin/python scripts/audit/smoke_annual_rec_suite.py
```

A auditoria de diversidade mede duplicados exatos, correlacoes entre perfis e
entre arquetipos, horas de pico, distribuicoes PV/BESS, chegadas, duracoes e
slack EV, e duplicacao de schedules de deferrables.

## Contrato de Unidades

| Dado | Unidade esperada |
|---|---:|
| Cargas e consumos por step | kWh/step |
| PV em `generation_mode="absolute"` | kWh/step |
| PV em `generation_mode="per_kw"` | W/kW |
| BESS/charger power limits | kW |
| EV required/estimated SOC no ficheiro charger | percent no CSV original, convertido para ratio internamente |
| Precos | currency/kWh |
| Target de pedidos DR | kW |
| Precos e penalizacoes DR | currency/kWh |
| Carbon intensity | kgCO2/kWh |
| Weather temperature | C |
| Irradiance | W/m2 |

Dados reais em potencia devem ser convertidos:

```text
kWh_per_step = kW * seconds_per_time_step / 3600
kW = kWh_per_step * 3600 / seconds_per_time_step
```

Para 15 segundos:

```text
1 kW durante 15s = 1 * 15 / 3600 = 0.0041666667 kWh
```

## `energy_simulation` Columns

| Coluna | Obrigatoria | Unidade | O que e |
|---|---:|---:|---|
| `month` | sim | 1-12 | Mes. |
| `hour` | sim | 1-24 | Hora. |
| `day_type` | sim | 1-8 | Dia da semana/especial. |
| `minutes` | recomendada sub-hora | 0-59 | Minuto. |
| `seconds` | recomendada sub-minuto | 0-59 | Segundo. |
| `indoor_dry_bulb_temperature` | sim | C | Temperatura interior. |
| `non_shiftable_load` | sim | kWh/step | Load nao flexivel. |
| `dhw_demand` | sim | kWh/step | Demanda DHW. |
| `cooling_demand` | sim | kWh/step | Demanda cooling. |
| `heating_demand` | sim | kWh/step | Demanda heating. |
| `solar_generation` | sim | depende PV mode | Input PV. |
| `daylight_savings_status` | nao | 0/1 | DST. |
| `average_unmet_cooling_setpoint_difference` | nao | C | Desconforto cooling. |
| `indoor_relative_humidity` | nao | percent | Humidade interior. |
| `occupant_count` | nao | pessoas | Ocupacao. |
| `indoor_dry_bulb_temperature_cooling_set_point` | nao | C | Setpoint cooling. |
| `indoor_dry_bulb_temperature_heating_set_point` | nao | C | Setpoint heating. |
| `hvac_mode` | nao | enum | 0 off, 1 cooling, 2 heating, 3 auto. |
| `power_outage` | nao | 0/1 | Outage. |
| `comfort_band` | nao | C | Banda conforto. |

Cooling e heating demand nao podem ser positivos no mesmo timestep.

## `weather` Columns

| Coluna | Unidade |
|---|---:|
| `outdoor_dry_bulb_temperature` | C |
| `outdoor_relative_humidity` | percent |
| `diffuse_solar_irradiance` | W/m2 |
| `direct_solar_irradiance` | W/m2 |
| `outdoor_dry_bulb_temperature_predicted_1/2/3` | C |
| `outdoor_relative_humidity_predicted_1/2/3` | percent |
| `diffuse_solar_irradiance_predicted_1/2/3` | W/m2 |
| `direct_solar_irradiance_predicted_1/2/3` | W/m2 |

## `pricing` Columns

| Coluna | Unidade |
|---|---:|
| `electricity_pricing` | currency/kWh |
| `electricity_pricing_predicted_1` | currency/kWh |
| `electricity_pricing_predicted_2` | currency/kWh |
| `electricity_pricing_predicted_3` | currency/kWh |

## `carbon_intensity` Columns

| Coluna | Unidade |
|---|---:|
| `carbon_intensity` | kgCO2/kWh |

## Charger Simulation Columns

| Coluna | Unidade/formato | O que e |
|---|---:|---|
| `electric_vehicle_charger_state` | enum | 1 connected, 2 incoming, 3 away/commuting. |
| `electric_vehicle_id` | string | ID do EV. |
| `electric_vehicle_departure_time` | steps | Steps ate departure. Default interno `-1`. |
| `electric_vehicle_required_soc_departure` | percent | SOC requerido. Convertido para ratio. Default interno `-0.1`. |
| `electric_vehicle_estimated_arrival_time` | steps | Steps ate chegada. Default interno `-1`. |
| `electric_vehicle_estimated_soc_arrival` | percent | SOC estimado na chegada. Convertido para ratio. |
| `electric_vehicle_current_soc` | percent ou ratio | Opcional. Telemetria de SOC atual. Na suite REC anual é uma referência neutra de inicialização de fronteira, com serviço a taxa constante, usada apenas quando um episódio começa dentro de uma sessão ocupada; não substitui a evolução do SOC num rollout ininterrupto. |

Para datasets sub-horarios, os campos `*_time` devem estar em numero de timesteps da resolucao do dataset. Ex.: 1 hora em 15s = 240 steps.

## Deferrable Appliances

O formato oficial e esparso: um catalogo de perfis e um schedule de pedidos. Nao repetir o `load_profile` em todas as linhas temporais.

### `cycle_profiles_file`

| Coluna | Unidade | O que e |
|---|---:|---|
| `profile_id` | string | ID do perfil. |
| `duration_steps` | steps | Duracao do ciclo. |
| `total_energy_kwh` | kWh | Soma do perfil. |
| `load_profile` | lista kWh/step | Perfil de energia por step. |

Validacoes:

| Validacao | Regra |
|---|---|
| `profile_id` | Nao vazio e unico. |
| `duration_steps` | Inteiro > 0. |
| `load_profile` | Lista nao vazia, finita, nao negativa. |
| Soma | `sum(load_profile) == total_energy_kwh` dentro de tolerancia. |

### `flexibility_schedule_file`

| Coluna | Unidade | O que e |
|---|---:|---|
| `cycle_id` | string | ID unico do pedido/ciclo. |
| `profile_id` | string | Referencia ao catalogo. |
| `earliest_start_time_step` | timestep global | Primeiro start permitido. |
| `latest_start_time_step` | timestep global | Ultimo start permitido. |
| `deadline_time_step` | timestep global | Deadline de conclusao. |
| `priority` | ratio | Prioridade 0-1, clipped. |
| `must_run` | bool | Se o pedido e obrigatorio. |

Validacoes:

| Validacao | Regra |
|---|---|
| `cycle_id` | Unico e nao vazio. |
| `profile_id` | Tem de existir no catalogo. |
| Janelas | `earliest <= latest`. |
| Deadline | `latest + duration_steps - 1 <= deadline`. |
| Timesteps | Inteiros >= 0 e globais. |

## PV Datasets

| Caso | Schema | Coluna `solar_generation` |
|---|---|---|
| Medicao real/absoluta | `"generation_mode": "absolute"` | `kWh/step`. |
| Perfil normalizado | `"generation_mode": "per_kw"` | `W/kW` por 1 kW instalado. |

Em datasets reais, recomenda-se `absolute`.

## 15 Segundos

Para datasets a 15s:

| Tema | Recomendacao |
|---|---|
| `seconds_per_time_step` | `15`. |
| Cargas em kW | Converter para kWh/step antes de escrever dataset. |
| PV real | Escrever kWh/step e usar `generation_mode="absolute"`. |
| EV countdowns | `*_departure_time` e `*_arrival_time` em steps de 15s. |
| Ficheiros grandes | Preferir Parquet. |
| Teste antes de treino | Correr smoke episode pequeno e `evaluate_v2()`. |
| Exemplo compacto com assets dinamicos | `data/datasets/citylearn_three_phase_dynamic_asset_changes_demo_15s_parquet/schema.json` tem 7 dias a 15s com eventos add/remove de chargers, PV e BESS. |
| Exemplo demand response | `data/datasets/citylearn_challenge_2022_phase_all_demand_response/schema.json` tem buildings 2022 phase-all sem EVs e pedidos DR district. |
| Exemplo robustez | `data/datasets/citylearn_challenge_2022_phase_all_robustness/schema.json` tem buildings 2022 phase-all sem EVs e eventos esparsos de robustez. |

## Bundles de Observacao Entity nos Datasets Incluidos

Os datasets entity a 15 segundos e `citylearn_challenge_2022_phase_all_plus_evs`
ativam todos os bundles de observacoes entity:

| Bundle | Objetivo |
|---|---|
| `entity_core_electrical` | Potencia, energia, SOC e capacidade fisica por asset. |
| `entity_community_operational` | Agregados de potencia, headroom e flexibilidade da comunidade. |
| `entity_forecasts_existing` | Observacoes `*_predicted_*` existentes no dataset. |
| `entity_forecasts_derived` | Forecasts pontuais compactos perfeitos para preco, load, PV e net demand. |
| `entity_temporal_derived` | Calendario robusto e lags curtos. |
| `entity_action_feedback` | Feedback de acao pedida, limitada e aplicada com motivos de clipping. |
| `entity_demand_response` | Pedido demand response district atual, baseline e delivery/shortfall anterior. |
| `entity_robustness` | Estado de robustez ativo e contadores de corrupcao do step anterior. |

Outros schemas mantem o comportamento compativel por default salvo se declararem
`observation_bundles`.

## Ficheiros de Pedidos Demand Response

Datasets de demand response definem `schema["demand_response"]["enabled"] = true`, apontam `requests_file` para CSV ou Parquet, usam `interface="entity"` e normalmente ativam `observation_bundles.entity_demand_response`.

| Coluna | Unidade/dominio | O que e |
|---|---:|---|
| `request_id` | string | ID unico do pedido. |
| `issuer` | `dso`/`tso` | Emissor do pedido. |
| `direction` | `up`/`down` | Perspetiva da carga: `up` aumenta net load, `down` reduz net load. |
| `start_time_step`, `end_time_step` | timestep global | Janela inclusiva de ativacao. |
| `target_power_kw` | kW | Target positivo da comunidade/district. |
| `activation_price_eur_per_kwh` | currency/kWh | Pagamento pela energia entregue creditada. |
| `shortfall_penalty_eur_per_kwh` | currency/kWh | Penalizacao por energia em shortfall. |
| `tolerance_power_kw` | kW | Tolerancia opcional; default `0`. |

Na v1 o simulador calcula uma baseline congelada no inicio do evento a partir dos `baseline_window_seconds` anteriores, liquida apenas steps ativos e guarda historico esparso em vez de arrays densos por timestep.

## Ficheiros de Eventos de Robustez

Datasets de robustez definem `schema["robustness"]["enabled"] = true`, apontam `events_file` para CSV ou Parquet e ligam apenas os modulos que querem estudar. Flat e entity sao suportados; em entity pode ativar-se `observation_bundles.entity_robustness` para diagnostico.

| Coluna | Unidade/dominio | O que e |
|---|---:|---|
| `event_id` | string | ID unico do evento. |
| `module` | enum | `observation`, `forecast`, `action` ou `asset`. |
| `target_type` | tipo de entidade | `district`, `building`, `storage`, `charger`, `ev`, `pv` ou `deferrable_appliance`. |
| `target_id` | id ou `*` | ID/nome da entidade ou todos os targets compativeis. |
| `target_feature` | feature/acao | Para assets usar `telemetry`, `control` ou `both`. |
| `start_time_step`, `end_time_step` | timestep global | Janela inclusiva do evento. |
| `mode` | enum | Modo suportado pelo modulo selecionado. |
| `value`, `std`, `min_value`, `max_value`, `replacement_value`, `delay_steps` | opcional | Parametros do modo. |

## Performance e Loader

| Otimizacao | O que faz |
|---|---|
| Windowed CSV | Usa `skiprows/nrows` para carregar apenas a janela. |
| Windowed Parquet | Le por batches e corta por rows. |
| Shared cache | Reusa weather/pricing/carbon quando varios buildings apontam para o mesmo ficheiro e `noise_std=0`. |
| Parquet | Menos disco, leitura tipada, melhor para 15s. |

## Checklist de Dataset Novo

1. Definir `seconds_per_time_step`.
2. Converter todas as potencias medidas para `kWh/step` onde o simulador espera energia.
3. Decidir PV `absolute` ou `per_kw`.
4. Garantir que EV schedule usa countdowns em timesteps da resolucao.
5. Escrever deferrables em catalogo + schedule.
6. Para demand response, escrever ficheiro esparso de pedidos e ativar entity mode mais `entity_demand_response`.
7. Para robustez, escrever ficheiro esparso de eventos e ativar apenas os modulos necessarios.
8. Preferir Parquet para datasets anuais/sub-minuto.
9. Correr smoke run e `evaluate_v2()`.
10. Correr `audit_physics.py` quando o dataset for novo ou critico.
