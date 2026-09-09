# EC_Ermesinde

Dataset anual final para a estação de Ermesinde, com 35 040 períodos de
15 minutos (2025, sem 29 de fevereiro) e seis ativos de escadas rolantes.
Foi preparado para a interface *flat* do simulador CityLearn.

## Conteúdo

- `schema.json` — cenário final a 900 s, com observações e ações das escadas
  ativas.
- `Building_1.csv`, `weather.csv` e `pricing.csv` — séries compatíveis com a
  resolução de 15 minutos.
- `EscadaRolante_1.csv` a `EscadaRolante_6.csv` — procura agregada e estado de
  disponibilidade de cada ativo lógico.
- `ermesinde_eventos_comboios.csv` — tabela validada dos 200 eventos de
  comboio usados para auditar a procura.

## Modelo das escadas rolantes

Cada ação `escalator_EscadaRolante_N` está normalizada entre 0 e 1:

| Intervalo da ação | Estado aplicado | Serviço quando há procura |
| --- | --- | --- |
| `[0, 1/3[` | `standby` | não servido |
| `[1/3, 2/3[` | `slow` | servido |
| `[2/3, 1]` | `normal` | servido |

Não há modelo de filas nem capacidade individual por passageiro. A procura é
agregada ao período de 15 min. Se existir procura acima de 0,5 passageiros e a
escada estiver em `standby`, essa procura entra em
`unserved_passengers_15min`; em `slow` e `normal` é considerada servida. Assim
o agente aprende uma troca direta e explícita entre energia e nível de serviço.

As potências são em kW por ativo lógico:

| Escadas | Standby | Slow | Normal | Interpretação |
| --- | ---: | ---: | ---: | --- |
| 1–2 | 0,08 | 0,42 | 2,10 | uma escada física |
| 3–6 | 0,10 | 0,52 | 2,62 | grupo de duas escadas físicas |

Nas escadas 3–6 as potências fornecidas eram por unidade física e foram
duplicadas. `minimum_state_steps=1`, portanto a escolha pode mudar em cada
período de 15 min. Quando `available=0`, a escada é forçada a `standby`.

### Observações principais

Para cada ativo são expostas, por exemplo,
`escalator_EscadaRolante_1_passengers_expected_15min`,
`..._people_detected`, `..._passing_trains`, `..._minutes_to_next_train`,
`..._state`, `..._power_kw`, `..._service_met` e
`..._unserved_passengers_15min`.

`people_detected` é um indicador de procura (`passengers_expected_15min >
0,5`), não uma regra de segurança que obrigue o simulador a ligar a escada.
Os campos de chegada e partida do ficheiro de origem representam o mesmo
comboio em passagem; o simulador expõe apenas `passing_trains` para não o
contar duas vezes.

## Correções e transformação dos dados

1. As séries base horárias foram convertidas para 15 min. Cargas e demandas
   energéticas foram divididas por quatro; meteorologia e estado térmico usam
   retenção do valor horário. Foram acrescentadas `minutes`, `dhw_demand=0` e
   `solar_generation=0` por ausência de medições de AQS e FV.
2. A tarifa é sintética, apenas para testar a observação de preço: 0,120
   EUR/kWh das 01:00–07:00, 0,180 EUR/kWh das 08:00–17:00 e 23:00–24:00, e
   0,260 EUR/kWh das 18:00–22:00. As previsões são perfeitas a 6, 12 e 24 h.
3. As escadas 1 e 2 (átrio) foram regeneradas a partir das séries já corrigidas
   das plataformas: 3+5 para o movimento em direção às plataformas e 4+6 para
   o sentido contrário. Isto elimina 1 230 eventos duplicados obsoletos que
   ainda persistiam no ficheiro de átrio recebido. Os peões independentes do
   átrio foram preservados.
4. Foi removido o único fluxo de fim de ano que pertencia ao próximo ano. Todos
   os ficheiros finais respeitam `passengers_expected_15min =
   passengers_from_trains_15min + background_pedestrians_15min`.

O script reprodutível está em
[`scripts/generate_ec_ermesinde_dataset.py`](../../../scripts/generate_ec_ermesinde_dataset.py).

## Uso

Use o caminho do schema para garantir que é lida esta versão local, sem uma
eventual cópia antiga em cache:

```python
from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(
    "data/datasets/EC_Ermesinde/schema.json",
    central_agent=True,
    episode_time_steps=96,  # um dia
    render_mode="none",
)
observations, info = env.reset(seed=0)
actions = [0.5] * 6  # slow nas seis escadas
observations, reward, terminated, truncated, info = env.step(actions)
```

Para resultados económicos reais, substituir a tarifa sintética por um
tarifário contratado ou por preços de mercado documentados.
