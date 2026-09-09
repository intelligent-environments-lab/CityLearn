# Actions Reference

Todas as acoes sao normalizadas para uma faixa simples no agente. O simulador converte para energia/potencia fisica usando capacidade, potencia nominal, limites e `seconds_per_time_step`.

## Regras Gerais

| Regra | Contrato |
|---|---|
| Bounds | Vem de `env.action_space` e `env.action_names`. |
| Storage actions | Valores positivos carregam, negativos descarregam. |
| EV charger actions | Positivo carrega EV, negativo descarrega/V2G se permitido. |
| Deferrable actions | Comando binario de start. `action > trigger_threshold` tenta iniciar o proximo ciclo pendente (`trigger_threshold` default: `0.5`). |
| Unidades fisicas internas | A energia aplicada no step e `kWh/step`. Limites de potencia sao `kW`. |
| Constraints | Charging/electrical service podem cortar a acao antes da aplicacao. |

## Acoes Flat

| Acao | Range tipico | Entidade | Efeito |
|---|---:|---|---|
| `cooling_or_heating_device` | `[-1, 1]` | building | Negativo controla cooling, positivo controla heating. Incompativel com `cooling_device` e `heating_device` ativos em simultaneo. |
| `cooling_device` | `[0, 1]` | building | Controla setpoint/demand cooling em buildings dinamicos. |
| `heating_device` | `[0, 1]` | building | Controla setpoint/demand heating em buildings dinamicos. |
| `cooling_storage` | `[-limit, limit]` | building | Carrega/descarrega storage termico de cooling. |
| `heating_storage` | `[-limit, limit]` | building | Carrega/descarrega storage termico de heating. |
| `dhw_storage` | `[-limit, limit]` | building | Carrega/descarrega storage DHW. |
| `electrical_storage` | `[-1, 1]` | building | Controla BESS. |
| `electric_vehicle_storage_{charger_id}` | `[-1, 1]` ou `[0, 1]` | charger | Controla EV ligado ao charger. Limite negativo depende de `max_discharging_power`. |
| `deferrable_appliance_{appliance_id}` | `[0, 1]` | appliance | Comando binario (`0` off, `1` on) para iniciar o proximo ciclo elegivel. |

### Conversao de BESS

Para `electrical_storage`, a acao normalizada e convertida com a potencia nominal:

```text
energy_kwh_step = action * nominal_power_kw * seconds_per_time_step / 3600
```

Depois o storage aplica eficiencia, SOC, depth of discharge, power curves e capacidade.

### Conversao de EV Charger

Para EVs, a acao normalizada e convertida usando `max_charging_power` ou `max_discharging_power`:

```text
charge_energy_kwh_step = action_positive * max_charging_power_kw * seconds_per_time_step / 3600
discharge_energy_kwh_step = action_negative * max_discharging_power_kw * seconds_per_time_step / 3600
```

O charger aplica disponibilidade de EV, SOC, required SOC, eficiencia e constraints.

### Deferrable Appliances

O appliance so inicia se:

| Condicao | Regra |
|---|---|
| Ciclo pendente | Existe um ciclo com estado `pending`. |
| Acao | `action > trigger_threshold` (default `0.5`). |
| Janela | `earliest_start_time_step <= current_global_time_step <= latest_start_time_step`. |
| Deadline | `current + duration_steps - 1 <= deadline_time_step`. |
| Episodio | O ciclo cabe dentro do episodio atual. |

Se passar `latest_start_time_step` sem start valido, o ciclo fica `missed`. Se nao houver ciclo pendente, a acao nao tem efeito.

Recomendacao para controladores: usar `deferrable_appliance_can_start` como sinal de disponibilidade e enviar `1.0` apenas quando pretende arrancar.

## Entity Actions

Em entity mode, as acoes ficam agrupadas por tabela:

| Tabela | Feature | Corresponde a |
|---|---|---|
| `building` | Actions de building | `cooling_storage`, `dhw_storage`, `electrical_storage`, etc. |
| `charger` | `electric_vehicle_storage` | Uma row por charger ativo. |
| `deferrable_appliance` | `start` | Uma row por appliance ativo. |

Exemplo por tabelas:

```python
actions = {
    "tables": {
        "building": [[0.0, 0.2, -0.1]],
        "charger": [[1.0], [0.0]],
        "deferrable_appliance": [[1.0]]
    }
}
obs, reward, terminated, truncated, info = env.step(actions)
```

Exemplo por `map`:

```python
actions = {
    "map": {
        "building:Building_1": {"electrical_storage": 0.1},
        "charger:Building_1:AC001": {"electric_vehicle_storage": 0.7},
        "deferrable_appliance:Building_1:washer_1": {"start": 1.0}
    }
}
```

## Ordem de Aplicacao no Step

A ordem simplificada dentro do building e:

1. Resolver acoes ativas e preencher inativas com zero ou `nan` conforme o tipo.
2. Aplicar charging/electrical constraints a EVs e BESS.
3. Atualizar demandas dinamicas quando aplicavel.
4. Atualizar dispositivos HVAC e storages termicos.
5. Atualizar non-shiftable load.
6. Atualizar BESS.
7. Atualizar EV chargers.
8. Atualizar deferrable appliances.
9. Atualizar net balance, rewards, KPIs e observacoes.

Quando `electrical_storage_action < 0`, a descarga BESS ganha prioridade antes de outros consumos eletricos.
