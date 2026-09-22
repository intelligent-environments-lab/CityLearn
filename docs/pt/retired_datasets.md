# Notas de formatos de datasets retirados

## Suite REC anual de 2023 retirada

As versoes 1.8.0 a 3.0.1 incluiam uma escada anual hibrida e reprodutivel de cenarios para a tese e
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

Os cinco datasets `rec_2023_*` foram retirados do repositorio publico e do
registry por defeito no CityLearn 3.0.2. Esta secao e mantida apenas como
documentacao historica do formato; os utilizadores devem fornecer os seus
proprios dados locais.

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
