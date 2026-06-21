# Autoavaliação pela Rubrica — Projeto Risco de Geada

Avaliação do entregável (`report.html`/`notebook.ipynb` + pipeline em `src/`) contra a rubrica do
professor. **É uma estimativa minha, baseada em evidência do próprio relatório/código — não é a
nota oficial.** Serve para você saber onde está forte e onde blindar antes da apresentação.

**Escala usada:** Excede ▸ Atende plenamente ▸ Atende ▸ Atende parcial ▸ Não atende.

---

## 1 · Quadro-resumo

| Critério (rubrica) | Conceito | Evidência-chave |
|---|---|---|
| **Implementação** (Spark end2end, tipos, limpo, sem abuso de Pandas) | **Excede** | schema explícito (`StructType`), Parquet particionado, `pandas` só em resultado agregado |
| **Exploração** (estats, manipular/organizar, visuais) | **Excede** | `describe`, nulos por coluna, +15 figuras com título/eixo/legenda |
| **Limpeza** (dados problemáticos, seleção, small data, justificar) | **Excede** | limpeza defensiva→NULL, whitelist de elementos, recorte = small data justificado |
| **Robustez** ("simples>complexo", decisão por dados) | **Excede** | baseline de persistência + ablação + leitura honesta do AUC 0,98 |
| **Modelagem** (labels, split, comparação, métrica+tuning, limitações) | **Atende plenamente** | `lead` p/ rótulo, split **temporal**, LR×RF×GBT, `CrossValidator`, limitações listadas |
| **Comunicação** (markdown, gráficos c/ título/legenda, storytelling) | **Excede** | 10 seções narradas, 52 títulos/eixos/legendas, TL;DR + mini-dicionário |
| **Avaliação geral** (solução completa, do problema à solução) | **Excede** | da disrupção de safra (§1) ao mapa de armazenagem D+1…15 (§8.8) |
| **ML end-to-end no Spark** | **Excede** | da leitura do `.csv.gz` ao GBT tunado, tudo em PySpark |
| **Uso do Spark & boas práticas** | **Excede** | pivô, broadcast join, funções de janela, medallion, AQE; sem abuso de Pandas |

**Conceito geral indicativo:** faixa **A / nota alta** — projeto completo, honesto e tecnicamente
bem executado. Os poucos riscos (seção 4) são de **apresentação/percepção**, não de substância.

---

## 2 · Avaliação detalhada por critério

### Implementação — Excede
- **Spark de ponta a ponta:** ingestão (`ingest_observations`, `stream_ingest`) → silver
  (`build_silver`) → gold (`build_gold`) → modelo (`train_frost_classifier`), tudo em PySpark.
- **Tipos de dado corretos:** `StructType`/`StructField` explícitos no bronze (sem `inferSchema`
  caro), `to_date`, casts deliberados.
- **Código limpo e modular:** funções pequenas e nomeadas (`pivot_wide`, `add_lag_features`,
  `add_regional_features`), config externalizada em `config.yaml`, paths ancorados ao root.
- **Sem abuso de Pandas (verificado):** *todo* `.toPandas()` no relatório vem **depois** de uma
  agregação ou `.sample()`. O corpus nunca é carregado em memória do driver. ✅ exatamente o que a
  rubrica pede para **não** perder nota.
- *Risco:* parte da lógica Spark mora em `src/` (o notebook reusa), então o avaliador vê
  `from src...` em vez do código inline → ver seção 4.

### Exploração — Excede
- **Estatísticas básicas:** `describe()` + schema antes de modelar (§4.2); distribuição por ELEMENT.
- **Selecionar/manipular/organizar:** pivô long→wide, derivação de colunas, agregações por mês /
  banda de elevação / célula de grade.
- **Análises visuais ricas (§5):** histograma de temperatura, taxa de geada por mês e por dia do
  ano, elevação×geada, mapas global e do Corn Belt, precipitação 90 dias. Todas com título e eixos.

### Limpeza — Excede
- **Dados problemáticos tratados:** drop de `Q_FLAG` no ingest; no silver, limpeza defensiva
  (faixa física de temperatura, consistência TMIN≤TMAX, precip≥0) → vira **NULL** (não derruba a
  estação-dia), depois `Imputer(median)`.
- **Seleção de características:** whitelist de elementos (TMAX/TMIN/PRCP/SNOW/SNWD); grupos de
  features explícitos no trainer; ablação de precipitação.
- **"Small data" gerado e justificado:** o **recorte geográfico** (Corn Belt, 20 anos) é a decisão
  central, com justificativa de densidade de sinal e tratabilidade (§2.1, §9.3).

### Robustez — Excede
- **"simples > complexo":** começa por um **baseline de persistência** e exige que todo modelo o
  supere (§7.1/§7.2). O ganho do ML é dimensionado honestamente, não inflado.
- **"simples ≠ simplório":** features de janela, sinal regional/vizinhança, multi-horizonte —
  profundidade sem complexidade gratuita.
- **Decisões por dados:** ablação mostra que precip ajuda só marginalmente; importâncias guiam a
  feature regional; recorte justificado por volume; `local[8]` escolhido após o OOM medido.

### Modelagem — Atende plenamente
- **Labels organizados:** `next_day_is_frost` via `F.lead` (sem vazamento) — bem feito.
- **Split treino×teste:** **temporal** (treina anos < teste; segura 2024), com a justificativa
  explícita de por que split aleatório vazaria.
- **Comparação de técnicas:** baseline → LR → RandomForest → GBT, com gráfico comparativo (§7.2).
- **Métrica + tuning:** `CrossValidator` + `ParamGridBuilder` otimizando areaUnderROC; relata
  `avgMetrics` e melhor combinação (§7.4).
- **Limitações apontadas:** subamostra no treino, folds aleatórios da CV (vazamento leve admitido),
  grade ~1° em vez de vizinho real, `lag(1)`=linha anterior, modelo global único (§10).
- *Por que "plenamente" e não "excede":* o tuning final roda em **subamostra** por restrição de
  máquina (métrica honesta no teste completo, mas o modelo "de produção" não foi treinado no corpus
  inteiro); a CV temporal rigorosa (forward-chaining) não foi usada. Ambos **assumidos** — o que já
  é o comportamento certo.

### Comunicação — Excede
- **Markdown rico:** 10 seções com narrativa, Resumo + TL;DR, "como ler o notebook",
  mini-dicionário de clima para leigos.
- **Gráficos com título/eixo/legenda:** **52** ocorrências medidas — cada figura é legendada.
- **Storytelling:** arco problema → dados → limpeza → features → modelo → avaliação honesta →
  decisão de negócio. Termina onde começou (a oferta da commodity).

### Avaliação geral — Excede
- **Solução completa de ponta a ponta:** do problema de negócio (§1) à entrega acionável — mapa de
  prioridade de armazenagem e horizonte D+1…D+15 (§8.7/§8.8).
- **Implantação faz sentido:** consumidor claro (cadeia de commodities), decisão clara (hedge,
  realocar originação, logística), e até o caminho de operacionalização (treinar com ERA5, servir
  com ECMWF OP).

---

## 3 · Destaques que pesam a favor (mostre na banca)

1. **Honestidade analítica (§8.4–8.6):** reconhecer que o AUC 0,98 é, em boa parte, **persistência
   térmica**, e que o ganho real do ML é trocar recall por precisão + probabilidade ajustável. Isso
   sinaliza maturidade — é raro e bem visto.
2. **Engenharia > complexidade:** o maior salto veio da **feature de vizinhança**, não de um modelo
   maior. Casa direto com "simples > complexo".
3. **Disciplina de Spark:** pivô + broadcast join sem shuffle, funções de janela com `lead`
   anti-vazamento, medallion que não reprocessa upstream, AQE ligado.
4. **Limite real exposto e tratado:** o OOM em `local[*]` → `local[8]` é exatamente o tipo de
   "decisão baseada em dados" sobre o trade-off paralelismo↔memória.
5. **Conexão problema→negócio fechada:** poucos projetos chegam ao "onde guardar a safra" e "quantos
   dias de antecedência".

---

## 4 · Riscos / o que poderia custar nota (e como mitigar)

| Risco | Por que importa | Mitigação |
|---|---|---|
| **Lógica Spark mora em `src/`, não inline no notebook** | O professor pediu "notebook com o processamento end-to-end"; um avaliador estrito pode querer ver as transformações no notebook, não `from src import...` | Na banca, **abra §3.1 e §6.5** (que mostram pivô/broadcast/camadas inline) e diga "o notebook orquestra o código de produção — mesma SparkSession; aqui está a transformação real". |
| **Densidade/extensão (10 seções, 1688 linhas)** | "simples > complexo" também vale para o relatório; pode soar pesado | Na fala, **não percorra tudo**; use as 7 figuras-herói do guia de apresentação. |
| **Treino em subamostra + CV com folds aleatórios** | Pode ser cobrado como rigor de modelagem | Já está admitido no §10; diga em voz alta que a **métrica final é no teste completo** e que cluster treinaria no corpus inteiro. |
| **Foco numa só região (Corn Belt)** | Generalização | Frase pronta: "mesmo código escala mudando uma linha de config; o gap de cobertura (sul do Brasil) tem solução proposta com ERA5/ECMWF". |
| **`pandas` aparece no relatório** | Redutor se for mal interpretado | Verificado: só pós-agregação/sample. Diga isso explicitamente ao mostrar a 1ª figura. |

---

## 5 · Veredito

Projeto **completo, honesto e tecnicamente sólido** — cumpre todos os critérios da rubrica, e
**excede** na maioria (implementação, exploração, limpeza, robustez, comunicação, uso de Spark,
avaliação geral). O único critério em "atende plenamente" (Modelagem) é por escolhas de máquina
**assumidas**, não por erro. Os riscos restantes são de **percepção** e se resolvem na narrativa da
apresentação — todos têm frase de mitigação acima.

> **Faixa indicativa:** topo da turma / conceito A. O diferencial frente a um projeto "bom" é a
> **leitura honesta dos resultados** e a **ponte fechada até a decisão de negócio** — garanta que
> os dois apareçam na banca (blocos 7 e 8 do guia de apresentação).
