# Guia de Apresentação — Risco de Geada (10 minutos)

Como apresentar o `report.html` (notebook end-to-end de previsão de geada) em **~10 minutos**,
cobrindo **todos** os critérios da rubrica do professor. O relatório tem 10 seções — você **não**
vai percorrer as 10 ao vivo. Você vai **contar uma história** (problema → dados → solução →
resultado → decisão de negócio) e **mergulhar** em poucas figuras-chave como evidência.

> **Regra de ouro do tempo:** 10 min ≈ **1 slide por minuto**. Fale a história; deixe as figuras
> do relatório serem a prova. Se atrasar, corte o que está marcado como ✂️ *(opcional)*.

---

## 1 · Orçamento de tempo (o esqueleto)

| # | Bloco | Tempo | Seções do relatório | Critérios atingidos |
|---|---|---|---|---|
| 1 | Problema & negócio (abertura/gancho) | **1:00** | §1.1, §1.2 | Avaliação geral · Negócio |
| 2 | O dataset e **a decisão central: o recorte** | **1:00** | §2.1, §2.4, §9.3 | Limpeza (small data) · Robustez · Criatividade |
| 3 | Arquitetura **medallion + Spark end-to-end** | **1:15** | §3.1, §6.5 | Implementação · ML end-to-end no Spark |
| 4 | Limpeza & EDA (o sinal nos dados) | **1:15** | §4.1, §5.2, §5.4 | Exploração · Limpeza · Comunicação |
| 5 | Feature engineering (gold) — **janelas** | **1:00** | §6.1, §6.2, §6.4 | Criatividade · Modelagem · Robustez |
| 6 | Modelagem: baseline → comparação → tuning | **1:30** | §7.1, §7.2, §7.4 | **Modelagem** · Implementação |
| 7 | Resultados & **avaliação honesta** | **1:30** | §8.1–8.5 | Robustez · Modelagem · Decisão por dados |
| 8 | Do modelo ao **negócio** | **0:45** | §8.7, §8.8 | Avaliação geral · Negócio |
| 9 | Limitações, **Spark na prática (OOM)** & conclusão | **0:45** | §9.5, §10 | Robustez · Boas práticas Spark |
| — | **Total** | **~10:00** | | **todos** |

> Sobra ~0 de folga: ensaie uma vez com cronômetro. Se você for de fala rápida, o bloco 7
> (avaliação honesta) é onde **ganhar** mais nota — não o encurte; encurte o bloco 4.

---

## 2 · As 7 figuras "herói" (mostre só estas ao vivo)

Não dá tempo de abrir tudo. Pré-marque estas no HTML (ou exporte para os slides). Elas sozinhas
contam a história inteira:

1. **§5.2 — Taxa de geada por mês** → "existe um sinal sazonal claro que o modelo precisa aprender".
2. **§6.5 — As três camadas lado a lado (bronze→silver→gold)** → prova visual do medallion + Spark.
3. **§7.2 — Comparação de modelos (LR × RF × GBT × baseline)** → decisão por dados entre técnicas.
4. **§8.3 — Importância das features** → "o modelo raciocina como um agrônomo".
5. **§8.4 — AUC 0,98 honesto vs. baseline de persistência** → maturidade analítica (o anti-hype).
6. **§8.5 — Ajuste de limiar ao custo de negócio (FN ≫ FP)** → conecta métrica à decisão.
7. **§8.8 — Mapa de prioridade de armazenagem** → fecha do problema à solução de negócio.

> Se faltar MUITO tempo, as 3 imprescindíveis são **§7.2**, **§8.4** e **§8.8** — comparação,
> honestidade e negócio.

---

## 3 · Roteiro detalhado (fala por bloco)

### Bloco 1 — Problema & negócio · 1:00 · §1.1–1.2
**Gancho (decore esta frase):**
> *"Uma geada fora de época destrói lavoura, derruba a oferta de uma commodity — milho, soja,
> café — e move preço e contratos. Nós construímos um classificador que responde uma pergunta de
> negócio: **vai gear amanhã nesta estação?**"*

- Tipo de problema: **classificação binária supervisionada**; alvo `next_day_is_frost`
  (TMIN de amanhã ≤ 0 °C).
- Consumidor: agente da **cadeia de commodities agrícolas**; risco central = **disrupção de safra**.
- Saída: classe 0/1 **+ probabilidade** → vira um *Frost-Risk Score* acionável.
- ✂️ *(opcional)* por que classificação e não regressão: a decisão (proteger/adiar plantio) é binária.

### Bloco 2 — Dataset & o recorte · 1:00 · §2.1, §2.4, §9.3
- Fonte: **NOAA GHCN-Daily** — +120 mil estações, diário, redistribuível, com flags de QC.
- **A decisão mais importante do projeto:** o **recorte geográfico**. Sair de ~3 bilhões de linhas
  (global) para **~40 M (Corn Belt dos EUA, 20 anos)** é o que torna o problema **tratável** e o
  sinal **denso** — gerar "small data" de propósito.
- Frase de robustez: *"recortamos no espaço para poder ir fundo no tempo num notebook comum; o
  mesmo código escala sem alteração — só muda o `region:` do config."*
- Mostre **§2.4** (mapa global das estações + retângulo do recorte).

### Bloco 3 — Medallion + Spark end-to-end · 1:15 · §3.1, §6.5
Este é o slide que prova **"ML end-to-end no Spark"** e **"boas práticas"**.
- Arquitetura **medallion**: **bronze** (CSV cru → Parquet particionado) → **silver** (long→wide,
  unidades SI, limpo) → **gold** (features de ML) → **modelo**.
- Cite as funções de aula usadas (o avaliador procura por isto):
  - **pivô** long→wide para unificar 5 elementos numa linha por estação-dia;
  - **broadcast join** para colar metadados/índice climático **sem shuffle**;
  - **Parquet particionado por ano** + `repartition`/`coalesce`.
- `pandas` aparece **só** em resultados pequenos e agregados, para plotar — **nunca** para carregar
  o corpus. (Diga isso em voz alta — é um redutor de nota evitado.)
- Mostre **§6.5** (a mesma estação nas três camadas) — é a foto do pipeline inteiro.

### Bloco 4 — Limpeza & EDA · 1:15 · §4.1, §5.2, §5.4
- **Limpeza defensiva** além do Q_FLAG da NOAA: faixa física de temperatura, consistência
  TMIN ≤ TMAX, precipitação ≥ 0 → valores ruins viram **NULL** (não derruba a estação-dia inteira;
  o Imputer trata depois).
- **Nulos por coluna (§4.1)** — agregação pequena, segura para o pandas; justifica a imputação.
- **§5.2 — taxa de geada por mês:** *"este é o sinal sazonal que o modelo precisa capturar."*
- ✂️ *(opcional)* **§5.4 — elevação vs. frequência de geada:** sanity check físico (mais alto →
  mais frio → mais geada).

### Bloco 5 — Feature engineering (gold) · 1:00 · §6.1, §6.2, §6.4
- **Funções de janela** (o coração técnico): `lag` (ontem/última semana), médias **móveis** 7/30
  dias, **acúmulo YTD** de GDD, e — crucial — **`lead` para criar o rótulo de amanhã sem
  vazamento**.
- **Criatividade que rendeu nota:** a **feature regional/vizinhança** (`region_tmin_mean`) via
  *grid* de 1° — virou a 2ª mais importante. **§6.2:** mostramos por dados que **precipitação ajuda
  só marginalmente** (ablação) — a temperatura domina.
- ✂️ *(opcional)* **§6.4 — inversão hemisférica** da estação de geada (e o gap de dados no sul do
  Brasil) — ótimo se sobrar fôlego; corte primeiro se atrasar.

### Bloco 6 — Modelagem · 1:30 · §7.1, §7.2, §7.4
O bloco que mais pesa em **"Modelagem"**. Cubra os 4 itens da rubrica:
1. **Labels:** `next_day_is_frost` (organizado via `lead`, sem vazamento).
2. **Split treino × teste:** **temporal**, não aleatório — *"split aleatório vazaria o futuro no
   treino e inflaria a métrica; treinamos nos anos anteriores e seguramos 2024."*
3. **Comparação de técnicas (§7.2):** **baseline de persistência** → LR → RandomForest → **GBT**.
   Todo modelo tem que **superar a barra do baseline**.
4. **Métrica + tuning (§7.4):** `CrossValidator` + `ParamGridBuilder` no GBT vencedor, otimizando
   **areaUnderROC**.
- Honestidade de método: a CV do Spark usa folds **aleatórios** (vaza um pouco no tempo) → serve só
  para **selecionar** hiperparâmetro; o número honesto vem do **ano de teste reservado**.

### Bloco 7 — Resultados & avaliação honesta · 1:30 · §8.1–8.5
**Este bloco é onde se ganha "Robustez" e "decisões baseadas em dados". Não corte.**
- **§8.1/8.2** — ROC/PR e matriz de confusão: **GBT tunado, ROC-AUC ≈ 0,98** no ano reservado.
- **O anti-hype (§8.4):** *"0,98 é bom demais? Sim e não."* Um **baseline de persistência**
  (`TMIN de hoje ≤ 2 °C → geada amanhã`) já vai longe — **geada do dia seguinte é, em grande parte,
  persistência térmica.** Isso mostra maturidade: você não se iludiu com a métrica.
- **§8.3 — importância das features:** TMIN de hoje, **TMIN média da vizinhança** e a noite mais
  fria da semana dominam → *"o modelo raciocina como um agrônomo."*
- **§8.5 — ajuste de limiar ao custo de negócio:** como **FN ≫ FP** (perder uma geada custa a
  lavoura), varremos o limiar de probabilidade — o ganho real do ML é **trocar um pouco de recall
  por bem mais precisão** e entregar uma **probabilidade ajustável**.

### Bloco 8 — Do modelo ao negócio · 0:45 · §8.7, §8.8
- **§8.7 — multi-horizonte D+1…D+15:** transforma "vai gear" em **dias de antecedência** — a janela
  para acionar colheita, secagem e logística.
- **§8.8 — mapa de prioridade de armazenagem:** *onde* o risco se concentra (franja noroeste/alta
  do Corn Belt) × exposição. Fecha **do problema à solução**.
- Frase de fechamento de negócio:
  > *"A pergunta que importa: **quanto da oferta de milho da região está em risco nesta janela — e
  > antes do mercado precificar?** O mapa diz **onde**, o modelo diz **quando**."*

### Bloco 9 — Limitações, Spark na prática & conclusão · 0:45 · §9.5, §10
- **Spark é escala, não mágica:** o build do gold (funções de janela) **estourou OOM** em
  `local[*]`; recuamos para `local[8]` — o trade-off **paralelismo ↔ memória** real.
- **Limitações honestas:** feature de vizinhança é grade ~1° (não vizinho-mais-próximo real);
  `lag(1)` é "linha anterior", não "dia de calendário"; modelo global único; treino em subamostra
  (métrica final no teste completo).
- **Conclusão (decore):**
  > *"O maior salto veio de **entender o problema** — a feature de vizinhança — não de empilhar
  > modelos. **Simples > complexo, mas simples ≠ simplório.**"*

---

## 4 · Mapa de cobertura da rubrica (checklist do avaliador)

Garanta que **cada** critério foi dito em voz alta. Se algum não couber, mencione-o em uma frase.

| Critério da rubrica | Onde você cobre | Frase-âncora para falar |
|---|---|---|
| **Implementação** (Spark end2end, tipos, limpo, sem abuso de Pandas) | Blocos 3, 6 | "schema explícito, Parquet particionado, pandas só no resultado agregado" |
| **Exploração** (estats, manipular/organizar, visuais) | Bloco 4 | "describe + nulos por coluna + sazonalidade da geada" |
| **Limpeza** (dados problemáticos, seleção, small data, justificar) | Blocos 2, 4 | "limpeza defensiva → NULL + recorte para gerar small data" |
| **Robustez** (simples>complexo, decisão por dados) | Blocos 5, 7, 9 | "baseline + ablação + avaliação honesta do 0,98" |
| **Modelagem** (labels, split, comparação, métrica+tuning, limitações) | Blocos 6, 7, 9 | "split temporal, 4 técnicas, CrossValidator, limitações" |
| **Comunicação** (markdown, gráficos com título/legenda, storytelling) | toda a fala | "história problema→dados→modelo→negócio" |
| **Avaliação geral** (solução completa, do problema à solução) | Blocos 1, 8 | "da disrupção de safra ao mapa de armazenagem" |
| **ML end-to-end no Spark** | Bloco 3 | "da leitura do .csv.gz ao GBT tunado, tudo em PySpark" |
| **Negócio / apresentação clara** | Blocos 1, 8 | "consumidor = trading de commodities; decisão acionável" |

---

## 5 · Perguntas prováveis do professor (e respostas curtas)

- **"Por que GBT e não algo mais simples/complexo?"** Comparamos contra baseline, LR e RF (§7.2);
  GBT venceu em dados tabulares com interações não-lineares. E mostramos que o baseline já vai longe
  — não complicamos sem ganho.
- **"AUC 0,98 não é vazamento?"** Não: split **temporal** (treina < 2024, testa = 2024) e rótulo via
  `lead` (sem olhar features do futuro). O 0,98 alto é porque geada é muito **persistente** — e nós
  dizemos isso explicitamente (§8.4).
- **"Por que `pandas` aparece?"** Só em resultados **pequenos e já agregados**, para plotar. O corpus
  inteiro nunca toca o pandas.
- **"Por que recortar a região? Não é trapaça?"** É a decisão de engenharia central: small data de
  propósito, sinal mais denso, e o **mesmo código escala** mudando uma linha de config.
- **"Limitação maior?"** Cobertura geográfica: onde a rede de estações é esparsa (sul do Brasil),
  não há TMIN. Próximo passo é dados em **grade** (ERA5/ECMWF) — "estações virtuais" globais.
- **"Por que `local[8]` e não `local[*]`?"** O gold com funções de janela estourou memória em
  `local[*]`; menos paralelismo = mais RAM por task. Trade-off real de Spark numa máquina só.

---

## 6 · Dicas de execução

- **Ensaie com cronômetro** ao menos uma vez. O ponto de corte natural se atrasar: §6.4 (inversão
  hemisférica) e §5.4 (elevação).
- **Abra o HTML já rolado** nas 7 figuras-herói (ou tenha-as em slides) — não role procurando ao
  vivo.
- **Comece pelo negócio, termine pelo negócio** (blocos 1 e 8). O miolo técnico (3–7) é a prova.
- **Diga os nomes das funções** (pivô, broadcast, window/`lag`/`lead`, `Pipeline`,
  `CrossValidator`) — o avaliador pontua "subutilização das funções de aula".
- **Não leia células de código ao vivo.** Mostre a figura, diga o "porquê", siga.
- **Frase de encerramento** (60 s finais): *"Um pipeline medallion completo, em escala de nuvem,
  terminando num classificador honesto — e, mais do que o número final, clareza sobre de onde vem e
  onde para o sinal."*

---

## 7 · Versão ultra-curta (se cortarem para ~6 min)

Mantenha **5 blocos**: Problema/negócio (1) → Recorte + medallion/Spark (2+3 fundidos) →
Modelagem com baseline e split temporal (6) → Avaliação honesta do 0,98 (7) → Mapa de negócio (8).
Figuras: **§6.5, §7.2, §8.4, §8.8**. Corte EDA detalhada, ablação e inversão hemisférica.
