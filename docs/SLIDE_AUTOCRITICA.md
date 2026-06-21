# Slide de Autocrítica & Limitações — pronto para a banca

Conteúdo pronto para você antecipar os pontos fracos antes que o professor pergunte. Bancas
premiam quem expõe os próprios limites com lucidez — vira **ponto de robustez**, não fraqueza.

> **Onde encaixa:** Bloco 9 do guia de apresentação (~0:45, perto do fim, antes da conclusão).
> **Princípio:** cada limitação **emparelhada com a decisão/justificativa** ou com o próximo passo.
> Nunca uma limitação "solta" — sempre "limite → por que tudo bem → como melhoraria".

---

## 1 · Slide principal (o que vai na tela)

```
┌──────────────────────────────────────────────────────────────────────┐
│  AUTOCRÍTICA & LIMITAÇÕES — onde o sinal começa e onde para           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  O QUE ASSUMIMOS (de propósito)        │  COMO MELHORARIA              │
│  ───────────────────────────────       │  ──────────────────          │
│  • Treino em SUBAMOSTRA (laptop)       →  treinar no corpus inteiro    │
│    métrica final no TESTE completo        num cluster (código não muda)│
│                                                                        │
│  • CV com folds ALEATÓRIOS             →  forward-chaining temporal    │
│    (vazamento leve, só p/ selecionar)     (Spark não traz nativo)      │
│                                                                        │
│  • Vizinhança = GRADE ~1°              →  vizinho-real (haversine)     │
│    (não vizinho-mais-próximo real)                                     │
│                                                                        │
│  • Modelo GLOBAL único                 →  modelos por regime de clima  │
│                                                                        │
│  • Cobertura: GAPS onde a rede é       →  dados em GRADE (ERA5/ECMWF): │
│    esparsa (ex.: sul do Brasil)           "estações virtuais" globais  │
│                                                                        │
│  • Spark numa máquina só: OOM no gold  →  escala, não mágica:          │
│    → recuamos p/ local[8]                 memória ainda manda          │
│                                                                        │
│  Honestidade-chave: AUC 0,98 é, em boa parte, PERSISTÊNCIA térmica.    │
│  O ganho do ML = trocar recall por precisão + probabilidade ajustável. │
└──────────────────────────────────────────────────────────────────────┘
```

**Versão enxuta (se o slide ficar cheio — corte para 4 linhas):**
- Treino em subamostra · **métrica final no teste completo**.
- Vizinhança por grade 1° · não vizinho-real (próximo passo: haversine).
- Cobertura geográfica · gaps onde a rede é esparsa → ERA5/ECMWF em grade.
- Spark numa máquina só · OOM no gold → `local[8]`: **escala, não mágica**.

---

## 2 · Fala (speaker notes, ~45 s)

> *"Antes de concluir, sou honesto sobre os limites. Três são escolhas de máquina que assumimos: o
> modelo de notebook treina numa **subamostra** — mas a métrica que reporto é no **teste completo**,
> e num cluster o mesmo código treina no corpus inteiro. A validação cruzada usa folds aleatórios,
> então serve só para **escolher** hiperparâmetro; o número honesto vem do **ano reservado**.*
>
> *Dois são limites de dado: a feature de vizinhança é uma **grade de 1°**, não o vizinho-mais-
> próximo real — refinável com haversine; e onde a **rede de estações é esparsa**, como no sul do
> Brasil, simplesmente não há dado — a saída é trocar estações por **grade ERA5/ECMWF**, estações
> virtuais que cobrem o globo todo.*
>
> *E o limite mais concreto: rodar Spark **numa máquina só** estourou memória no gold, e recuamos
> para `local[8]`. Spark resolve **escala, não mágica** — a memória ainda manda.*
>
> *A leitura mais importante: o AUC de 0,98 é, em boa parte, **persistência térmica** — geada de
> amanhã segue muito o frio de hoje. O valor do ML aqui não é 'acertar muito mais', é **trocar
> recall por precisão** e entregar uma **probabilidade ajustável** ao custo do negócio."*

---

## 3 · Frase de virada (transforma autocrítica em força)

Termine o bloco com uma destas — conecta direto à rubrica ("simples > complexo", "decisão por
dados"):

- > *"Nenhum desses limites é acidental: cada um foi uma **decisão consciente** — e o maior ganho
  >  veio de **entender o problema** (a feature de vizinhança), não de empilhar modelo. Simples >
  >  complexo, mas simples ≠ simplório."*

- *(alternativa mais curta)* > *"Sabemos de onde vem o sinal **e onde ele para** — e isso vale mais
  que o número final."*

---

## 4 · Mapa: cada limitação → critério da rubrica que ela reforça

Dizer estas em voz alta marca pontos que o avaliador procura:

| Limitação exposta | Critério que você está, na verdade, **pontuando** |
|---|---|
| Subamostra + métrica no teste completo | Modelagem (rigor de avaliação) |
| CV aleatória usada só p/ selecionar | Modelagem (consciência de vazamento) |
| Vizinhança em grade + próximo passo haversine | Robustez · Criatividade |
| Gaps de cobertura → ERA5/ECMWF | Avaliação geral (caminho de implantação) |
| OOM → `local[8]` | Boas práticas Spark · decisão por dados |
| AUC 0,98 = persistência | Robustez (anti-hype, decisão por dados) |

---

## 5 · Se o professor cutucar (respostas de 1 linha)

- **"Subamostra não enfraquece o resultado?"** A métrica final é no **teste completo, ano
  reservado**; a subamostra só acelera o ajuste no notebook.
- **"Por que não corrigiu a CV temporal?"** Spark ML não traz forward-chaining nativo; o risco é
  mitigado porque a nota honesta vem do **teste fora da CV**.
- **"E a generalização para o Brasil?"** É o gap que mais nos incomoda — e por isso o próximo passo
  já está desenhado: **grade ERA5** para treinar onde não há estação.
- **"local[8] não é pouco?"** É o ponto: numa máquina só, **mais paralelismo = menos RAM por task**.
  Em cluster o mesmo código sobe sem mudança.

---

### Como exportar para slide
Copie o bloco da seção 1 para uma slide de duas colunas ("Assumimos" × "Como melhoraria"), ponha a
*frase de virada* (seção 3) no rodapé, e use a seção 2 como roteiro de fala. Mantenha **uma** slide
— autocrítica longa demais vira insegurança; o objetivo é mostrar controle, não listar defeitos.
