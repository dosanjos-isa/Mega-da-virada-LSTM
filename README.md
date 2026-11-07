# LSTM para a Mega da Virada

Bem-vindo ao projeto **LSTM Mega da Virada**. Este é um experimento reproduzível para testar, de forma estatisticamente honesta, se existe sinal preditivo nos sorteios da Mega-Sena/Mega da Virada, e utilizar os resultados desse processo para gerar jogos válidos de seis dezenas.

> **⚠️ Alerta Ético e Científico:** Sorteios lotéricos são desenhados para serem processos i.i.d. (independentes e identicamente distribuídos). **Não há garantia de que um modelo neural extraia sinal preditivo persistente nem garantia de retorno financeiro.** O objetivo do projeto é a excelência de engenharia de software (SDD), reprodutibilidade e honestidade estatística — a conclusão "sem evidência de sinal" é um resultado científico válido e esperado.

---

## 1. Começando por Aqui

Se você é novo no projeto, siga esta ordem:

| Passo | O que fazer | Onde |
|:---:|---|:---|
| 1 | Leia a especificação matemática | `spec/math_spec.md` |
| 2 | Leia o pré-registro estatístico | `spec/pre_registration.md` |
| 3 | Prepare o ambiente base (sem TensorFlow) | Seção 5.1 |
| 4 | Rode os testes de dados, encoding, splits, baselines, métricas e stats | Seção 6 |
| 5 | (Opcional) Prepare o ambiente de treino com TensorFlow | Seção 5.2 |
| 6 | Treine modelos (M1 a M5) com walk-forward | `scripts/run_treino.py` |

---

## 2. Especificação e Fonte de Verdade (SDD)

O projeto é desenvolvido estritamente sob a metodologia **Spec Driven Development (SDD)**. Se houver qualquer divergência entre a implementação e a especificação, **a spec prevalece**.

### Documentos Normativos

| Documento | O que contém | Para que serve |
|:---|:---|:---|
| `spec/math_spec.md` | Modelos M0–M5, likelihoods, normalizações, baselines, avaliação, geração | Especificação matemática principal |
| `spec/pre_registration.md` | Hipóteses H1–H7, splits Teste A/Teste B, métricas, testes, veredito | Pré-registro estatístico |
| `spec/adr/ADR-001-geracao-de-jogos-e-diversidade.md` | Geração de jogos, amostragem sequencial, filtro de diversidade | Decisão arquitetural de geração |
| `spec/adr/ADR-002-modelo-probabilistico-primario.md` | Modelo primário M1, M2 (sensibilidade), M3/M4/M5 (ablações) | Decisão de modelo |
| `spec/adr/ADR-003-likelihood-e-comparabilidade.md` | Likelihood, normalização, comparabilidade entre modelos | Decisão de comparabilidade |
| `spec/spec_lstm_mega_virada_v1.md` | Versão histórica (v0.1) | Referência arquivada |
| `spec/spec_lstm_mega_virada_v0.2.md` | Versão histórica (v0.2) | Referência arquivada |

---

## 3. Estrutura do Repositório

```
Mega da virada LSTM/
├── spec/                                       # Especificação SDD e ADRs
│   ├── math_spec.md                            # Especificação matemática vigente (v0.3)
│   ├── pre_registration.md                     # Pré-registro estatístico
│   ├── spec_lstm_mega_virada_v0.2.md          # Versão histórica preservada (v0.2)
│   ├── spec_lstm_mega_virada_v1.md            # Versão histórica preservada (v0.1)
│   └── adr/
│       ├── ADR-001-geracao-de-jogos-e-diversidade.md
│       ├── ADR-002-modelo-probabilistico-primario.md
│       └── ADR-003-likelihood-e-comparabilidade.md
├── scripts/                                    # Scripts de execução
│   ├── activar_treino.ps1                     # Ativa .venv-treino (TensorFlow, Windows)
│   ├── activar_treino_wsl.sh                  # Define LD_LIBRARY_PATH para CUDA (WSL2)
│   ├── run_treino.py                           # Ponto de entrada do treino
│   ├── run_generation.py                       # Ponto de entrada da geração de jogos
│   └── check_gpu.py                            # Verificação de GPU via DirectML
├── configs/                                    # Arquivos YAML versionados (RC-003)
│   ├── README.md                               # Guia de parâmetros de referência
│   └── experiments/                            # Configurações de treinos e ablações
├── data/
│   ├── raw/                                    # Dados brutos imutáveis
│   │   ├── megasena-download-resultados.csv    # 3.074 concursos oficiais da Caixa
│   │   └── mega_da_virada.csv                  # 17 concursos da Mega da Virada (2009–2025)
│   ├── manifests/                              # Manifestos com hashes SHA-256 (P0-01 a P0-03)
│   │   └── manifest.json
│   └── processed/                              # Datasets unificados com hash SHA-256
│       └── dataset_megasena_virada_processed.csv
├── notebooks/                                  # Ambiente interativo Jupyter
│   ├── 00_format_megasena_geral.ipynb          # 1º: Normalização do histórico Caixa
│   ├── 01_format_virada.ipynb                  # 2º: Normalização e conferência da Virada (RT-002)
│   ├── 02_inspecao_e_eda.ipynb                 # 3º: Análise Exploratória de Dados (EDA)
│   └── 03_experimentos.ipynb                   # 4º: Treinamento, métricas e geração de jogos
├── src/
│   └── lstm_virada/                            # Pacote de lógica de produção pura
│       ├── __init__.py
│       ├── data.py                             # Ingestão, validação RT-003, virada_ano, manifesto (P0-01 a P0-04)
│       ├── encoding.py                         # Multi-hot 61 features, janelas W, targets_to_draw (P0-06, P0-10)
│       ├── decode.py                           # Decodificação ótima por DP, exceções em vez de assert (P0-10)
│       ├── generation.py                        # Geração de jogos com filtros de diversidade e manifesto (P4)
│       ├── metrics.py                          # Baselines B0–B3, métricas M1–M10 (P0-08)
│       ├── splits.py                           # TemporalSplitter, WalkForwardSplitter, InferenceWindowExtractor (P0-05)
│       ├── stats.py                            # Testes de hipótese, bootstrap BCa, veredito (P0-09)
│       ├── baselines.py                        # B0, B1, B2a, B2b, B3 (P0-07)
│       └── models/                             # Modelos probabilísticos (Fase 3)
│           ├── __init__.py
│           ├── plackett_luce.py                # M1 — Plackett–Luce marginalizado (P1-01)
│           ├── fisher.py                       # M2 — Fisher (produto condicional) (P1-02)
│           ├── positional.py                   # M3 — Produto posicional normalizado (P1-03)
│           ├── seq_pl.py                       # M4 — PL Neural Sequencial (P1-04)
│           └── lstm_seis_cabecas.py            # M5 — LSTM de 6 cabeças (M5-SET e M5-CCE) (P1-05)
├── train_walkforward.py                        # Orquestração do treino walk-forward (P1-06)
├── tests/                                      # Bateria de testes unitários com rastreabilidade (# cobre: ID)
│   ├── conftest.py
│   ├── data/
│   │   └── test_data.py                        # 18 testes (RT-001 a RT-004, P0-01 a P0-04)
│   ├── encoding/
│   │   └── test_encoding.py                    # 25 testes (RT-005, RT-006, P0-06, P0-10)
│   ├── splits/
│   │   └── test_splits.py                      # 15 testes (P0-05)
│   ├── baselines/
│   │   └── test_baselines.py                   # 16 testes (P0-07)
│   ├── metrics/
│   │   └── test_metrics.py                     # 23 testes (P0-08)
│   ├── stats/
│   │   └── test_stats.py                       # 16 testes (P0-09)
│   ├── models/
│   │   ├── test_plackett_luce.py               # 9 testes (P1-01)
│   │   ├── test_fisher.py                      # 10 testes (P1-02)
│   │   ├── test_positional.py                  # 11 testes (P1-03)
│   │   ├── test_seq_pl.py                      # 5 testes (P1-04)
│   │   └── test_lstm_seis_cabecas.py           # 7 testes (P1-05)
│   └── train_walkforward/
│       └── test_walkforward.py                 # 9 testes (P1-06)
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt                            # Ambiente base (sem TensorFlow)
├── requirements-base.txt                       # Ambiente base (sem TensorFlow)
└── requirements-gpu.txt                        # Ambiente de treino (com TensorFlow/DirectML)
```

---

## 6. Conceitos Fundamentais

Antes de entender os modelos, precisamos entender o que estamos tentando fazer.

### 6.1. O que é um sorteio da Mega-Sena?

Um sorteio da Mega-Sena consiste em escolher **6 dezenas distintas** de um universo de **60 números** (de 1 a 60). A ordem em que as bolas são sorteadas não importa para o apostador: o que importa é o conjunto final de seis números.

Por exemplo, esses dois sorteios são o mesmo jogo:

| Ordem de saída | Conjunto ordenado |
|----------------|-------------------|
| 5, 23, 1, 44, 12, 30 | 1, 5, 12, 23, 30, 44 |
| 30, 1, 12, 5, 44, 23 | 1, 5, 12, 23, 30, 44 |

Em ambos os casos, o conjunto é $\{1, 5, 12, 23, 30, 44\}$.

O número total de combinações possíveis é:

$$\binom{60}{6} = 50.063.860$$

Isso significa que, se todas as combinações fossem igualmente prováveis, cada conjunto teria exatamente **1 chance em 50.063.860** de ser sorteado. Isso é cerca de 0,000002%. Esse é o **piso absoluto**: nenhum modelo pode ter uma probabilidade maior que essa para um conjunto específico, a menos que exista algum viés real nos sorteios (o que o projeto busca testar, não assumir).

### 6.2. Modelo estatístico

O projeto assume inicialmente que cada conjunto tem a mesma probabilidade de ser sorteado. Isso é chamado de **distribuição uniforme** ou **modelo B0**. É o "piso" contra o qual todos os outros modelos são comparados.

Se um modelo consegue produzir log-score melhor que o uniforme, isso pode indicar sinal. Se não consegue, não há evidência de que o modelo capture algo além do acaso.

---

## 3. Os Modelos

O projeto implementa sete modelos probabilísticos e dois baselines pontuais. Eles são agrupados em três famílias:

1. **Modelo uniforme (M0)** — o piso, sem parâmetros.
2. **Modelos estáticos (M1, M2)** — atribuem um peso fixo a cada dezena, não usam histórico de concursos anteriores.
3. **Modelos neurais (M3, M4, M5)** — usam uma rede neural para produzir probabilidades condicionadas ao histórico recente.

### 6.1. M0 — Uniforme

É o modelo mais simples possível: assume que todos os $\binom{60}{6}$ conjuntos são igualmente prováveis.

$$p(S) = \frac{1}{\binom{60}{6}}$$

Não aprende nada com o histórico. Serve como **piso de referência**: se um modelo não consegue superar o uniforme, ele não tem sinal.

### 6.2. M1 — Plackett–Luce (modelo primário)

O M1 atribui um **peso** $w_i > 0$ a cada dezena $i$. A probabilidade de um conjunto $S = \{d_1, \dots, d_6\}$ é obtida somando-se a probabilidade de todas as $6! = 720$ sequências ordenadas que formam esse conjunto.

Imagine que as dezenas são bolas em uma urna, cada uma com um peso. Você retira uma dezena com probabilidade proporcional ao seu peso, depois remove-a e repete. A ordem em que você retira as dezenas não é observada (só vemos o conjunto final ordenado), então somamos a probabilidade de todas as ordens possíveis.

**Por que M1 é o modelo primário?**

- Tem apenas 60 parâmetros (59 graus de liberdade efetivos, devido a uma restrição de normalização), o que reduz o risco de superajuste com 3.074 concursos.
- Tem uma interpretação física direta: cada dezena tem um "peso" ou "atração" relativa.
- Compartilha a mesma família probabilística com o baseline B2b, permitindo isolar se o ajuste de pesos adiciona algo além das frequências empíricas.

### 6.3. M2 — Fisher (sensibilidade)

O M2 também atribui um peso a cada dezena, mas usa uma fórmula diferente para combinar os pesos em uma probabilidade de conjunto. É uma verificação de sensibilidade: verifica se a conclusão sobre M1 depende especificamente da fórmula de Plackett–Luce ou se é mais geral.

### 6.3. B0 — Uniforme (baseline)

O uniforme não aprende nada. É o ponto de referência absoluto.

### 6.4. B1 — 6 dezenas mais frequentes

Conta quantas vezes cada dezena apareceu no histórico e escolhe as 6 mais frequentes. Não produz probabilidades, apenas um conjunto fixo. Não tem log-score.

### 6.5. B2a — Produto posicional empírico

Para cada posição ordenada (primeira, segunda, ..., sexta dezena), estima qual dezena costuma aparecer nessa posição. Suaviza as frequências para evitar zeros. Produz probabilidade de conjunto normalizada.

### 6.6. B3 — Concurso anterior

Repete as dezenas do concurso imediatamente anterior. Modelo pontual, sem NLL.

---

## 7. Treinamento e Avaliação

O projeto não treina um modelo para "prever" o próximo sorteio. Ele treina modelos para estimar probabilidades sobre conjuntos de dezenas e avalia se essas probabilidades são melhores que o acaso em concursos que o modelo não viu durante o treino.

### 7.1. Walk-forward

O projeto usa **walk-forward por edição da Mega da Virada**. Para cada edição $Y$ (2015 a 2025):

1. Treina com todos os concursos anteriores à Virada $Y$.
2. Valida com os últimos 10% dos concursos regulares do treino.
3. Avalia na Virada $Y$ (teste).

Isso garante que o modelo nunca veja o concurso que está sendo avaliado durante o treinamento.

### 7.2. Conjuntos de teste

| Conjunto | Tamanho | Uso |
|----------|---------|-----|
| Teste A | 300 concursos regulares (2773–3074, excluindo Viradas) | Análise confirmatória primária (H1) |
| Teste B | 11 Viradas (2015–2025) | Análise secundária descritiva |

O Teste A tem mais poder estatístico (mais concursos) e é a análise principal. O Teste B tem apenas 11 observações, então seu poder é limitado.

---

## 8. Métricas e Veredito

### 8.1. Log-score de conjunto

A métrica principal é o **log-score de conjunto**:

$$\text{LS} = \frac{1}{n}\sum_{t=1}^{n} \log p(S_t)$$

onde $p(S_t)$ é a probabilidade que o modelo atribui ao conjunto observado no concurso $t$.

A diferença pareada entre modelo e baseline é:

$$\Delta_t = \log p_{\text{modelo}}(S_t) - \log p_{\text{baseline}}(S_t)$$

Se $\Delta_t > 0$, o modelo atribuiu maior probabilidade ao concurso observado que o baseline.

### 8.2. Hipóteses

| Hipótese | Comparação | Amostra | Status |
|----------|-----------|---------|--------|
| H1 (primária) | M1 vs B0 (uniforme) | Teste A | Confirmatória |
| S2 | M1 vs B2b | Teste A | Secundária |
| S1 | M3 vs B2a | Teste A | Secundária |

H1 testa se existe qualquer desvio do acaso (viés global de frequência). S2 testa se o ajuste de peso do Plackett–Luce agrega algo além das frequências empíricas. S1 testa se a LSTM captura sinal temporal.

### 6.3. Correção de Holm

Como S1 e S2 são testadas múltiplas vezes, usa-se a correção de Holm para controlar a taxa de erro Tipo I (falso positivo).

---

## 9. Geração de Jogos

A geração de jogos é um **experimento estatístico**, não uma recomendação de apostas. O modelo produz distribuições de probabilidade, e esses são usados para amostrar conjuntos de dezenas.

**Não garante ganho.** A loteria é i.i.d. por desenho.

### 9.1. Como a geração funciona

1. O modelo treinado produz probabilidades para cada posição (cabeça 1 a cabeça 6).
2. Essas probabilidades são usadas para amostrar dezenas sem repetição, respeitando a ordem crescente.
3. Um filtro de diversidade pode ser aplicado para evitar jogos muito similares.

A geração uniforme (M0) também está disponível como baseline de comparação.

### 9.2. Manifesto

Cada geração grava um arquivo JSON com modelo, dataset, config, seed, diversidade e jogos gerados, permitindo reprodução e rastreabilidade.

---

## 9. Limitações Científicas

- Sorteios lotéricos são projetados para serem i.i.d.
- Não há garantia de que um modelo extraia sinal preditivo persistente.
- Não há garantia de retorno financeiro.
- Desempenho no treino não implica desempenho fora da amostra.
- O projeto testa hipótese, não garante resultado.

---

## 10. Preparação do Ambiente

O projeto usa dois ambientes separados:

### 5.1 Ambiente Base (desenvolvimento e testes)

Para testes que não dependem do TensorFlow:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 5.2 Ambiente de Treino (somente quando for treinar modelos)

O treino de modelos neurais (M4 e M5) requer o TensorFlow.

#### Windows (CPU)

```powershell
.\scripts\activar_treino.ps1
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10
deactivate
```

#### WSL2 Ubuntu 22.04 (GPU/CUDA via TensorFlow)

O WSL2 permite aceleração por GPU NVIDIA via CUDA:

```bash
source /home/<usuario>/.venvs/lstm-virada-tf221/bin/activate
source scripts/activar_treino_wsl.sh
python scripts/run_treino.py --modelo M5-SET --epocas 100 --seeds 10
deactivate
```

O script `scripts/configurar_venv_wsl_gpu_native.sh` documenta como criar este venv do zero.

---

## 9. Fluxo Rápido (Resumo)

```powershell
# Ambiente base
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# Testar tudo que não precisa de TensorFlow
pytest tests\data\ tests\encoding\ tests\splits\ tests\baselines\ tests\metrics\ tests\stats\

# Quando quiser treinar modelos neurais
.\scripts\activar_treino.ps1
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10
deactivate
```

---

Este documento é um guia introdutório. Para detalhes completos, consulte `spec/math_spec.md` (especificação matemática) e `spec/pre_registration.md` (pré-registro estatístico).