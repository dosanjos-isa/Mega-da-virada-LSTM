# LSTM para a Mega da Virada

Bem-vindo ao projeto **LSTM Mega da Virada**. Este é um experimento reproduzível para testar, de forma estatisticamente honesta, se existe sinal preditivo nos sorteios da Mega-Sena e da Mega da Virada, e usar esse processo para gerar jogos válidos de seis dezenas.

> **⚠️ Alerta Ético e Científico:** Sorteios lotéricos são desenhados para serem processos i.i.d. (independentes e identicamente distribuídos). **Não há garantia de que um modelo neural extraia sinal preditivo persistente nem garantia de retorno financeiro.** O objetivo do projeto é a excelência de engenharia de software (SDD), reprodutibilidade e honestidade estatística — a conclusão "sem evidência de sinal" é um resultado científico válido e esperado.

---

## Índice

1. [Começando por Aqui](#1-começando-por-aqui)
2. [Especificação e Fonte de Verdade (SDD)](#2-especificação-e-fonte-de-verdade-sdd)
3. [Estrutura do Repositório](#3-estrutura-do-repositório)
4. [Conceitos Fundamentais](#4-conceitos-fundamentais)
5. [Os Modelos](#5-os-modelos)
6. [Treinamento e Avaliação](#6-treinamento-e-avaliação)
7. [Métricas e Veredito](#7-métricas-e-veredito)
8. [Geração de Jogos](#8-geração-de-jogos)
9. [Limitações Científicas](#9-limitações-científicas)
10. [Preparação do Ambiente](#10-preparação-do-ambiente)
11. [Fluxo Rápido (Resumo)](#11-fluxo-rápido-resumo)

---

## 1. Começando por Aqui

Se você é novo no projeto, siga esta ordem:

| Passo | O que fazer | Onde |
|:---:|---|:---|
| 1 | Leia a especificação matemática e metodológica | `spec/PROJECT_SPEC.md` |
| 2 | Prepare o ambiente base (sem TensorFlow) | Seção 10.1 |
| 3 | Rode os testes de dados, encoding, splits, baselines, métricas e stats | Seção 7 |
| 4 | (Opcional) Prepare o ambiente de treino com TensorFlow | Seção 10.2 |
| 5 | Treine modelos (M1 a M5) com walk-forward | `scripts/run_treino.py` |

---

## 2. Especificação e Fonte de Verdade (SDD)

O projeto é desenvolvido estritamente sob a metodologia **Spec Driven Development (SDD)**. Se houver qualquer divergência entre a implementação e a especificação, **a spec prevalece**.

### Documento Normativo

| Documento | O que contém | Para que serve |
|:---|:---|:---|
| `spec/PROJECT_SPEC.md` | Especificação única do projeto: modelos, baselines, validação, geração, ambiente, comandos, pré-registro | Especificação normativa única |

Todo o conteúdo deste README é derivado de `spec/PROJECT_SPEC.md`. Em caso de divergência, **a spec prevalece**.

Os documentos históricos (`math_spec.md`, `pre_registration.md`, versões anteriores da spec) foram consolidados em `spec/PROJECT_SPEC.md` e **não existem mais como arquivos separados** no repositório.

---

## 3. Estrutura do Repositório

```text
Mega da virada LSTM/
├── spec/                                       # Especificação única
│   └── PROJECT_SPEC.md                         # Especificação única (vigente)
├── scripts/                                    # Scripts de execução
│   ├── activar_treino.ps1                     # Ativa .venv-treino (TensorFlow, Windows)
│   ├── activar_treino_wsl.sh                  # Define LD_LIBRARY_PATH para CUDA (WSL2)
│   ├── check_gpu.py                            # Verificação de GPU (Windows/DirectML)
│   ├── run_treino.py                           # Ponto de entrada do treino
│   ├── run_generation.py                       # Ponto de entrada da geração de jogos
│   └── configurar_venv_wsl_gpu_native.sh       # Configuração do venv WSL (referência)
├── configs/                                    # Arquivos YAML versionados (RC-003)
│   ├── README.md                               # Guia de parâmetros de referência
│   └── experiments/                            # Configurações de treinos e ablações
├── data/
│   ├── raw/                                    # Dados brutos imutáveis
│   │   ├── megasena-download-resultados.csv    # Histórico oficial da Caixa
│   │   └── mega_da_virada.csv                  # 17 concursos da Mega da Virada (2009–2025)
│   ├── manifests/                              # Manifestos com hashes SHA-256
│   │   └── manifest.json
│   └── processed/                              # Datasets unificados com hash SHA-256
│       └── dataset_megasena_virada_processed.csv
├── notebooks/                                  # Ambiente interativo Jupyter
│   ├── 00_format_megasena_geral.ipynb
│   ├── 01_format_virada.ipynb
│   ├── 02_inspecao_e_eda.ipynb
│   └── 03_experimentos.ipynb
├── src/
│   └── lstm_virada/                            # Pacote de lógica de produção pura
│       ├── __init__.py
│       ├── baselines.py                        # B0, B1, B2a, B2b, B3 (P0-07)
│       ├── data.py                             # Ingestão, validação RT-003, virada_ano, manifesto (P0-01 a P0-04)
│       ├── decode.py                           # Decodificação ótima por DP, exceções em vez de assert (P0-10)
│       ├── encoding.py                         # Multi-hot 61 features, janelas W, targets_to_draw (P0-06, P0-10)
│       ├── generation.py                       # Geração de jogos com filtros de diversidade e manifesto (P4)
│       ├── metrics.py                          # Métricas M1–M10 (P0-08)
│       ├── splits.py                           # TemporalSplitter, WalkForwardSplitter, InferenceWindowExtractor (P0-05)
│       ├── stats.py                            # Testes de hipótese, bootstrap BCa, veredito (P0-09)
│       ├── train_walkforward.py                # Orquestração do treino walk-forward (P1-06)
│       └── models/                             # Modelos probabilísticos (Fase 3)
│           ├── __init__.py
│           ├── plackett_luce.py                # M1 — Plackett–Luce marginalizado (P1-01)
│           ├── fisher.py                       # M2 — Fisher (produto condicional) (P1-02)
│           ├── positional.py                   # M3 — Produto posicional normalizado (P1-03)
│           ├── seq_pl.py                       # M4 — PL Neural Sequencial (P1-04)
│           └── lstm_seis_cabecas.py            # M5 — LSTM de 6 cabeças (M5-SET e M5-CCE) (P1-05)
├── tests/                                      # Bateria de testes unitários com rastreabilidade
│   ├── conftest.py
│   ├── data/test_data.py                       # RT-001 a RT-004, P0-01 a P0-04
│   ├── encoding/test_encoding.py               # RT-005, RT-006, P0-06, P0-10
│   ├── splits/test_splits.py                   # P0-05
│   ├── baselines/test_baselines.py             # P0-07
│   ├── metrics/test_metrics.py                 # P0-08
│   ├── stats/test_stats.py                     # P0-09
│   ├── models/test_plackett_luce.py            # P1-01
│   ├── models/test_fisher.py                   # P1-02
│   ├── models/test_positional.py               # P1-03
│   ├── models/test_seq_pl.py                   # P1-04
│   ├── models/test_lstm_seis_cabecas.py        # P1-05
│   ├── train_walkforward/test_walkforward.py   # P1-06
│   └── integration/                            # Testes de integração E2E
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt                            # Ambiente base (sem TensorFlow)
├── requirements-base.txt                       # Ambiente base (sem TensorFlow)
└── requirements-gpu.txt                        # Ambiente de treino (com TensorFlow/CUDA)
```

> **Nota:** Diretórios não versionados pelo Git (`.venv-treino/`, `.venv-treino-wsl/`, `python311/`, `artifacts/`, `__pycache__/`, `.pytest_cache/`) não aparecem nesta árvore porque estão em `.gitignore`.

---

## 4. Conceitos Fundamentais

Antes de entender os modelos, precisamos entender o que estamos tentando fazer.

### 4.1. O que é um sorteio da Mega-Sena?

Um sorteio da Mega-Sena consiste em escolher **6 dezenas distintas** de um universo de **60 números** (de 1 a 60). A ordem em que as bolas saem **não importa** para o apostador: o que importa é o conjunto final de seis números.

Por exemplo, esses dois sorteios são o mesmo jogo:

| Ordem de saída | Conjunto ordenado |
|----------------|-------------------|
| 5, 23, 1, 44, 12, 30 | 1, 5, 12, 23, 30, 44 |
| 30, 1, 12, 5, 44, 23 | 1, 5, 12, 23, 30, 44 |

Em ambos os casos, o conjunto é `{1, 5, 12, 23, 30, 44}`.

O número total de combinações possíveis é:

$$\binom{60}{6} = 50.063.860$$

Isso significa que, se todas as combinações fossem igualmente prováveis, cada conjunto teria exatamente **1 chance em 50.063.860** de ser sorteado. Isso é cerca de $0,000002\%$. Esse é o **piso absoluto**: nenhum modelo pode ter uma probabilidade maior que essa para um conjunto específico, a menos que exista algum viés real nos sorteios (o que o projeto busca testar, não assumir).

### 4.2. Modelo estatístico base

O projeto assume inicialmente que cada conjunto tem a mesma probabilidade de ser sorteado. Isso é chamado de **distribuição uniforme** ou **modelo B0**. É o "piso" contra o qual todos os outros modelos são comparados.

Se um modelo consegue produzir log-score melhor que o uniforme, isso pode indicar sinal. Se não consegue, não há evidência de que o modelo capture algo além do acaso.

---

## 5. Os Modelos

O projeto implementa modelos probabilísticos e baselines. Eles são agrupados em três famílias:

1. **M0 — Uniforme** — o piso, sem parâmetros.
2. **Modelos estáticos (M1, M2)** — atribuem um peso fixo a cada dezena, não usam histórico de concursos anteriores.
3. **Modelos neurais (M3, M4, M5)** — usam uma rede neural para produzir probabilidades condicionadas ao histórico recente.

### 5.1. M0 — Uniforme

É o modelo mais simples possível: assume que todos os $\binom{60}{6}$ conjuntos são igualmente prováveis.

$$p(S) = \frac{1}{\binom{60}{6}}$$

Não aprende nada com o histórico. Serve como **piso de referência**: se um modelo não consegue superar o uniforme, ele não tem sinal.

### 5.2. M1 — Plackett–Luce (modelo primário)

O M1 atribui um **peso** $w_i > 0$ a cada dezena $i$. A probabilidade de um conjunto $S = \{d_1, \dots, d_6\}$ é obtida somando-se a probabilidade de todas as $6! = 720$ sequências ordenadas que formam esse conjunto.

Imagine que as dezenas são bolas em uma urna, cada uma com um peso. Você retira uma dezena com probabilidade proporcional ao seu peso, depois remove-a e repete. A ordem em que você retira as dezenas **não é observada** (só vemos o conjunto final ordenado), então somamos a probabilidade de todas as ordens possíveis.

$$P_{\text{M1}}(S) = \sum_{\pi \in \operatorname{Perm}(S)} \prod_{j=1}^{6} \frac{w_{\pi_j}}{W - \sum_{l<j} w_{\pi_l}}$$

**Por que M1 é o modelo primário?**

- Tem apenas 60 parâmetros (59 graus de liberdade efetivos, devido a uma restrição de normalização $\sum_i \log w_i = 0$), o que reduz o risco de superajuste com 3.074 concursos.
- Tem uma interpretação física direta: cada dezena tem um "peso" ou "atração" relativa.
- Compartilha a mesma família probabilística com o baseline B2b, permitindo isolar se o ajuste de pesos adiciona algo além das frequências empíricas.
- Se todos $w_i = 1$, então $P_{\text{M1}}(S) = 1/\binom{60}{6} = P_{\text{M0}}(S)$ (caso de controle).

### 5.3. M2 — Fisher (sensibilidade)

O M2 também atribui um peso a cada dezena, mas usa uma fórmula diferente para combinar os pesos em uma probabilidade de conjunto:

$$P_{\text{M2}}(S) = \frac{\prod_{i \in S} w_i}{e_6(w)}, \qquad e_6(w) = \sum_{|T|=6} \prod_{i \in T} w_i$$

É uma verificação de sensibilidade: verifica se a conclusão sobre M1 depende especificamente da fórmula de Plackett–Luce ou se é mais geral.

### 5.4. M3, M4, M5 — Modelos neurais (ablações)

- **M3** — Produto posicional normalizado por $Z$. É a formalização correta da LSTM de 6 cabeças quando normalizada por $Z$, mas não modela dependência entre posições (as seis cabeças são independentes).
- **M4** — PL Neural Sequencial. Modela dependência entre posições mas tem custo de $720 \times 6$ avaliações de rede por concurso, inviável para 3.074 concursos sem orçamento computacional significativo.
- **M5** — LSTM de 6 cabeças. Mantida como ablação. Não é Plackett–Luce (não condiciona no prefixo de dezenas já escolhidas). Tem duas variantes:
  - **M5-SET** (likelihood de conjunto): treinada com $-\sum_j \log P_{t,j}(d_{t,j}) + \log Z_t$, comparável a M1/B2b por NLL de conjunto.
  - **M5-CCE** (pseudo-likelihood): treinada com $\sum_j -\log P_{t,j}(d_{t,j})$, estima marginais posicionais, não comparável com NLL de conjunto.

### 5.5. Baselines pontuais

| ID | Definição | Tipo | Tem NLL? |
|:---|:---|:---|:---:|
| **B0** | Uniforme sobre $\Omega$: $p(S) = 1/\binom{60}{6}$ | probabilístico | sim |
| **B1** | 6 dezenas mais frequentes no treino; desempate por menor dezena | pontual | não |
| **B2a** | M3 com marginais posicionais empíricas suavizadas ($\kappa = 1.0$) | probabilístico | sim |
| **B2b** | M1 com pesos proporcionais à frequência global ($w_i \propto \hat f_i$) | probabilístico | sim |
| **B3** | 6 dezenas do concurso imediatamente anterior | pontual | não |

B1 e B3 produzem **um** conjunto, não uma distribuição. Não têm log-score. São avaliados por acertos ($|\hat S \cap S|$).

---

## 6. Treinamento e Avaliação

O projeto não treina um modelo para "prever" o próximo sorteio. Ele treina modelos para estimar probabilidades sobre conjuntos de dezenas e avalia se essas probabilidades são melhores que o acaso em concursos que o modelo não viu durante o treino.

### 6.1. Walk-forward por edição da Virada

O projeto usa **walk-forward por edição da Mega da Virada**. Para cada edição $Y$ (2015–2025):

1. Treina com todos os concursos anteriores à Virada $Y$.
2. Valida com os últimos 10% dos concursos regulares do treino.
3. Avalia na Virada $Y$ (teste).

Isso garante que o modelo nunca veja o concurso que está sendo avaliado durante o treinamento.

### 6.2. Conjuntos de teste

| Conjunto | Tamanho | Uso |
|----------|---------|-----|
| **Teste A** | 300 concursos regulares (2773–3074, excluindo Viradas) | Análise confirmatória primária (H1) |
| **Teste B** | 11 Viradas (2015–2025) | Análise secundária descritiva |

O Teste A tem mais poder estatístico (mais concursos) e é a análise principal. O Teste B tem apenas 11 observações, então seu poder é limitado. Teste A e Teste B **não são independentes** para as Viradas de 2024 e 2025, porque os folds treinam com concursos do período de teste do Teste A.

### 6.3. Métrica primária: log-score de conjunto

A métrica principal é o **log-score de conjunto**:

$$\text{LS} = \frac{1}{n}\sum_{t=1}^{n} \log p(S_t)$$

onde $p(S_t)$ é a probabilidade que o modelo atribui ao conjunto observado no concurso $t$.

A diferença pareada entre modelo e baseline é:

$$\Delta_t = \log p_{\text{modelo}}(S_t) - \log p_{\text{baseline}}(S_t)$$

Se $\Delta_t > 0$, o modelo atribuiu maior probabilidade ao concurso observado que o baseline.

### 6.4. Hipóteses

| Hipótese | Comparação | Amostra | Status |
|----------|-----------|---------|--------|
| **H1** (primária) | M1 vs B0 (uniforme) | Teste A | Confirmatória |
| **S2** | M1 vs B2b | Teste A | Secundária |
| **S1** | M3 vs B2a | Teste A | Secundária |

- **H1** testa se existe qualquer desvio do acaso (viés global de frequência).
- **S2** testa se o ajuste de peso do Plackett–Luce agrega algo além das frequências empíricas.
- **S1** testa se a LSTM captura sinal temporal.

### 6.5. Correção de Holm

As hipóteses secundárias S1 e S2 são testadas múltiplas vezes. Sem correção, a probabilidade de erro Tipo I (falso positivo) aumenta com o número de testes. A correção de Holm controla a taxa de erro Tipo I família (FWER) de forma menos conservadora que Bonferroni, mantendo poder estatístico razoável.

---

## 7. Testes de Software

O projeto mantém uma suíte de testes unitários com rastreabilidade aos requisitos do spec (`# cobre: ID`). Os testes que não dependem de TensorFlow podem ser executados no ambiente base:

```powershell
pytest tests\data\ tests\encoding\ tests\splits\ tests\baselines\ tests\metrics\ tests\stats\
```

| Grupo | Requisitos cobertos | Testes |
|-------|---------------------|--------|
| `tests/data/` | RT-001 a RT-004, P0-01 a P0-04 | 18 |
| `tests/encoding/` | RT-005, RT-006, P0-06, P0-10 | 26 |
| `tests/splits/` | P0-05 | 15 |
| `tests/baselines/` | P0-07 | 14 |
| `tests/metrics/` | P0-08 | 22 |
| `tests/stats/` | P0-09 | 16 |
| `tests/models/` | P1-01 a P1-05 | 9 + 10 + 11 + 5 + 7 = 42 |
| `tests/train_walkforward/` | P1-06 | 9 |
| `tests/integration/` | E2E | 2 |

Os testes de modelos (`tests/models/`) e o treino walk-forward (`tests/train_walkforward/`) podem exigir TensorFlow dependendo da configuração.

---

## 8. Geração de Jogos

A geração de jogos é um **experimento estatístico**, não uma recomendação de apostas. O modelo produz distribuições de probabilidade, e esses são usados para amostrar conjuntos de dezenas.

**Não garante ganho.** A loteria é i.i.d. por desenho.

### 8.1. Como a geração funciona

1. O modelo treinado produz probabilidades para cada posição (cabeça 1 a cabeça 6).
2. Essas probabilidades são usadas para amostrar dezenas sem repetição, respeitando a ordem crescente ($d_1 < d_2 < \dots < d_6$).
3. Um filtro de diversidade (`max_shared_numbers`) pode ser aplicado para evitar jogos muito similares na mesma rodada.

### 8.2. Comandos de geração

```bash
# WSL2 (GPU/CUDA)
source /home/<usuario>/.venv-treino-wsl/bin/activate
source scripts/activar_treino_wsl.sh
python scripts/run_generation.py --modelo M1 --K 10 --seed 42 --diversidade 4
deactivate
```

### 8.3. Manifesto de geração

Cada geração grava um arquivo JSON em `artifacts/generations/<timestamp>_<hash>.json` com: modelo (hash), dataset (hash), config (hash), commit, seed, temperatura $T$, modelo probabilístico, $K$, diversidade, tentativas, jogos e flag de saturação. Nada é sobrescrito, permitindo reprodução e rastreabilidade.

---

## 9. Limitações Científicas

1. **Independência estocástica:** Sorteios da Mega-Sena são projetados e auditados para serem processos i.i.d. (independentes e identicamente distribuídos).
2. **Equiprobabilidade teórica:** Em um globo justo com 60 bolas, todas as $\binom{60}{6} = 50\,063\,860$ combinações possíveis possuem exatamente a mesma probabilidade teórica de ocorrência ($1 / 50\,063\,860 \approx 2 \times 10^{-8}$).
3. **Ausência de garantia de ganho:** O treinamento de modelos neurais sobre séries temporais de loterias visa capturar eventuais desvios empíricos ou artefatos mecânicos sutis de forma experimental, mas **não confere aumento comprovado de probabilidade nem garantia de ganho financeiro**.
4. **Alerta de aposta:** A geração de jogos destina-se a fins estritamente educacionais, técnicos e de pesquisa em Machine Learning.

---

## 10. Preparação do Ambiente

O projeto usa **dois ambientes separados**: um para desenvolvimento/testes (sem TensorFlow) e outro dedicado ao treino (com TensorFlow). Isso evita instalar o TensorFlow (pacote pesado) se você só quer rodar testes de dados, encoding, splitters, baselines, métricas ou estatística.

### 10.1 Ambiente Base (desenvolvimento e testes)

Use este ambiente para rodar a suíte completa de testes que **não** depende do TensorFlow:

```powershell
# 1. Criar o ambiente virtual com Python 3.11
py -3.11 -m venv .venv

# 2. Ativar o ambiente virtual
.\.venv\Scripts\Activate.ps1

# 3. Atualizar o pip
python -m pip install --upgrade pip

# 4. Instalar as dependências do projeto
pip install -r requirements.txt

# 5. Instalar o pacote lstm_virada em modo editável
pip install -e .
```

Após ativar, você pode rodar todos os testes que não dependem de TensorFlow:

```powershell
pytest tests\data\ tests\encoding\ tests\splits\ tests\baselines\ tests\metrics\ tests\stats\
```

### 10.2 Ambiente de Treino (somente quando for treinar modelos)

O treino de modelos neurais (M4 e M5) requer o TensorFlow. Use um **venv dedicado** para não poluir o ambiente base.

#### Windows (CPU/DirectML)

```powershell
# 1. Ativar o ambiente de treino
.\scripts\activar_treino.ps1

# 2. Verificar se a GPU foi detectada (opcional, recomendado)
python scripts/check_gpu.py

# 3. Executar o treino walk-forward
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 42 43 44 --window-size 20

# 4. Sair do ambiente quando terminar
deactivate
```

#### WSL2 Ubuntu 22.04 (GPU/CUDA via TensorFlow)

O WSL2 permite aceleração por GPU NVIDIA via CUDA para treinamento de modelos neurais:

```bash
# 1. Dentro do WSL, ativar o venv dedicado
source /home/<usuario>/.venv-treino-wsl/bin/activate
source scripts/activar_treino_wsl.sh   # define LD_LIBRARY_PATH para CUDA

# 2. Verificar se a GPU foi detectada pelo TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. Executar o treino walk-forward
python scripts/run_treino.py --modelo M5-SET --epocas 100 --seeds 10 --window-size 20

# 4. Sair do ambiente quando terminar
deactivate
```

O script `scripts/configurar_venv_wsl_gpu_native.sh` documenta como criar este venv do zero (instala TensorFlow + bibliotecas CUDA via pip, sem depender de `apt`).

> **Nota:** O caminho `/home/<usuario>/.venv-treino-wsl/bin/activate` é um exemplo. Substitua `<usuario>` pelo seu usuário no WSL. O ambiente deve ser criado no disco nativo do WSL para evitar lentidão com arquivos em OneDrive.

### 10.3 Verificação de GPU

Para confirmar que o TensorFlow está usando a GPU no WSL:

```bash
source /home/<usuario>/.venv-treino-wsl/bin/activate
source scripts/activar_treino_wsl.sh
python scripts/check_gpu.py
```

Saída esperada (quando GPU está disponível):

```
GPUs detectadas: 1
  - /physical_device:GPU:0
[OK] Teste de alocacao na GPU: sucesso
```

---

## 11. Fluxo Rápido (Resumo)

```powershell
# Ambiente base
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# Testar tudo que não precisa de TensorFlow
pytest tests\data\ tests\encoding\ tests\splits\ tests\baselines\ tests\metrics\ tests\stats\

# Quando quiser treinar modelos neurais (Windows)
.\scripts\activar_treino.ps1
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 42 43 44 --window-size 20
deactivate
```

---

Este documento é um guia introdutório. Para detalhes completos, consulte `spec/PROJECT_SPEC.md` (especificação matemática e metodológica vigente, incluindo o pré-registro estatístico integrado).