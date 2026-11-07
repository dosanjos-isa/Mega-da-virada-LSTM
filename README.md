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
| 1 | Leia a especificação matemática | `spec/PROJECT_SPEC.md` |
| 2 | Leia o pré-registro estatístico | `spec/PROJECT_SPEC.md` |
| 3 | Prepare o ambiente base (sem TensorFlow) | Seção 10.1 |
| 4 | Rode os testes de dados, encoding, splits, baselines, métricas e stats | Seção 7 |
| 5 | (Opcional) Prepare o ambiente de treino com TensorFlow | Seção 5.2 |
| 6 | Treine modelos (M1 a M5) com walk-forward | `scripts/run_treino.py` |

---

## 2. Especificação e Fonte de Verdade (SDD)

O projeto é desenvolvido estritamente sob a metodologia **Spec Driven Development (SDD)**. Se houver qualquer divergência entre a implementação e a especificação, **a spec prevalece**.

### Documentos Normativos

| Documento | O que contém | Para que serve |
|:---|:---|:---|
| `spec/PROJECT_SPEC.md` | Especificação única do projeto: modelos, baselines, validação, geração, ambiente, comandos | Especificação normativa única |
| `spec/pre_registration.md` | Hipóteses H1–H7, splits Teste A/Teste B, métricas, testes, veredito | Pré-registro estatístico |
| `spec/adr/ADR-001-geracao-de-jogos-e-diversidade.md` | Geração de jogos, amostragem sequencial, filtro de diversidade | Decisão arquitetural de geração |
| `spec/adr/ADR-002-modelo-probabilistico-primario.md` | Modelo primário M1, M2 (sensibilidade), M3/M4/M5 (ablações) | Decisão de modelo |
| `spec/adr/ADR-003-likelihood-e-comparabilidade.md` | Likelihood, normalização, comparabilidade entre modelos | Decisão de comparabilidade |

---

## 3. Estrutura do Repositório

```
Mega da virada LSTM/
├── spec/                                       # Especificação SDD e ADRs
│   ├── PROJECT_SPEC.md                         # Especificação única (vigente)
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
│   └── check_gpu.py                            # Verificação de GPU
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
└── requirements-gpu.txt                        # Ambiente de treino (com TensorFlow/CUDA)
```

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

### 6.1. M0 — Uniforme

É o modelo mais simples possível: assume que todos os $\binom{60}{6}$ conjuntos são igualmente prováveis.

$$p(S) = \frac{1}{\binom{60}{6}}$$

Não aprende nada com o histórico. Serve como **piso de referência**: se um modelo não consegue superar o uniforme, ele não tem sinal.

### 6.2. M1 — Plackett–Luce (modelo primário)

O M1 atribui um **peso** $w_i > 0$ a cada dezena $i$. A probabilidade de um conjunto $S = \{d_1, \dots, d_6\}$ é obtida somando-se a probabilidade de todas as $6! = 720$ sequências ordenadas que formam esse conjunto.

Imagine que as dezenas são bolas em uma urna, cada uma com um peso. Você retira uma dezena com probabilidade proporcional ao seu peso, depois remove-a e repete. A ordem em que você retira as dezenas **não é observada** (só vemos o conjunto final ordenado), então somamos a probabilidade de todas as ordens possíveis.

**Por que M1 é o modelo primário?**

- Tem apenas 60 parâmetros (59 graus de liberdade efetivos, devido a uma restrição de normalização $\sum_i \log w_i = 0$), o que reduz o risco de superajuste com 3.074 concursos.
- Tem uma interpretação física direta: cada dezena tem um "peso" ou "atração" relativa.
- Compartilha a mesma família probabilística com o baseline B2b, permitindo isolar se o ajuste de pesos adiciona algo além das frequências empíricas.

### 6.3. M2 — Fisher (sensibilidade)

O M2 também atribui um peso a cada dezena, mas usa uma fórmula diferente para combinar os pesos em uma probabilidade de conjunto. É uma verificação de sensibilidade: verifica se a conclusão sobre M1 depende especificamente da fórmula de Plackett–Luce ou se é mais geral.

### 6.4. M3, M4, M5 — Modelos neurais (ablações)

- **M3** — Produto posicional normalizado por $Z$. É a formalização correta da LSTM de 6 cabeças quando normalizada por $Z$, mas não modela dependência entre posições (as seis cabeças são independentes).
- **M4** — PL Neural Sequencial. Modela dependência entre posições mas tem custo de $720 \times 6$ avaliações de rede por concurso, inviável para 3.074 concursos sem orçamento computacional significativo.
- **M5** — LSTM de 6 cabeças. Mantida como ablação. Não é Plackett–Luce (não condiciona no prefixo de dezenas já escolhidas). Treinada com CCE, é pseudo-likelihood, não comparável com NLL de conjunto.

### 6.5. Baselines pontuais

| ID | Definição | Tipo | Tem NLL? |
|:---|:---|:---|:---:|
| **B0** | Uniforme sobre $\Omega$: $p(S) = 1/\binom{60}{6}$ | probabilístico | sim |
| **B1** | 6 dezenas mais frequentes no treino; desempate por menor dezena | pontual | não |
| **B2a** | M3 com marginais posicionais empíricas suavizadas ($\kappa = 1.0$) | probabilístico | sim |
| **B2b** | M1 com pesos proporcionais à frequência global | probabilístico | sim |
| **B3** | 6 dezenas do concurso imediatamente anterior | pontual | não |

B1 e B3 produzem **um** conjunto, não uma distribuição. Não têm log-score. São avaliados por acertos ($|\hat S \cap S|$).

---

## 7. Treinamento e Avaliação

O projeto não treina um modelo para "prever" o próximo sorteio. Ele treina modelos para estimar probabilidades sobre conjuntos de dezenas e avalia se essas probabilidades são melhores que o acaso em concursos que o modelo não viu durante o treino.

### 7.1. Walk-forward por edição da Virada

O projeto usa **walk-forward por edição da Mega da Virada**. Para cada edição $Y$ (2015–2025):

1. Treina com todos os concursos anteriores à Virada $Y$.
2. Valida com os últimos 10% dos concursos regulares do treino.
3. Avalia na Virada $Y$ (teste).

Isso garante que o modelo nunca veja o concurso que está sendo avaliado durante o treinamento.

### 7.2. Conjuntos de teste

| Conjunto | Tamanho | Uso |
|----------|---------|-----|
| **Teste A** | 300 concursos regulares (2773–3074, excluindo Viradas) | Análise confirmatória primária (H1) |
| **Teste B** | 11 Viradas (2015–2025) | Análise secundária descritiva |

O Teste A tem mais poder estatístico (mais concursos) e é a análise principal. O Teste B tem apenas 11 observações, então seu poder é limitado.

### 7.3. Métrica primária: log-score de conjunto

A métrica principal é o **log-score de conjunto**:

$$\text{LS} = \frac{1}{n}\sum_{t=1}^{n} \log p(S_t)$$

onde $p(S_t)$ é a probabilidade que o modelo atribui ao conjunto observado no concurso $t$.

A diferença pareada entre modelo e baseline é:

$$\Delta_t = \log p_{\text{modelo}}(S_t) - \log p_{\text{baseline}}(S_t)$$

Se $\Delta_t > 0$, o modelo atribuiu maior probabilidade ao concurso observado que o baseline.

### 8. Hipóteses

| Hipótese | Comparação | Amostra | Status |
|----------|-----------|---------|--------|
| **H1** (primária) | M1 vs B0 (uniforme) | Teste A | Confirmatória |
| **S2** | M1 vs B2b | Teste A | Secundária |
| **S1** | M3 vs B2a | Teste A | Secundária |

- **H1** testa se existe qualquer desvio do acaso (viés global de frequência).
- **S2** testa se o ajuste de peso do Plackett–Luce agrega algo além das frequências empíricas.
- **S1** testa se a LSTM captura sinal temporal.

### 6.3. Correção de Holm

As hipóteses secundárias S1 e S2 são testadas múltiplas vezes. Sem correção, a probabilidade de erro Tipo I (falso positivo) aumenta com o número de testes. A correção de Holm controla a taxa de erro Tipo I família (FWER) de forma menos conservadora que Bonferroni, mantendo poder estatístico razoável.

---

## 10. Geração de Jogos

A geração de jogos é um **experimento estatístico**, não uma recomendação de apostas. O modelo produz distribuições de probabilidade, e esses são usados para amostrar conjuntos de dezenas.

**Não garante ganho.** A loteria é i.i.d. por desenho.

### 10.1. Como a geração funciona

1. O modelo treinado produz probabilidades para cada posição (cabeça 1 a cabeça 6).
2. Essas probabilidades são usadas para amostrar dezenas sem repetição, respeitando a ordem crescente.
3. Um filtro de diversidade pode ser aplicado para evitar jogos muito similares.

A geração uniforme (M0) também está disponível como baseline de comparação.

### 9.2. Manifesto

Cada geração grava um arquivo JSON com modelo, dataset, config, seed, diversidade e jogos gerados, permitindo reprodução e rastreabilidade.

---

## 10. Limitações Científicas

- Sorteios lotéricos são projetados para serem i.i.d.
- Não há garantia de que um modelo extraia sinal preditivo persistente.
- Não há garantia de retorno financeiro.
- Desempenho no treino não implica desempenho fora da amostra.
- O projeto testa hipótese, não garante resultado.

---

## 10. Preparação do Ambiente

O projeto usa **dois ambientes separados**: um para desenvolvimento/testes (sem TensorFlow) e outro dedicado ao treino (com TensorFlow). Isso evita instalar o TensorFlow (pacote pesado) se você só quer rodar testes de dados, encoding, splitters, baselines, métricas ou estatística.

### 11.1 Ambiente Base (desenvolvimento e testes)

Use este ambiente para rodar a suíte completa de testes que **não** depende do TensorFlow (113 testes):

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

### 11.2 Ambiente de Treino (somente quando for treinar modelos)

O treino de modelos neurais (M4 e M5) requer o TensorFlow. Use um **venv dedicado** para não poluir o ambiente base.

#### Windows (CPU)

```powershell
# 1. Criar e ativar o ambiente de treino
.\scripts\activar_treino.ps1

# 2. Verificar se a GPU foi detectada (opcional, recomendado)
python scripts/check_gpu.py

# 3. Executar o treino walk-forward
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10

# 4. Sair do ambiente quando terminar
deactivate
```

#### WSL2 Ubuntu 22.04 (GPU/CUDA via TensorFlow)

O WSL2 permite aceleração por GPU NVIDIA via CUDA para treinamento de modelos neurais:

```bash
# 1. Dentro do WSL, criar e ativar o venv dedicado (criado no disco nativo ~ para evitar lentidão do OneDrive)
source /home/matheus_pinheiro/.venvs/lstm-virada-tf221/bin/activate
source scripts/activar_treino_wsl.sh   # define LD_LIBRARY_PATH para CUDA

# 2. Verificar se a GPU foi detectada pelo TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. Executar o treino walk-forward
python scripts/run_treino.py --modelo M5-SET --epocas 100 --seeds 10

# 4. Sair do ambiente quando terminar
deactivate
```

O script `scripts/configurar_venv_wsl_gpu_native.sh` documenta como criar este venv do zero (instala TensorFlow + bibliotecas CUDA via pip, sem depender de `apt`).

### 5.3. Verificação de GPU

Para confirmar que o TensorFlow está usando a GPU:

```bash
source /home/matheus_pinheiro/.venvs/lstm-virada-tf221/bin/activate
source scripts/activar_treino_wsl.sh
python scripts/check_gpu.py
```

Saída esperada:

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

# Quando quiser treinar modelos neurais
.\scripts\activar_treino.ps1
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10
deactivate
```

---

## 11. Justificativas Metodológicas

### 11.1. Por que Plackett–Luce como modelo primário (M1)?

O modelo M1 (Plackett–Luce com marginalização sobre permutações) foi escolhido como modelo primário por cinco razões:

1. **Interpretação física direta:** Corresponde a um modelo de urna com bolas de pesos diferentes, retiradas sem reposição — modelo natural para sorteios lotéricos.
2. **Likelihood de conjunto bem definida:** A probabilidade de um conjunto $S = \{d_1,\dots,d_6\}$ é a soma sobre as $6! = 720$ sequências ordenadas que formam esse conjunto, corrigindo o fato de que a ordem real do sorteio não é observada (apenas a ordenação crescente $d_1 < \dots < d_6$ é conhecida).
3. **Comparabilidade:** Compartilha a mesma família probabilística com o baseline B2b, permitindo isolar se o ajuste de pesos adiciona algo além das frequências empíricas.
4. **Baixa complexidade:** Apenas 60 parâmetros (59 graus de liberdade efetivos, devido a uma restrição de normalização $\sum_i \log w_i = 0$), reduzindo risco de superajuste com 3.074 concursos.
5. **Independência de histórico:** É um modelo estático (não usa janela temporal), ideal para testar hipótese de viés global de frequência sem confundir com efeitos temporais.

### 6.2. Por que não M3/M4/M5 como modelos primários?

- **M3 (Produto Posicional):** Embora seja a formalização correta da LSTM de 6 cabeças quando normalizada por $Z$, não modela dependência entre posições (as seis cabeças são independentes) e, se treinada com CCE, otimiza pseudo-likelihood, não likelihood de conjunto.
- **M4 (PL Neural Sequencial):** Modela dependência entre posições mas tem custo de $720 \times 6$ avaliações de rede por concurso, inviável para 3.074 concursos sem orçamento computacional significativo.
- **M5 (LSTM de 6 cabeças):** Mantida como ablação. Não é Plackett–Luce (não condiciona no prefixo de dezenas já escolhidas). Treinada com CCE, é pseudo-likelihood, não comparável com NLL de conjunto.

### 6.3. Por que B0 (uniforme) é o piso de H1?

A hipótese primária H1 testa se existe **sinal preditivo** nos sorteios. O baseline uniforme B0 ($p(S) = 1/\binom{60}{6}$) representa a ausência absoluta de qualquer viés — ou seja, todos os $\binom{60}{6} = 50.063.860$ conjuntos são igualmente prováveis. Comparar M1 contra B0 testa diretamente a pergunta de pesquisa: **existe algum sinal preditivo além do acaso?**

Comparar M1 apenas com B2b (pesos empíricos) testaria apenas se o ajuste PL agrega valor sobre frequências empíricas, não se há desvio do acaso. Por isso, B2b é hipótese secundária (S2), não primária.

### 6.4. Por que correção de Holm?

As hipóteses secundárias S1–S5 são testadas múltiplas vezes. Sem correção, a probabilidade de erro Tipo I (falso positivo) aumenta com o número de testes. A correção de Holm controla a taxa de erro Tipo I família (FWER) de forma menos conservadora que Bonferroni, mantendo poder estatístico razoável.

### 6.5. Por que Teste A (300 concursos) e não apenas as 11 Viradas?

O Teste B (11 Viradas) tem poder estatístico muito baixo (MDE ≈ 0,75 $\sigma_d$ para log-score). O Teste A (300 concursos regulares) tem poder adequado (MDE ≈ 0,144 $\sigma_d$). O Teste A é a análise confirmatória; o Teste B é secundário e descritivo. Além disso, Teste A e Teste B **não são independentes** para as Viradas de 2024 e 2025, porque os folds treinam com concursos do período de teste do Teste A.

### 6.6. Por que geração de jogos é um experimento, não predição?

O projeto **não busca provar que a loteria é previsível**. A geração de jogos é um **registro operacional** e **experimento estatístico honesto**. Mesmo que um modelo produza jogos com distribuição estatisticamente diferente do acaso em dados históricos, isso **não implica** capacidade preditiva para sorteios futuros, que são i.i.d. (independentes e identicamente distribuídos) por desenho.

### 6.7. Por que walk-forward por edição da Virada?

O walk-forward por edição da Virada (2015–2025) testa a capacidade do modelo de generalizar para concursos futuros usando apenas dados passados. Para cada edição $Y$, treina-se com concursos estritamente anteriores à Virada $Y$ e testa-se na Virada $Y$. Isso respeita a causalidade temporal e evita vazamento de dados.

---

## 11. Preparação do Ambiente

O projeto usa **dois ambientes separados**: um para desenvolvimento/testes (sem TensorFlow) e outro dedicado ao treino (com TensorFlow). Isso evita instalar o TensorFlow (pacote pesado) se você só quer rodar testes de dados, encoding, splitters, baselines, métricas ou estatística.

### 11.1 Ambiente Base (desenvolvimento e testes)

Use este ambiente para rodar a suíte completa de testes que **não** depende do TensorFlow (113 testes):

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

# 6. Registrar o kernel Jupyter dedicado
python -m ipykernel install --user --name lstm-virada --display-name "Python (LSTM Mega da Virada)"

# 7. Iniciar o JupyterLab
jupyter lab
```

Após ativar, você pode rodar todos os testes que não dependem de TensorFlow:

```powershell
pytest tests\data\ tests\encoding\ tests\splits\ tests\baselines\ tests\metrics\ tests\stats\
```

### 11.2 Ambiente de Treino (somente quando for treinar modelos)

O treino de modelos neurais (M4 e M5) requer o TensorFlow. Use um **venv dedicado** para não poluir o ambiente base.

#### Windows (CPU)

```powershell
# 1. Criar e ativar o ambiente de treino
.\scripts\activar_treino.ps1

# 2. Verificar se a GPU foi detectada (opcional, recomendado)
python scripts/check_gpu.py

# 3. Executar o treino walk-forward
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10

# 4. Sair do ambiente quando terminar
deactivate
```

#### WSL2 Ubuntu 22.04 (GPU/CUDA via TensorFlow)

O WSL2 permite aceleração por GPU NVIDIA via CUDA para treinamento de modelos neurais:

```bash
# 1. Dentro do WSL, criar e ativar o venv dedicado (criado no disco nativo ~ para evitar lentidão do OneDrive)
source /home/matheus_pinheiro/.venvs/lstm-virada-tf221/bin/activate
source scripts/activar_treino_wsl.sh   # define LD_LIBRARY_PATH para CUDA

# 2. Verificar se a GPU foi detectada pelo TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. Executar o treino walk-forward
python scripts/run_treino.py --modelo M5-SET --epocas 100 --seeds 10 --window-size 20

# 4. Sair do ambiente quando terminar
deactivate
```

O script `scripts/configurar_venv_wsl_gpu_native.sh` documenta como criar este venv do zero (instala TensorFlow + bibliotecas CUDA via pip, sem depender de `apt`).

### 5.3. Verificação de GPU

Para confirmar que o TensorFlow está usando a GPU:

```bash
source /home/matheus_pinheiro/.venvs/lstm-virada-tf221/bin/activate
source scripts/activar_treino_wsl.sh
python scripts/check_gpu.py
```

Saída esperada:

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

# Quando quiser treinar modelos neurais
.\scripts\activar_treino.ps1
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10
deactivate
```

---

## 11. Justificativas Metodológicas

### 11.1. Por que Plackett–Luce como modelo primário (M1)?

O modelo M1 (Plackett–Luce com marginalização sobre permutações) foi escolhido como modelo primário por cinco razões:

1. **Interpretação física direta:** Corresponde a um modelo de urna com bolas de pesos diferentes, retiradas sem reposição — modelo natural para sorteios lotéricos.
2. **Likelihood de conjunto bem definida:** A probabilidade de um conjunto $S = \{d_1,\dots,d_6\}$ é a soma sobre as $6! = 720$ sequências ordenadas que formam esse conjunto, corrigindo o fato de que a ordem real do sorteio não é observada (apenas a ordenação crescente $d_1 < \dots < d_6$ é conhecida).
3. **Comparabilidade:** Compartilha a mesma família probabilística com o baseline B2b, permitindo isolar se o ajuste de pesos adiciona algo além das frequências empíricas.
4. **Baixa complexidade:** Apenas 60 parâmetros (59 graus de liberdade efetivos, devido a uma restrição de normalização $\sum_i \log w_i = 0$), reduzindo risco de superajuste com 3.074 concursos.
5. **Independência de histórico:** É um modelo estático (não usa janela temporal), ideal para testar hipótese de viés global de frequência sem confundir com efeitos temporais.

### 6.2. Por que não M3/M4/M5 como modelos primários?

- **M3 (Produto Posicional):** Embora seja a formalização correta da LSTM de 6 cabeças quando normalizada por $Z$, não modela dependência entre posições (as seis cabeças são independentes) e, se treinada com CCE, otimiza pseudo-likelihood, não likelihood de conjunto.
- **M4 (PL Neural Sequencial):** Modela dependência entre posições mas tem custo de $720 \times 6$ avaliações de rede por concurso, inviável para 3.074 concursos sem orçamento computacional significativo.
- **M5 (LSTM de 6 cabeças):** Mantida como ablação. Não é Plackett–Luce (não condiciona no prefixo de dezenas já escolhidas). Treinada com CCE, é pseudo-likelihood, não comparável com NLL de conjunto.

### 6.3. Por que B0 (uniforme) é o piso de H1?

A hipótese primária H1 testa se existe **sinal preditivo** nos sorteios. O baseline uniforme B0 ($p(S) = 1/\binom{60}{6}$) representa a ausência absoluta de qualquer viés — ou seja, todos os $\binom{60}{6} = 50.063.860$ conjuntos são igualmente prováveis. Comparar M1 contra B0 testa diretamente a pergunta de pesquisa: **existe algum sinal preditivo além do acaso?**

Comparar M1 apenas com B2b (pesos empíricos) testaria apenas se o ajuste PL agrega valor sobre frequências empíricas, não se há desvio do acaso. Por isso, B2b é hipótese secundária (S2), não primária.

### 6.4. Por que correção de Holm?

As hipóteses secundárias S1–S5 são testadas múltiplas vezes. Sem correção, a probabilidade de erro Tipo I (falso positivo) aumenta com o número de testes. A correção de Holm controla a taxa de erro Tipo I família (FWER) de forma menos conservadora que Bonferroni, mantendo poder estatístico razoável.

### 6.5. Por que Teste A (300 concursos) e não apenas as 11 Viradas?

O Teste B (11 Viradas) tem poder estatístico muito baixo (MDE ≈ 0,75 $\sigma_d$ para log-score). O Teste A (300 concursos regulares) tem poder adequado (MDE ≈ 0,144 $\sigma_d$). O Teste A é a análise confirmatória; o Teste B é secundário e descritivo. Além disso, Teste A e Teste B **não são independentes** para as Viradas de 2024 e 2025, porque os folds treinam com concursos do período de teste do Teste A.

### 6.6. Por que geração de jogos é um experimento, não predição?

O projeto **não busca provar que a loteria é previsível**. A geração de jogos é um **registro operacional** e **experimento estatístico honesto**. Mesmo que um modelo produza jogos com distribuição estatisticamente diferente do acaso em dados históricos, isso **não implica** capacidade preditiva para sorteios futuros, que são i.i.d. (independentes e identicamente distribuídos) por desenho.

### 6.7. Por que walk-forward por edição da Virada?

O walk-forward por edição da Virada (2015–2025) testa a capacidade do modelo de generalizar para concursos futuros usando apenas dados passados. Para cada edição $Y$, treina-se com concursos estritamente anteriores à Virada $Y$ e testa-se na Virada $Y$. Isso respeita a causalidade temporal e evita vazamento de dados.

---

## 11. Preparação do Ambiente

O projeto usa **dois ambientes separados**: um para desenvolvimento/testes (sem TensorFlow) e outro dedicado ao treino (com TensorFlow). Isso evita instalar o TensorFlow (pacote pesado) se você só quer rodar testes de dados, encoding, splitters, baselines, métricas ou estatística.

### 11.1 Ambiente Base (desenvolvimento e testes)

Use este ambiente para rodar a suíte completa de testes que **não** depende do TensorFlow (113 testes):

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

### 11.2 Ambiente de Treino (somente quando for treinar modelos)

O treino de modelos neurais (M4 e M5) requer o TensorFlow. Use um **venv dedicado** para não poluir o ambiente base.

#### Windows (CPU)

```powershell
# 1. Criar e ativar o ambiente de treino
.\scripts\activar_treino.ps1

# 2. Verificar se a GPU foi detectada (opcional, recomendado)
python scripts/check_gpu.py

# 3. Executar o treino walk-forward
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10

# 4. Sair do ambiente quando terminar
deactivate
```

#### WSL2 Ubuntu 22.04 (GPU/CUDA via TensorFlow)

O WSL2 permite aceleração por GPU NVIDIA via CUDA para treinamento de modelos neurais:

```bash
# 1. Dentro do WSL, criar e ativar o venv dedicado (criado no disco nativo ~ para evitar lentidão do OneDrive)
source /home/matheus_pinheiro/.venvs/lstm-virada-tf221/bin/activate
source scripts/activar_treino_wsl.sh   # define LD_LIBRARY_PATH para CUDA

# 2. Verificar se a GPU foi detectada pelo TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. Executar o treino walk-forward
python scripts/run_treino.py --modelo M5-SET --epocas 100 --seeds 10 --window-size 20

# 4. Sair do ambiente quando terminar
deactivate
```

O script `scripts/configurar_venv_wsl_gpu_native.sh` documenta como criar este venv do zero (instala TensorFlow + bibliotecas CUDA via pip, sem depender de `apt`).

### 5.3. Verificação de GPU

Para confirmar que o TensorFlow está usando a GPU:

```bash
source /home/matheus_pinheiro/.venvs/lstm-virada-tf221/bin/activate
source scripts/activar_treino_wsl.sh
python scripts/check_gpu.py
```

Saída esperada:

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

# Quando quiser treinar modelos neurais
.\scripts\activar_treino.ps1
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10
deactivate
```

---

Este documento é um guia introdutório. Para detalhes completos, consulte `spec/PROJECT_SPEC.md` (especificação matemática vigente) e `spec/pre_registration.md` (pré-registro estatístico).