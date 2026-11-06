# LSTM para a Mega da Virada

Bem-vindo ao projeto **LSTM Mega da Virada**. Este é um experimento reproduzível para testar, de forma estatisticamente honesta, se existe sinal preditivo nos sorteios da Mega-Sena/Mega da Virada.

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
| `spec/adr/ADR-001` | Geração de jogos, amostragem sequencial, filtro de diversidade | Decisão arquitetural de geração |
| `spec/adr/ADR-002` | Modelo primário M1, M2 (sensibilidade), M3/M4/M5 (ablações) | Decisão de modelo |
| `spec/adr/ADR-003` | Likelihood, normalização, comparabilidade entre modelos | Decisão de comparabilidade |
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
│       ├── stats.py                            # Testes de hipótese, bootstrap BCa, veredito (P0-09)
│       ├── splits.py                           # TemporalSplitter, WalkForwardSplitter, InferenceWindowExtractor (P0-05)
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
└── requirements-gpu.txt                        # Ambiente de treino (com tensorflow-directml)
```

---

## 3. Geração de Jogos e Mecanismo de Diversidade (RG)

O sistema foi desenhado para **desacoplar o treinamento da inferência**: um modelo treinado e salvo pode gerar novos bilhetes sob demanda sem exigir retreino a cada chamada.

### Amostragem Sequencial Condicionada

A LSTM produz distribuições softmax nas 6 cabeças posicionais. Para gerar combinações matematicamente válidas ($N1 < N2 < N3 < N4 < N5 < N6$) sem repetições:

1. $d_1$ é amostrado da cabeça 1 truncada no intervalo $[1, 55]$.
2. Cada $d_i$ subsequente é amostrado da cabeça $i$ truncada estritamente em $[d_{i-1} + 1, 60 - (6 - i)]$.
3. Todas as combinações geradas são garantidas como válidas **por construção**.

### Filtro de Diversidade (`max_shared_numbers`)

Para evitar conjuntos de apostas redundantes ao solicitar múltiplos bilhetes (ex: $K = 20$ jogos):

- Um jogo candidato só é aceito se compartilhar no máximo `max_shared_numbers` dezenas (padrão: 4) com qualquer outro jogo já gerado na mesma rodada.
- Garante variedade estratégica sem deformar as probabilidades aprendidas pela rede.

---

## 4. Status de Implementação (Fases 0 a 3)

O projeto foi desenvolvido em três fases, cada uma com um **gate** de aprovação antes de avançar para a próxima.

| Fase | Item | Escopo | Status |
|:---:|---|---|:---:|
| **Fase 0** | `spec/math_spec.md` | Modelos M0–M5, likelihoods, normalizações, baselines, avaliação | ✅ Concluída (Gate 0 aprovado) |
| **Fase 0** | `spec/adr/ADR-002` | Modelo primário M1, M2, M3/M4/M5, LSTM como ablação | ✅ Concluída |
| **Fase 0** | `spec/adr/ADR-003` | Likelihood, normalização, comparabilidade | ✅ Concluída |
| **Fase 0** | `spec/pre_registration.md` | Hipóteses H1–H7, splits, métricas, testes, veredito | ✅ Concluída |
| **Fase 1** | P0-01 a P0-04 | Proveniência, `virada_ano`, manifesto, testes invariantes | ✅ Concluída (Gate 1 aprovado) |
| **Fase 2** | P0-05 a P0-10 | Splitter, baselines, métricas, estatística, contratos | ✅ Concluída (Gate 2 aprovado) |
| **Fase 3** | P1-01 | M1 Plackett–Luce (vetores Apêndice B) | ✅ Concluído (9 testes) |
| **Fase 3** | P1-02 | M2 Fisher (vetores Apêndice B) | ✅ Concluído (10 testes) |
| **Fase 3** | P1-03 | M3 Produto Posicional (DP, modo exato) | ✅ Concluído (11 testes) |
| **Fase 3** | P1-04 | M4 PL Neural Sequencial | ✅ Concluído (5 testes) |
| **Fase 3** | P1-05 | M5 LSTM 6 cabeças (M5-SET e M5-CCE) | ✅ Concluído (7 testes) |
| **Fase 3** | P1-06 | Treino walk-forward (100 épocas, early stopping, 10 seeds) | ✅ Concluído (9 testes) |
| **Fase 4** | P4-1 a P4-4 | Inferência e geração com artefatos treinados (`generation.py`) | ✅ Concluído |

**Total de testes não-TF aprovados: 111** (5 erros pré-existentes são funções `teste_*` no código-fonte, não testes reais).

---

## 5. Preparação do Ambiente no Windows

O projeto usa **dois ambientes separados**: um para desenvolvimento/testes (sem TensorFlow) e outro dedicado ao treino (com TensorFlow/DirectML). Isso evita instalar o TensorFlow (pacote pesado) se você só quer rodar testes de dados, encoding, splitters, baselines, métricas ou estatística.

### 5.1 Ambiente Base (desenvolvimento e testes)

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
python tests\data\test_data.py
python tests\encoding\test_encoding.py
python tests\splits\test_splits.py
python tests\baselines\test_baselines.py
python tests\metrics\test_metrics.py
python tests\stats\test_stats.py
```

---

### 5.2 Ambiente de Treino (somente quando for treinar modelos)

O treino de modelos neurais (M4 e M5) requer o TensorFlow. Use um **venv dedicado** para não poluir o ambiente base:

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
source /home/<usuario>/.venv-treino-wsl/bin/activate
source scripts/activar_treino_wsl.sh   # define LD_LIBRARY_PATH para CUDA

# 2. Verificar se a GPU foi detectada pelo TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. Executar o treino walk-forward (CPU por padrão; defina CUDA_VISIBLE_DEVICES="" para forçar CPU)
python scripts/run_treino.py --modelo M5-SET --epocas 100 --seeds 10

# 4. Sair do ambiente quando terminar
deactivate
```

O script `scripts/configurar_venv_wsl_gpu_native.sh` documenta como criar este venv do zero (instala TensorFlow + bibliotecas CUDA via pip, sem depender de `apt`).

### Modelos disponíveis para treino

| Modelo | Descrição | Requer TensorFlow? |
|:---|:---|:---|
| **M1** | Plackett–Luce marginalizado (modelo primário) | Não |
| **M2** | Fisher (produto condicional, sensibilidade) | Não |
| **M3** | Produto Posicional normalizado (DP, modo exato) | Não |
| **M4** | PL Neural Sequencial | Sim (rede neural) |
| **M5-SET** | LSTM 6 cabeças com likelihood de conjunto | Sim (rede neural) |
| **M5-CCE** | LSTM 6 cabeças com pseudo-likelihood (CCE) | Sim (rede neural) |

### Nota sobre GPU (DirectML vs CUDA via WSL2)

O `tensorflow-directml` não tem build para Python 3.13+. Se estiver no Python 3.13:

- **Opção A:** Use Python 3.11 ou 3.12 para DirectML nativo no Windows.
- **Opção B:** Use WSL2 com TensorFlow CUDA (mantém Python 3.13).

O código dos modelos não precisa de alteração — o Keras detecta o dispositivo automaticamente.

---

## 5.3 Configuração de GPU no WSL2 (Recomendado para Treino)

O WSL2 permite execução de TensorFlow com suporte a GPU NVIDIA via CUDA, oferecendo desempenho significativamente superior ao CPU para treinamento de modelos neurais (M4 e M5).

### 5.3.1 Ambiente de Treino com GPU no WSL2

O ambiente dedicado para treino com GPU fica em `~/.venvs/lstm-virada-tf221/` dentro do WSL2, fora do OneDrive para evitar lentidão de sincronização.

#### Para ativar o ambiente existente:

```bash
# Dentro do WSL (Ubuntu 22.04):
source /home/matheus_pinheiro/.venvs/lstm-virada-tf221/bin/activate
source scripts/activar_treino_wsl.sh   # define LD_LIBRARY_PATH para CUDA

# Verificar se a GPU foi detectada pelo TensorFlow:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Executar treino walk-forward:
python scripts/run_treino.py --modelo M5-SET --epocas 100 --seeds 10 --window-size 20

# Sair do ambiente:
deactivate
```

#### Para criar o ambiente do zero (se necessário):

```bash
# Dentro do WSL:
bash scripts/configurar_venv_wsl_gpu_native.sh
```

Este script instala Python 3.10.12, TensorFlow 2.21.0 com suporte CUDA e todas as dependências do projeto, sem depender do `apt` (que pode estar travado no WSL).

### 5.3.2 Verificação de GPU

Para confirmar que o TensorFlow está usando a GPU:

```bash
source /home/matheus_pinheiro/.venvs/lstm-virada-tf221/bin/activate
source scripts/activar_treino_wsl.sh
python scripts/check_gpu.py
```

Saída esperada:
```
============================================================
  Verificação de GPU — Mega da Virada / WSL2 + CUDA
============================================================

TensorFlow: 2.21.0

=== Verificando GPU no sistema (nvidia-smi / libcuda) ===
[OK] GPU disponível para o sistema (nvidia-smi):
       - NVIDIA GeForce RTX 4050 Laptop GPU, 581.86, 6141MiB
[OK] Biblioteca libcuda.so.1 encontrada no sistema.

=== Verificando GPU no TensorFlow ===
[OK] 1 GPU(s) detectada(s) pelo TensorFlow:
       - /physical_device:GPU:0

=== Verificando operação na GPU ===
[OK] Teste de alocacao na GPU: sucesso

============================================================
  RESUMO DA VERIFICAÇÃO
============================================================
  1. GPU no sistema:          OK
  2. GPU no TensorFlow:       OK
  3. Operação na GPU:         OK
============================================================
  >>> AMBIENTE VALIDADO — GPU RTX 4050 pronta para treinamento.
```

---

## 6. Justificativas Metodológicas

### 6.1. Por que Plackett–Luce como modelo primário (M1)?

O modelo M1 (Plackett–Luce com marginalização sobre permutações) foi escolhido como modelo primário por cinco razões fundamentais:

1. **Interpretação física direta:** Corresponde a um modelo de urna com bolas de pesos diferentes, retiradas sem reposição — modelo natural para sorteios lotéricos.
2. **Likelihood de conjunto bem definida:** A probabilidade de um conjunto $S = \{d_1,\dots,d_6\}$ é a soma sobre as $6! = 720$ permutações possíveis, corrigindo o fato de que a ordem real do sorteio não é observada (apenas a ordenação crescente $d_1 < \dots < d_6$ é conhecida).
3. **Comparabilidade:** Compartilha a mesma família probabilística com B2b (M1 com pesos empíricos), permitindo isolar o valor do ajuste de máxima verossimilhança em relação a frequências brutas.
4. **Baixa complexidade:** Apenas 60 pesos (59 graus de liberdade efetivos após a restrição $\sum_i \log w_i = 0$), reduzindo risco de overfitting com 3.074 concursos.
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

## 7. Escopo Operacional Local

## 6. Qualidade de Código e Testes Automatizados

Seguindo as convenções de engenharia de software da especificação (RC-002):

### Executar todos os testes que não precisam de TensorFlow (ambiente base)

```powershell
pytest tests\data\ tests\encoding\ tests\splits\ tests\baselines\ tests\metrics\ tests\stats\
```

### Executar com cobertura (mínimo de 85%)

```powershell
pytest --cov=lstm_virada --cov-report=term-missing
```

### Verificação estática de tipos

```powershell
mypy src/
```

### Linter e formatação

```powershell
ruff check src/ tests/
ruff format src/ tests/
```

---

## 7. Escopo Operacional Local

- O projeto é executado **exclusivamente de forma local** na máquina do desenvolvedor.
- Não há treinamento em nuvem, interface gráfica de produto, API pública ou serviço remoto.
- Os resultados, relatórios de veredito e jogos gerados são apresentados diretamente no terminal em tabelas de texto e renderizados no Jupyter Notebook.

---

## 8. Fluxo Rápido (Resumo)

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
python scripts/check_gpu.py
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10
deactivate
```

---
