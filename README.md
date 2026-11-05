# LSTM para a Mega da Virada

Projeto de Machine Learning que treina uma rede neural recorrente LSTM sobre o histórico oficial da Mega-Sena para **gerar jogos válidos e diversificados de 6 dezenas para a Mega da Virada**, combinando aprendizado de representações temporais com amostragem estocástica condicionada e rigorosa validação científica.

> **⚠️ Alerta Ético e Científico:** Sorteios lotéricos são desenhados para serem processos i.i.d. (independentes e identicamente distribuídos). **Não há garantia de que um modelo neural extraia sinal preditivo persistente nem garantia de retorno financeiro.** A geração de jogos expressa as preferências probabilísticas aprendidas pela rede a partir dos dados históricos. O objetivo do projeto é a excelência de engenharia de software (SDD), reprodutibilidade e honestidade estatística — a conclusão "sem evidência de sinal" é um resultado científico válido e esperado.

---

## 1. Especificação e Fonte de Verdade (SDD)

O projeto é desenvolvido estritamente sob a metodologia **Spec Driven Development (SDD)**:

- 📄 **Especificação Matemática Vigente (v0.3):** [`spec/math_spec.md`](spec/math_spec.md)  
  *Define modelos probabilísticos (M0–M5), likelihoods, normalizações, baselines B0–B3, avaliação, geração e reprodutibilidade.*
- 🏛️ **Decisões Arquiteturais (ADRs):**
  - [`spec/adr/ADR-001-geracao-de-jogos-e-diversidade.md`](spec/adr/ADR-001-geracao-de-jogos-e-diversidade.md)  
    *Geração de jogos, amostragem sequencial condicionada e filtro de diversidade (`max_shared_numbers`).*
  - [`spec/adr/ADR-002-modelo-probabilistico-primario.md`](spec/adr/ADR-002-modelo-probabilistico-primario.md)  
    *Modelo primário M1 (Plackett–Luce), M2 (Fisher) como sensibilidade, M3/M4/M5 como ablações, LSTM como ablação de DL.*
  - [`spec/adr/ADR-003-likelihood-e-comparabilidade.md`](spec/adr/ADR-003-likelihood-e-comparabilidade.md)  
    *Likelihood, normalização e comparabilidade entre modelos.*
- 📝 **Pré-Registro Estatístico:** [`spec/pre_registration.md`](spec/pre_registration.md)  
  *Hipóteses H1–H7, splits Teste A/Teste B, métricas, testes estatísticos, seeds, poder, critérios de veredito.*
- 📜 **Versões Anteriores Preservadas:**
  - [`spec/spec_lstm_mega_virada_v1.md`](spec/spec_lstm_mega_virada_v1.md) (v0.1 — substituída por math_spec.md v0.3)
  - [`spec/spec_lstm_mega_virada_v0.2.md`](spec/spec_lstm_mega_virada_v0.2.md) (v0.2 — substituída por math_spec.md v0.3)

Se houver qualquer divergência entre a implementação e a especificação, **a spec prevalece**.

---

## 2. Estrutura do Repositório

```text
Mega da virada LSTM/
├── spec/                                       # Especificação SDD e ADRs
│   ├── math_spec.md                            # Especificação matemática vigente (v0.3)
│   ├── pre_registration.md                     # Pré-registro estatístico (hipóteses, splits, veredito)
│   ├── spec_lstm_mega_virada_v0.2.md          # Versão histórica preservada (v0.2)
│   ├── spec_lstm_mega_virada_v1.md            # Versão histórica preservada (v0.1)
│   └── adr/
│       ├── ADR-001-geracao-de-jogos-e-diversidade.md
│       ├── ADR-002-modelo-probabilistico-primario.md
│       └── ADR-003-likelihood-e-comparabilidade.md
├── scripts/                                    # Scripts de execução
│   ├── activar_treino.ps1                     # Cria e ativa .venv-treino (TensorFlow)
│   ├── run_treino.py                           # Ponto de entrada do treino
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
│       ├── model.py                            # Stub (arquitetura LSTM não implementada como modelo principal)
│       ├── train.py                            # Stub (treino não implementado como modelo principal)
│       ├── generation.py                        # Ausente (não implementado)
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
| **—** | Geração de jogos | `generation.py` ausente | ❌ Não iniciado |
| **—** | Notebooks 02/03 | Placeholders, config inline | ⚠️ Obsoleto |

---

## 5. Preparação do Ambiente no Windows

O projeto usa **dois ambientes separados**: um para desenvolvimento/testes (sem TensorFlow) e outro dedicado ao treino (com TensorFlow/DirectML).

### 5.1 Ambiente Base (desenvolvimento e testes)

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

### 5.2 Ambiente de Treino (somente quando for treinar modelos)

O treino de modelos neurais (M4 e M5) requer o TensorFlow. Use um **venv dedicado** para não poluir o ambiente base:

```powershell
# 1. Criar e ativar o ambiente de treino
.\scripts\activar_treino.ps1

# 2. Verificar se a GPU foi detectada (opcional)
python scripts/check_gpu.py

# 3. Executar o treino walk-forward
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10

# 4. Sair do ambiente quando terminar
deactivate
```

**Nota sobre GPU (DirectML):** O `tensorflow-directml` não tem build para Python 3.13+. Se estiver no Python 3.13, use Python 3.11/3.12 para DirectML nativo no Windows, ou WSL2 com TensorFlow CUDA. O código dos modelos não precisa de alteração — o Keras detecta o dispositivo automaticamente.

---

## 6. Qualidade de Código e Testes Automatizados

Seguindo as convenções de engenharia de software da especificação (RC-002):

```powershell
# Executar a suíte de testes unitários com pytest
pytest

# Executar com relatório detalhado de cobertura (mínimo de 85%)
pytest --cov=lstm_virada --cov-report=term-missing

# Verificação estática rigorosa de tipos
mypy src/

# Verificação de linter e formatação de código
ruff check src/ tests/
ruff format src/ tests/
```

---

## 7. Escopo Operacional Local

- O projeto é executado **exclusivamente de forma local** na máquina do desenvolvedor.
- Não há treinamento em nuvem, interface gráfica de produto, API pública ou serviço remoto.
- Os resultados, relatórios de veredito e jogos gerados são apresentados diretamente no terminal em tabelas de texto e renderizados no Jupyter Notebook.
