# LSTM para a Mega da Virada

Projeto de Machine Learning que treina uma rede neural recorrente LSTM sobre o histórico oficial da Mega-Sena para **gerar jogos válidos e diversificados de 6 dezenas para a Mega da Virada**, combinando aprendizado de representações temporais com amostragem estocástica condicionada e rigorosa validação científica.

> **⚠️ Alerta Ético e Científico:** Sorteios lotéricos são desenhados para serem processos i.i.d. (independentes e identicamente distribuídos). **Não há garantia de que um modelo neural extraia sinal preditivo persistente nem garantia de retorno financeiro.** A geração de jogos expressa as preferências probabilísticas aprendidas pela rede a partir dos dados históricos. O objetivo do projeto é a excelência de engenharia de software (SDD), reprodutibilidade e honestidade estatística — a conclusão "sem evidência de sinal" é um resultado científico válido e esperado.

---

## 1. Especificação e Fonte de Verdade (SDD)

O projeto é desenvolvido estritamente sob a metodologia **Spec Driven Development (SDD)**:

- 📄 **Especificação Oficial Vigente (v0.2):** [`spec/spec_lstm_mega_virada_v0.2.md`](spec/spec_lstm_mega_virada_v0.2.md)  
  *Formaliza o processo operacional de geração de jogos (RG-001 a RG-012), amostragem sequencial e métricas de diversidade.*
- 🏛️ **Decisão Arquitetural:** [`spec/adr/ADR-001-geracao-de-jogos-e-diversidade.md`](spec/adr/ADR-001-geracao-de-jogos-e-diversidade.md)  
  *Documenta a escolha da amostragem autorregressiva condicionada e controle por dezenas compartilhadas (`max_shared_numbers`).*
- 📜 **Versão Anterior Preservada (v0.1):** [`spec/spec_lstm_mega_virada_v1.md`](spec/spec_lstm_mega_virada_v1.md)

Se houver qualquer divergência entre a implementação e a especificação, **a spec prevalece**.

---

## 2. Estrutura do Repositório

```text
Mega da virada LSTM/
├── spec/                                       # Especificação SDD e ADRs
│   ├── spec_lstm_mega_virada_v0.2.md          # Especificação vigente (v0.2)
│   ├── spec_lstm_mega_virada_v1.md            # Versão histórica preservada (v0.1)
│   └── adr/
│       └── ADR-001-geracao-de-jogos-e-diversidade.md
├── configs/                                    # Arquivos YAML versionados (RC-003)
│   ├── README.md                               # Guia de parâmetros de referência
│   └── experiments/                            # Configurações de treinos e ablações
├── data/
│   ├── raw/                                    # Dados brutos imutáveis
│   │   ├── megasena-download-resultados.csv    # 3.074 concursos oficiais da Caixa
│   │   └── mega_da_virada.csv                  # 17 concursos da Mega da Virada (2009–2025)
│   └── processed/                              # Datasets unificados com hash SHA-256
│       └── dataset_megasena_virada_processed.csv
├── notebooks/                                  # Ambiente interativo Jupyter
│   ├── 00_format_megasena_geral.ipynb          # 1º: Normalização do histórico Caixa
│   ├── 01_format_virada.ipynb                  # 2º: Normalização e conferência da Virada (RT-002)
│   ├── 02_inspecao_e_eda.ipynb                 # 3º: Análise Exploratória de Dados (EDA)
│   └── 03_experimentos.ipynb                   # 4º: Treinamento, métricas e geração de jogos
├── reports/                                    # Logs, auditorias e relatórios de execução
├── src/
│   └── lstm_virada/                            # Pacote de lógica de produção pura
│       ├── __init__.py
│       ├── data.py                             # Ingestão, validação RT-003 e exportação SHA-256 (T1)
│       ├── encoding.py                         # Multi-hot 61 features e janelas temporais W (T2)
│       ├── decode.py                           # Decodificação ótima de máxima verossimilhança por DP (T2)
│       ├── model.py                            # Arquitetura LSTM com 6 cabeças softmax (T4)
│       ├── train.py                            # Treinamento com pesos α por amostra (T4)
│       ├── generation.py                        # Geração estocástica de jogos e diversidade (T8)
│       ├── metrics.py                          # Baselines B0..B3 e métricas M1..M10 (T3, T5)
│       └── stats.py                            # Testes de hipótese, bootstrap BCa e veredito RV-011 (T6)
├── tests/                                      # Bateria de testes unitários com rastreabilidade (# cobre: ID)
│   ├── conftest.py
│   ├── data/
│   │   └── test_data.py                        # 18 testes automatizados (RT-001 a RT-004)
│   ├── encoding/
│   │   └── test_encoding.py                    # 15 testes automatizados (RT-005, RT-006, DP)
│   ├── model/
│   ├── metrics/
│   └── stats/
├── .gitignore
├── README.md
├── pyproject.toml
└── requirements.txt
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

## 4. Status de Implementação das Tarefas

| Tarefa | Escopo | Requisitos | Status |
|:---:|---|---|:---:|
| **T0** | Estruturação inicial do repositório, ambiente e notebooks | RF-001, RC-001, RC-002 | ✅ Concluída |
| **T1** | Ingestão, unificação da série e validação estrita de dados | RT-001, RT-002, RT-003, RT-004 | ✅ Concluída (18 testes) |
| **T7** | Evolução da SDD (v0.2) e formalização de Geração de Jogos | RG-001 a RG-012 | ✅ Concluída (Spec v0.2 + ADR-001) |
| **T2** | Codificação multi-hot e decodificação determinística (DP Top-1) | RT-005, RT-006 | ⏳ Próxima |
| **T3** | Implementação dos baselines de referência B0 a B3 | RV-001, RV-004 | ⏳ Planejada |
| **T4** | Arquitetura LSTM com 6 cabeças e treinamento com pesos $\alpha$ | RT-007 a RT-013 | ⏳ Planejada |
| **T8** | Módulo de geração de múltiplos jogos e filtro de diversidade | RG-002 a RG-008, RG-011 | ⏳ Planejada |
| **T9–T10** | Integração da geração com modelo salvo e configuração YAML | RG-001, RG-004, RG-009, RG-010 | ⏳ Planejada |
| **T11** | Suíte de testes automatizados para geração de jogos | RG-001 a RG-012 (14 cenários) | ⏳ Planejada |
| **T5–T6** | Validação Walk-Forward (2015–2025) e veredito estatístico | RV-001 a RV-014, RM-001 a RM-003 | ⏳ Planejada |
| **T12** | Integração e renderização visual no Jupyter Notebook | RG-012, RF-007 | ⏳ Planejada |
| **T13–T14** | Avaliação do portfólio de jogos e previsão prospectiva 2026 | RV-003, RG-005, RG-010 | ⏳ Planejada |

---

## 5. Preparação do Ambiente no Windows

Execute os seguintes comandos no PowerShell a partir da raiz do repositório:

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
