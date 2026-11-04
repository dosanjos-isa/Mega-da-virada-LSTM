# LSTM para a Mega da Virada

Projeto experimental que treina e avalia um modelo LSTM sobre o histórico completo de concursos da Mega-Sena, com foco na previsão e análise das dezenas da Mega da Virada.

> **⚠️ Aviso importante:** Sorteios de loteria são processos desenhados para serem independentes e estocasticamente uniformes. **Não há garantia de que um modelo LSTM consiga extrair sinal preditivo desses dados.** A hipótese de modelagem é experimental. O sucesso deste projeto consiste em construir um pipeline correto, reprodutível e estatisticamente honesto — não em acertar dezenas. A conclusão "sem evidência de sinal" é um resultado científico válido e esperado.

---

## 1. Especificação e Fonte de Verdade

A especificação oficial e fonte de verdade do projeto está documentada em:

📄 [`spec/spec_lstm_mega_virada_v1.md`](spec/spec_lstm_mega_virada_v1.md)

Todas as decisões arquiteturais, requisitos, testes e critérios de aceite derivam estritamente da especificação (SDD — *Spec Driven Development*). Se houver qualquer divergência entre a implementação e a spec, **a spec prevalece**.

---

## 2. Estrutura do Repositório

```text
Mega da virada LSTM/
├── spec/                        # Especificação oficial e ADRs
│   ├── spec_lstm_mega_virada_v1.md
│   └── adr/
├── configs/                     # Configurações versionadas dos experimentos (YAML)
│   ├── README.md
│   └── experiments/
├── data/
│   ├── raw/                     # Dados brutos imutáveis
│   │   ├── megasena-download-resultados.csv
│   │   └── mega_da_virada.csv
│   └── processed/               # Datasets processados versionados por SHA-256
├── notebooks/                   # Exploração interativa e acompanhamento
│   ├── 00_format_megasena_geral.ipynb # Normalização controlada do histórico Caixa
│   ├── 01_format_virada.ipynb   # Normalização controlada de mega_da_virada.csv (T0.1)
│   ├── 02_inspecao_e_eda.ipynb  # Inspeção e Análise Exploratória de Dados (EDA)
│   └── 03_experimentos.ipynb    # Execução controlada e visualização de resultados
├── reports/                     # Relatórios consolidados (seção 10)
├── src/
│   └── lstm_virada/             # Pacote Python com a lógica pura e reutilizável
│       ├── __init__.py
│       ├── data.py              # Ingestão, unificação e validação de qualidade (T1)
│       ├── encoding.py          # Vetorização multi-hot e janelas temporais (T2)
│       ├── model.py             # Arquitetura LSTM com 6 cabeças softmax (T4)
│       ├── train.py             # Treinamento com pesos por amostra (T4)
│       ├── decode.py            # Decodificação dinâmica d1 < ... < d6 (T2)
│       ├── metrics.py           # Baselines B0..B3 e métricas M1..M10 (T3, T5)
│       └── stats.py             # Validação estatística, bootstrap e veredito (T6)
├── tests/                       # Testes automatizados rastreáveis (# cobre: ID)
│   ├── conftest.py
│   ├── data/
│   ├── encoding/
│   ├── model/
│   ├── metrics/
│   └── stats/
├── .gitignore
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 3. Papel dos Notebooks e dos Módulos Python

- **Módulos Python (`src/lstm_virada/`):** Centralizam toda a lógica de negócio, transformações puras de dados, decodificação, treinamento e testes estatísticos. Garantem reprodutibilidade, tipagem estática e cobertura de testes automatizados.
- **Notebooks Jupyter (`notebooks/`):** Atuam como camada de experimentação interativa, análise exploratória (EDA), visualização gráfica (Matplotlib) e inspeção controlada. Não duplicam lógica de domínio; consomem os módulos de `src/lstm_virada/`.

---

## 4. Dados Brutos

Os dados brutos estão localizados no diretório `data/raw/` e são tratados como fontes imutáveis:

- `megasena-download-resultados.csv`: Histórico completo oficial de todos os concursos da Mega-Sena.
- `mega_da_virada.csv`: Histórico dos concursos específicos da Mega da Virada (2009–2025).

---

## 5. Preparação do Ambiente no Windows

Execute os seguintes comandos no PowerShell a partir da raiz do projeto:

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

## 6. Qualidade, Testes e Lint

Para assegurar o cumprimento dos critérios de engenharia de software da especificação (RC-002):

```powershell
# Execução da suíte de testes unitários com pytest
pytest

# Execução com relatório de cobertura de testes (mínimo de 85% em dados e métricas)
pytest --cov=lstm_virada --cov-report=term-missing

# Verificação estática de tipos com mypy
mypy src/

# Linter e formatação com Ruff
ruff check src/ tests/
ruff format src/ tests/
```

---

## 7. Escopo Operacional e Execução Local (v1)

- O projeto é executado **exclusivamente de forma local** na máquina do desenvolvedor.
- Não há treinamento em nuvem, interface gráfica de produto, API pública, serviço remoto ou implantação web.
- Os resultados e o relatório consolidado serão apresentados no ambiente interativo Jupyter e também impressos diretamente no terminal em formato de tabela de texto ao final da execução.
