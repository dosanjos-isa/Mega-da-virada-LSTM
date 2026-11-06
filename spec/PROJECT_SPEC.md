# Especificação Única do Projeto — LSTM Mega da Virada

**Versão:** 1.0.0  
**Status:** Oficial — fonte única de verdade (single source of truth)  
**Data:** 06/11/2026  
**Idioma:** pt-BR  
**Repositório:** `Mega da virada LSTM`

> **Aviso Ético e Científico:** Sorteios lotéricos são desenhados para serem processos i.i.d. (independentes e identicamente distribuídos). **Não há garantia de que um modelo neural extraia sinal preditivo persistente nem garantia de retorno financeiro.** O objetivo do projeto é a excelência de engenharia de software (SDD), reprodutibilidade e honestidade estatística — a conclusão "sem evidência de sinal" é um resultado científico válido e esperado.

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Objetivo do Projeto](#2-objetivo-do-projeto)
3. [Escopo Funcional](#3-escopo-funcional)
4. [Requisitos Funcionais](#4-requisitos-funcionais)
5. [Requisitos Não Funcionais](#5-requisitos-não-funcionais)
6. [Conceitos e Terminologia](#6-conceitos-e-terminologia)
7. [Fontes e Preparação dos Dados](#7-fontes-e-preparação-dos-dados)
8. [Representação Matemática dos Sorteios](#8-representação-matemática-dos-sorteios)
9. [Espaço Amostral e Probabilidades](#9-espaco-amostral-e-probabilidades)
10. [Fundamentação dos Modelos](#10-fundamentação-dos-modelos)
11. [Baselines](#11-baselines)
12. [Modelos M1–M5](#12-modelos-m1-m5)
13. [Estratégias de Treinamento](#13-estratégias-de-treinamento)
14. [Protocolo de Validação Temporal](#15-protocolo-de-validação-temporal)
15. [Métricas e Critérios Estatísticos](#16-métricas-e-critérios-estatísticos)
16. [Testes de Sanidade](#17-testes-de-sanidade)
17. [Seleção de Modelos](#18-seleção-de-modelos)
18. [Treinamento Final](#19-treinamento-final)
19. [Inferência](#20-inferência)
20. [Geração de Jogos](#21-geração-de-jogos)
21. [Amostragem e Decodificação](#22-amostragem-e-decodificação)
22. [Diversidade de Portfólio](#23-diversidade-de-portfólio)
23. [Manifestos e Rastreabilidade](#24-manifestos-e-rastreabilidade)
24. [Arquitetura do Software](#25-arquitetura-do-software)
25. [Estrutura do Repositório](#26-estrutura-do-repositório)
26. [Ambientes e Dependências](#27-ambientes-e-dependências)
27. [Comandos de Execução](#28-comandos-de-execução)
28. [Testes de Software](#29-testes-de-software)
29. [Estado de Implementação](#30-estado-de-implementação)
30. [Limitações Científicas](#31-limitações-científicas)
31. [Riscos e Decisões Pendentes](#32-riscos-e-decisões-pendentes)
33. [Pré-Registro Científico Original](#34-pré-registro-científico-original)
34. [Emendas e Desvios Metodológicos](#35-emendas-e-desvios-metodológicos)
35. [Histórico de Decisões Arquiteturais](#36-histórico-de-decisões-arquiteturais)
36. [Critérios de Conclusão](#37-critérios-de-conclusão)

---

## 1. Visão Geral

O projeto **LSTM Mega da Virada** é um experimento reproduzível para testar, de forma estatisticamente honesta, se existe sinal preditivo nos sorteios da Mega-Sena e da Mega da Virada, e utilizar os resultados desse processo para gerar jogos válidos de seis dezenas.

O sistema opera **exclusivamente de forma local** na máquina do desenvolvedor. Não há treinamento em nuvem, interface gráfica de produto, API pública ou serviço remoto. Os resultados, relatórios de veredito e jogos gerados são apresentados diretamente no terminal em tabelas de texto e renderizados no Jupyter Notebook.

O fluxo funcional do projeto é:

```
Resultados históricos da Mega-Sena
              +
Resultados históricos da Mega da Virada
              |
              v
     Preparação e validação
              |
              v
      Análise exploratória
              |
              v
   Modelagem probabilística
              |
              v
      Treinamento e ajuste
              |
              v
    Avaliação fora da amostra
              |
              v
      Seleção de modelo
              |
              v
     Treinamento final
              |
              v
       Inferência
              |
              v
 Distribuição de probabilidades
              |
              v
    Geração de combinações
              |
              v
   Jogos de seis dezenas distintas
              |
              v
 Manifesto e rastreabilidade
```

---

## 2. Objetivo do Projeto

Desenvolver um sistema capaz de analisar os resultados históricos da Mega-Sena e da Mega da Virada, aprender padrões e distribuições estatísticas por meio de modelos probabilísticos e de Machine Learning, e utilizar os resultados desse processo para gerar jogos válidos de seis dezenas para a Mega da Virada.

O produto final é a **geração de jogos**. O treinamento dos modelos, a análise histórica, a avaliação estatística e a inferência são etapas que sustentam esse objetivo.

### 2.1. O que o sistema deve entregar

- Utilizar os resultados históricos disponíveis.
- Construir representações adequadas dos sorteios.
- Treinar e avaliar modelos probabilísticos e neurais.
- Estimar probabilidades associadas às dezenas e aos conjuntos.
- Gerar combinações válidas de seis dezenas distintas entre 1 e 60.
- Permitir geração de múltiplos jogos.
- Controlar a diversidade entre os jogos gerados, quando solicitado.
- Registrar o modelo, os parâmetros e os dados utilizados.
- Produzir manifestos que permitam rastrear e reproduzir uma geração.
- Permitir comparar diferentes estratégias de geração.

Não prometa que os jogos gerados terão maior chance real de ganhar. A loteria é um processo aleatório. O projeto investiga se informações históricas podem produzir modelos estatisticamente úteis e se existe evidência de desempenho fora da amostra. A geração uniforme deve permanecer como referência fundamental.

---

## 3. Escopo Funcional

### 3.1. Em escopo

- Preparação e validação dos dados históricos da Mega-Sena e Mega da Virada.
- Codificação e decodificação dos sorteios (vetor multi-hot de 61 features, janelas W).
- Divisão temporal dos dados (walk-forward por edição da Virada).
- Modelos probabilísticos (M0–M3, M5-SET) e neurais (M3, M4, M5).
- Baselines (B0–B3).
- Treinamento walk-forward com log-probabilidade fora da amostra.
- Métricas e estatísticas (log-score de conjunto, Wilcoxon, bootstrap BCa, correção de Holm).
- Inferência e geração de combinações com artefatos treinados.
- Salvamento e carregamento de artefatos em formato `.keras` e `.json`.
- Manifestos de geração com hash SHA-256.
- Testes unitários e de integração.

### 3.2. Fora de escopo

- Estratégias de apostas ou recomendação de retorno financeiro.
- Provar que a loteria é previsível.
- Interface gráfica de produto, API pública, serviço remoto ou treinamento em nuvem.
- Interface web/mobile ou publicação/implantação do modelo.

---

## 4. Requisitos Funcionais

O projeto adota a metodologia **Spec Driven Development (SDD)**. Todo requisito tem ID rastreável com prefixo de categoria (`RF-` Funcionais, `RT-` Técnicos de Dados/Modelo, `RG-` Geração de Jogos, `RM-` Métricas, `RV-` Validação Estatística, `RC-` Convenções de Código) e critério de aceite verificável.

Os requisitos são rastreados através de comentários no código-fonte e nos testes (exemplo: `# cobre: RV-004`).

### 4.1. Dados (RT-001 a RT-004)

| ID | Requisito | Critério de Aceite |
|----|-----------|--------------------|
| RT-001 | Série única, ordenada por concurso. As Viradas já integram o histórico (`is_virada = 1`) e não são duplicadas. | Pipeline de ingestão produz dataset unificado sem duplicatas. |
| RT-002 | Concursos da Mega da Virada reconciliados contra a base da Caixa; registros sem número preenchidos pela data de 31/12. | Dezenas da Virada coincidem exatamente com o concurso oficial da Caixa. |
| RT-003 | Validações de integridade interrompem o pipeline em caso de falha: exatamente 6 dezenas distintas, intervalo [1, 60], concursos estritamente contínuos, datas crescentes, sem duplicatas e nulos. | `DataValidationError` lançado em qualquer violação. |
| RT-004 | Dados brutos são imutáveis. Dataset processado versionado por hash SHA-256, registrado em cada execução. | Manifesto JSON gerado ao lado do dataset processado. |

### 4.2. Codificação e Janelas Temporais (RT-005 a RT-006)

| ID | Requisito | Critério de Aceite |
|----|-----------|--------------------|
| RT-005 | Codificação e decodificação são funções puras e determinísticas, com testes de ida e volta. | `decode(encode(S)) == S` para todo sorteio válido. |
| RT-006 | Nenhuma janela pode conter o concurso-alvo nem concursos posteriores a ele. Teste de vazamento temporal é obrigatório. | `TemporalSplitter` rejeita `as_of_concurso` maior que o último disponível. |

### 4.3. Treinamento com Pesos (RT-007 a RT-013)

| ID | Requisito | Critério de Aceite |
|----|-----------|--------------------|
| RT-007 | Peso $w_i = 1.0$ se o alvo é Mega da Virada; $w_i = \alpha$ se o alvo é concurso regular, com $0 < \alpha \le 1$. Padrão $\alpha = 1.0$. | Amostra de Virada recebe peso 1.0; amostra regular recebe $\alpha$. |
| RT-008 | Perda por amostra é a soma das entropias cruzadas categóricas esparsas das 6 cabeças. Perda do lote é média ponderada $\sum(w_i L_i) / \sum w_i$. | Implementação de perda ponderada no loop de treinamento. |
| RT-009 | Pesos aplicados somente no treino. Validação e teste utilizam métricas sem peso. | Validação e avaliação usam $w_i = 1.0$ para todas as amostras. |
| RT-010 | Participação efetiva das Viradas no peso total deve ser reportada ($\Sigma w_{\text{virada}} / \Sigma w$). | Relatório de treinamento inclui $N_{\text{eff}}$. |
| RT-011 | Grade de busca de hiperparâmetros limitada a 20 combinações ($W \times \alpha$). | Pelo menos 20 configurações testadas ou emenda registrada. |
| RT-012 | Embaralhamento restrito a lotes de treino já formados. Amostras temporais jamais misturadas. | `shuffle = "batches_only"`; nunca misturar treino/validação/teste. |
| RT-013 | Hiperparâmetros fixados antes do teste ou ajustados via validação aninhada temporal estrita. | Hiperparâmetros escolhidos usando apenas dados anteriores ao alvo. |

### 4.4. Geração de Jogos (RG-001 a RG-012)

| ID | Requisito | Critério de Aceite |
|----|-----------|--------------------|
| RG-001 | Gerar jogos utilizando um modelo LSTM treinado previamente carregado de disco, sem exigir retreinamento da rede a cada execução de inferência. | Carrega modelo `.keras` e gera jogos em < 5s para até 50 jogos. |
| RG-002 | Cada jogo gerado deve conter exatamente seis dezenas distintas no intervalo fechado [1, 60]. | `len(jogo) == 6`, `len(set(jogo)) == 6`, `all(1 <= d <= 60 for d in jogo)`. |
| RG-003 | As dezenas de cada jogo devem ser estritamente ordenadas de forma crescente ($N1 < N2 < ... < N6$). | `all(jogo[i] < jogo[i+1] for i in range(5))` retorna `True` para 100% dos jogos. |
| RG-004 | A quantidade de jogos a gerar ($K$) deve ser configurável via parâmetro e arquivo YAML. | `K=5` produz 5 jogos; `K=20` produz 20 jogos válidos. |
| RG-005 | O processo de geração deve utilizar diretamente as probabilidades produzidas pelas seis cabeças softmax da rede neural. | Amostragem utiliza as matrizes de saída da LSTM ponderadas pela temperatura configurada. |
| RG-006 | O sistema deve impedir a geração de jogos duplicados na mesma rodada. | `len(set(jogos)) == K`. |
| RG-007 | O processo de geração deve permitir controle estrito por semente aleatória (`seed`). | Mesma seed, mesmo modelo e mesma configuração geram a mesma lista de jogos. |
| RG-008 | O sistema deve oferecer mecanismo configurável de diversidade baseado no limite máximo de dezenas compartilhadas (`max_shared_numbers`). | Com `max_shared_numbers = 4`, nenhum par de jogos compartilha 5 ou 6 dezenas. |
| RG-009 | O sistema deve registrar metadados completos de auditoria para cada geração de jogos. | Artefato JSON com timestamp, hash do modelo, hash do dataset, seed, parâmetros de diversidade e jogos. |
| RG-010 | O sistema deve permitir reproduzir com exatidão qualquer geração anterior informando os mesmos metadados e seed. | Teste de idempotência confirma reprodução exata dos bilhetes. |
| RG-011 | O sistema deve validar matematicamente todos os jogos antes de apresentá-los ou persisti-los, levantando exceção caso alguma combinação seja inválida. | Função de validação pura rejeita jogos inválidos. |
| RG-012 | O sistema deve apresentar os jogos formatados de maneira legível no terminal e no Jupyter Notebook. | Formato `JOGO XX: D1 - D2 - D3 - D4 - D5 - D6`. |

---

## 5. Requisitos Não Funcionais

| ID | Requisito | Detalhe |
|----|-----------|---------|
| RC-001 | Python 3.11, TensorFlow 2.x / Keras | Ambiente de execução |
| RC-002 | Tipagem estática (`mypy`), lint e formatação (`ruff`), testes (`pytest`) com cobertura mínima de 85% em dados e métricas | Qualidade de código |
| RC-003 | Configuração em arquivos YAML versionados (sem constantes mágicas no código) | Configurabilidade |
| RC-004 | Funções puras e isoladas para codificação, decodificação, amostragem e validação de jogos. Efeitos colaterais (disco e rede) isolados em módulos de I/O. | Testabilidade |
| RC-005 | Apresentação dos jogos e relatórios no terminal e em tabelas do Jupyter Notebook | Interface local |
| RC-006 | Logs, mensagens e docstrings em português; nomes de identificadores em inglês | Convenção de nomenclatura |

---

## 6. Conceitos e Terminologia

| Conceito | Definição |
|----------|-----------|
| **Conjunto** | 6 dezenas distintas, em ordem crescente (representação canônica). $S = \{d_1, \dots, d_6\}$ com $d_1 < \dots < d_6$. |
| **Sequência** | Ordem de escolha das 6 dezenas; **não observada** no dataset (o dataset armazena apenas a ordenação crescente). |
| **Espaço amostral ($\Omega$)** | Conjunto de todas as combinações válidas de 6 dezenas entre 1 e 60. $|\Omega| = \binom{60}{6} = 50\,063\,860$. |
| **Likelihood de conjunto** | $P(S)$, normalizada sobre $\Omega$: $\sum_{S \in \Omega} P(S) = 1$. |
| **Pseudo-likelihood** | Soma de CCEs por posição; não é $P(S)$. Não comparável com likelihood de conjunto. |
| **Normalização** | Garantir que $\sum_S p(S) = 1$ sobre $\Omega$. |
| **Modo** | Conjunto $S$ de maior $p(S)$. |
| **Walk-forward** | Treinar apenas com concursos estritamente anteriores ao alvo. |
| **Teste A** | 300 concursos regulares 2773–3074 (excluindo Viradas 2810 e 2955). |
| **Teste B** | 11 Viradas 2015–2025, uma por fold walk-forward. |
| **Delta ($\Delta_t$)** | Diferença pareada de log-score por concurso: $\Delta_t = \log p_{\text{modelo}}(S_t) - \log p_{\text{baseline}}(S_t)$. |
| **Virada** | Concurso especial da Mega da Virada, ocorrido anualmente desde 2009. |

---

## 7. Fontes e Preparação dos Dados

### 7.1. Fontes históricas

| Arquivo | Conteúdo | Uso |
|---------|----------|-----|
| `data/raw/megasena-download-resultados.csv` | Histórico oficial de todos os concursos regulares e especiais da Caixa (1 a $N$) | Série temporal completa de treino, validação e inferência |
| `data/raw/mega_da_virada.csv` | Registros específicos dos concursos da Mega da Virada de 2009 a 2025 | Identificação das Viradas (`is_virada = 1`) e validação temporal walk-forward |
| `data/processed/dataset_megasena_virada_processed.csv` | Dataset consolidado, ordenado e validado | Fonte de entrada única para janelamento, treino e inferência |

### 7.2. Requisitos de dados

- **RT-001**: Série temporal única, estritamente ordenada por número de concurso. As Viradas já integram o histórico e não são duplicadas: os concursos correspondentes apenas recebem `is_virada = 1` e os regulares `is_virada = 0`.
- **RT-002**: Concursos da Mega da Virada reconciliados contra a base da Caixa. Registros sem número de concurso (ex: 2023 e 2024) reconciliados pela data de 31/12 e dezenas sorteadas.
- **RT-003**: Validações de integridade: exatamente 6 dezenas distintas por concurso; dezenas em $[1, 60]$; concursos estritamente contínuos e sem lacunas (1 a $N$); datas em ordem cronológica; ausência de duplicatas e nulos.
- **RT-004**: Dados brutos em `data/raw/` são imutáveis. Dataset processado em `data/processed/` versionado com hash SHA-256 obrigatório registrado em cada execução.

### 7.3. Rotulagem temporal das Viradas

O campo `ano` (ano civil da data do sorteio) é inadequado para walk-forward porque a Virada de 2025 ocorreu em 2026-01-01 (concurso 2955) e foi rotulada como `ano = 2026`. Além disso, as Viradas de 2009–2014 ocorreram em 30/12, não em 31/12.

O dataset deve conter uma coluna explícita **`virada_ano`** (edição da Virada, 2009–2025), independente do ano civil:

| `virada_ano` | Concurso | Data | Observação |
|---|---|---|---|
| 2009 | 1140 | 2009-12-30 | exceção de data |
| 2010 | 1245 | 2010-12-30 | exceção de data |
| 2011 | 1350 | 2011-12-30 | exceção de data |
| 2012 | 1455 | 2012-12-30 | exceção de data |
| 2013 | 1560 | 2013-12-30 | exceção de data |
| 2014 | 1665 | 2014-12-30 | exceção de data |
| 2015 | 1775 | 2015-12-31 | — |
| 2016 | 1890 | 2016-12-31 | — |
| 2017 | 2000 | 2017-12-31 | — |
| 2018 | 2110 | 2018-12-31 | — |
| 2019 | 2220 | 2019-12-31 | — |
| 2020 | 2330 | 2020-12-31 | — |
| 2021 | 2440 | 2021-12-31 | — |
| 2022 | 2550 | 2022-12-31 | — |
| 2023 | 2670 | 2023-12-31 | — |
| 2024 | 2810 | 2024-12-31 | — |
| 2025 | 2955 | 2026-01-01 | exceção de data; edição 2025 |

### 7.3. Problema de proveniência (conhecido)

O notebook `00_format_megasena_geral.ipynb` registra um hash inicial `088db01f...` para o arquivo bruto, mas o arquivo atual tem hash `043a16a7...` e já está normalizado. O commit `8efe765` contém a versão normalizada; o arquivo original não está no Git. Até que isso seja resolvido, nenhuma afirmação confirmatória sobre proveniência pode ser feita.

---

## 8. Representação Matemática dos Sorteios

Cada concurso é representado pelas seis dezenas ordenadas $d_1 < d_2 < \dots < d_6$. Essa ordenação é uma **ordenação canônica do conjunto**, não a ordem em que as bolas foram fisicamente sorteadas.

A ordenação crescente é uma bijeção entre o conjunto $S$ e a tupla observada $(d_1, \dots, d_6)$. Portanto, a probabilidade de observar essa tupla é exatamente $P_{\text{M1}}(S)$; não é preciso modelar a ordem real do sorteio, apenas somar sobre as ordens compatíveis (marginalização sobre permutações).

---

## 9. Espaço Amostral e Probabilidades

O espaço amostral é:

$$
\Omega = \{S \subset \{1,\dots,60\} : |S| = 6\}, \qquad |\Omega| = \binom{60}{6} = 50\,063\,860.
$$

Uma **distribuição válida** sobre $\Omega$ é qualquer $p(S) \ge 0$ com $\sum_{S \in \Omega} p(S) = 1$.

A probabilidade uniforme de uma combinação específica é:

$$
p_{\text{M0}}(S) = \frac{1}{\binom{60}{6}}.
$$

É o piso absoluto. Todo modelo deve ser comparado a M0, e nenhum resultado é interpretável sem essa referência.

---

## 10. Fundamentação dos Modelos

### 10.1. Requisitos de comparabilidade (ADR-003)

Para que duas distribuições possam ser comparadas por log-score/NLL:

- **MC-01**: Ambas devem estar definidas no mesmo $\Omega$ (conjuntos de 6 dezenas).
- **MC-02**: Ambas devem estar normalizadas: $\sum_{S \in \Omega} p(S) = 1$.
- **MC-03**: Ambas devem ser avaliadas nos mesmos concursos, com o mesmo pré-processamento e a mesma política de suavização.
- **MC-04**: A unidade de análise é o **concurso**, não a dezena. As 60 dezenas de um concurso são dependentes.
- **MC-05**: Comparações de NLL são **pareadas por concurso**: $\Delta_t = \log p_{\text{modelo}}(S_t) - \log p_{\text{baseline}}(S_t)$.
- **MC-06**: Modelos pontuais (B1, B3, decodificação top-1) **não** têm NLL. Eles são avaliados por acertos e métricas de portfólio. Se uma NLL for reportada para um modelo pontual, ela deve ser explicitamente rotulada como "ponto suavizado" e não entra no teste principal.
- **MC-07**: Pseudo-likelihoods (soma de CCEs por posição) só podem ser comparadas com outras pseudo-likelihoods da mesma família posicional.

### 10.2. Matriz de comparabilidade

| Modelo | Normalizado? | Likelihood de conjunto? | Pode entrar em teste de log-score? |
|--------|-------------|------------------------|------------------------------------|
| M0 | sim | sim | como piso |
| M1 | sim | sim | **sim** |
| M2 | sim | sim | sim (sensibilidade) |
| M3 | sim (com $Z$) | sim | sim |
| M4 | sim | sim | exploratório |
| M5-SET | sim (com $Z$) | sim | sim |
| M5-CCE | não (pseudo) | não | não |
| B0 | sim | sim | baseline |
| B1 | não (pontual) | não | baseline de acertos |
| B2a | sim (com $Z$) | sim | baseline |
| B2b | sim | sim | baseline |
| B3 | não (pontual) | não | baseline de acertos |

---

## 11. Baselines

| ID | Definição | Tipo | Tem NLL? |
|----|-----------|------|----------|
| **B0** | Uniforme sobre $\Omega$: $p(S) = 1/\binom{60}{6}$ | probabilístico | sim |
| **B1** | 6 dezenas mais frequentes no treino; desempate por menor dezena | pontual | não |
| **B2a** | M3 com marginais posicionais empíricas suavizadas ($\kappa = 1.0$) | probabilístico | sim |
| **B2b** | M1 com pesos proporcionais à frequência global ($w_i \propto \hat f_i$) | probabilístico | sim |
| **B3** | 6 dezenas do concurso imediatamente anterior | pontual | não |

### 11.1. B2a — produto posicional com suavização

Para cada posição $j$ e dezena $c \in \operatorname{supp}_j = \{j, \dots, 60-6+j\}$:

$$
\hat P_j(c) = \frac{n_{j,c} + \kappa}{N_{\text{tr}} + \kappa \cdot 55},
\qquad
\hat P_j(c) = 0 \text{ se } c \notin \operatorname{supp}_j.
$$

O denominador usa 55 porque $|\operatorname{supp}_j| = 55$ para todo $j$. A suavização é **obrigatória**: sem ela, uma dezena nunca observada em uma posição teria probabilidade zero e a NLL de um conjunto de teste seria infinita.

- **Suavização primária:** $\kappa = 1.0$ (Laplace).
- **Sensibilidade:** $\kappa \in \{0.1; 0.5; 2.0\}$.

### 11.2. B2b — PL com pesos empíricos

$$
w_i \propto \hat f_i = \frac{\#\{t \in \text{treino} : i \in S_t\}}{N_{\text{tr}}},
\qquad
P_{\text{B2b}} = P_{\text{M1}}(w).
$$

Como todas as 60 dezenas aparecem em algum concurso, todos os $w_i > 0$ e a likelihood é finita. B2b é o baseline **da mesma família** para M1, porque compartilha a mesma forma probabilística e isola a diferença entre "usar a frequência diretamente" e "ajustar os pesos pela verossimilhança PL". B2b **não** é o null de "não há sinal": ele já pode conter desvio de frequência. O null de sinal é B0 (uniforme).

### 11.3. B1 e B3 não têm NLL

B1 e B3 produzem **um** conjunto, não uma distribuição. Não faz sentido calcular $-\log p(S_t)$ de um ponto. Eles são avaliados por acertos por concurso ($k_t = |\hat S_t \cap S_t|$), média e máximo de acertos em um portfólio de $K$ jogos, e contagem de quadras/quinas/senas comparada com a distribuição nula simulada.

---

## 12. Modelos M1–M5

### 12.1. M0 — Uniforme

$$
p_{\text{M0}}(S) = \frac{1}{\binom{60}{6}}.
$$

É o piso absoluto. Todo modelo deve ser comparado a M0, e nenhum resultado é interpretável sem essa referência.

### 12.2. M1 — Plackett–Luce com marginalização sobre permutações (modelo primário proposto)

Pesos $w_i > 0$, $W = \sum_i w_i$. Para uma sequência ordenada de escolhas sem reposição $\pi = (\pi_1,\dots,\pi_6)$:

$$
P_{\text{ord}}(\pi) = \prod_{j=1}^{6} \frac{w_{\pi_j}}{W - \sum_{l<j} w_{\pi_l}}.
$$

Como a ordem real do sorteio não é observada, a probabilidade do conjunto $S$ é a soma sobre as $6! = 720$ permutações:

$$
P_{\text{M1}}(S) = \sum_{\pi \in \operatorname{Perm}(S)} \prod_{j=1}^{6} \frac{w_{\pi_j}}{W - \sum_{l<j} w_{\pi_l}}.
$$

**Normalização.** Cada sequência ordenada tem probabilidade $P_{\text{ord}}(\pi)$, e o conjunto de todas as sequências ordenadas de 6 dezenas é uma partição de $\Omega$ segundo a ordem. Logo, $\sum_{S \in \Omega} P_{\text{M1}}(S) = 1$.

**Forma computacional:**

$$
P_{\text{M1}}(S) = \left(\prod_{i \in S} w_i\right) \cdot \sum_{\pi \in \operatorname{Perm}(S)} \prod_{j=1}^{6} \frac{1}{W - \sum_{l<j} w_{\pi_l}}.
$$

**Identificabilidade.** $w$ e $c\,w$ produzem a mesma distribuição. Fixa-se $\sum_i \log w_i = 0$ (média geométrica 1). A estimação é feita em $\theta_i = \log w_i$ com a restrição $\sum_i \theta_i = 0$.

**Caso de controle.** Se todos $w_i = 1$, então $P_{\text{M1}}(S) = 1/\binom{60}{6} = P_{\text{M0}}(S)$.

**Custo.** $720 \times 6$ termos por concurso. Viável em TF/NumPy para 3.074 concursos.

**Modo.** Não existe DP simples para o modo de M1; usar beam search rotulado como heurístico.

### 12.3. M2 — Fisher (produto condicional, sensibilidade)

Pesos $w_i > 0$. Para $S = \{d_1,\dots,d_6\}$:

$$
P_{\text{M2}}(S) = \frac{\prod_{i \in S} w_i}{e_6(w)}, \qquad e_6(w) = \sum_{|T|=6} \prod_{i \in T} w_i.
$$

**Normalização.** Por definição de $e_6$, $\sum_{S \in \Omega} \prod_{i \in S} w_i = e_6(w)$. Logo, $\sum_S P_{\text{M2}}(S) = 1$.

**Cálculo de $e_6$.** DP em $O(Nk)$:

```
e[0] = 1
e[j] = 0 para j = 1..k
para i = 1..N:
    para j = min(i, k) até 1:
        e[j] += w[i] * e[j-1]
```

Ao final, $e_k(w) = e[k]$.

**Modo.** Como $e_6$ é constante, o modo de M2 é simplesmente o conjunto das 6 dezenas de maior peso.

### 12.4. M3 — Produto posicional normalizado (order statistics)

Distribuições $P_j$ sobre a $j$-ésima menor dezena, com suporte $\operatorname{supp}_j = \{j,\dots,60-6+j\}$ e $\sum_{c \in \operatorname{supp}_j} P_j(c) = 1$. Para $S = \{d_1<\dots<d_6\}$:

$$
q(S) = \prod_{j=1}^{6} P_j(d_j),
\qquad
Z = \sum_{d_1 < \dots < d_6} \prod_{j=1}^{6} P_j(d_j),
\qquad
P_{\text{M3}}(S) = \frac{q(S)}{Z}.
$$

**Cálculo de $Z$.** DP em $O(6 \cdot 60)$.

**Amostragem exata.** A amostragem sequencial truncada do ADR-001 **não** é exata para M3. Para amostrar exatamente de M3, usa-se amostragem backward pelo DP (§9.3 da math_spec.md).

**Relação com a LSTM de 6 cabeças.** A arquitetura de 6 softmax independentes, condicionadas apenas no histórico e não no prefixo de dezenas, é exatamente M3 quando normalizada por $Z$. Sem $Z$, é uma função de escore, não uma probabilidade sobre $\Omega$.

### 12.5. M4 — Plackett–Luce neural sequencial (ablação)

Uma rede (LSTM/GRU) condicionada no histórico e no prefixo de dezenas já escolhidas produz, a cada passo $j$, pesos $w_{j,i}$ para as dezenas restantes. A verossimilhança de uma sequência é:

$$
P_{\text{ord}}(\pi \mid \text{histórico}) = \prod_{j=1}^{6} \frac{w_{j,\pi_j}}{\sum_{i \notin \{\pi_1,\dots,\pi_{j-1}\}} w_{j,i}}.
$$

A probabilidade do conjunto é a soma sobre as 720 permutações:

$$
P_{\text{M4}}(S \mid \text{histórico}) = \sum_{\pi \in \operatorname{Perm}(S)} P_{\text{ord}}(\pi \mid \text{histórico}).
$$

**Custo.** $720 \times 6$ avaliações de rede por concurso, o que é caro.

### 12.6. M5 — LSTM de 6 cabeças como M3 / pseudo-likelihood

A arquitetura de referência da v1 produz seis distribuições $P_{t,j}$ por concurso $t$, condicionadas no histórico. Ela é uma instância de M3 com $P_j = P_{t,j}$, desde que normalizada por $Z_t$. Duas formas de treino:

- **M5-CCE (pseudo-likelihood):** minimizar $\sum_j -\log P_{t,j}(d_{t,j})$. Isso estima as marginais posicionais, não a likelihood do conjunto. A comparação justa é contra a pseudo-likelihood de B2a (soma de CCEs posicionais), não contra a NLL de M1.
- **M5-SET (likelihood de conjunto):** minimizar $-\log P_{\text{M3}}(S_t) = -\sum_j \log P_{t,j}(d_{t,j}) + \log Z_t$. Isso é comparável a M1/B2b por NLL de conjunto.

A escolha entre M5-CCE e M5-SET é registrada no ADR-003. A recomendação é M5-SET para a comparação principal e M5-CCE como ablação de custo.

### 12.7. Quadro-resumo dos modelos

| Modelo | Parâmetros | Usa histórico? | Likelihood de conjunto | Normalização | Papel |
|--------|-----------|---------------|------------------------|--------------|-------|
| M0 | 0 | não | sim | trivial | piso |
| M1 | 60 (59 gl) | não | sim (720 permutações) | automática | **primário** |
| M2 | 60 (59 gl) | não | sim ($e_6$) | automática | sensibilidade |
| M3 | $6 \times 55$ (ou rede) | opcional | sim ($Z$ DP) | $Z$ | ablação |
| M4 | rede | sim | sim (720 permutações) | automática | ablação neural |
| M5 | rede | sim | sim (se M5-SET) ou pseudo | $Z$ | ablação DL |

---

## 13. Estratégias de Treinamento

### 13.1. M1/M2 — máxima verossimilhança de conjunto

- Otimizador: Adam, `learning_rate = 1e-2`, full-batch ou mini-batch 64.
- `max_epochs = 100`.
- Early stopping: paciência 10 sobre o log-score de validação (sem peso).
- Restauração dos melhores pesos.
- $\lambda \in \{0; 10^{-4}; 10^{-3}\}$, escolhido na validação **interna ao fold** (nunca com dados posteriores ao alvo).
- 3 reinicializações para confirmar convergência; relatar a dispersão.

### 13.2. M3/M4/M5 — treino neural

- `max_epochs = 100`.
- Early stopping: paciência 10 sobre a perda de validação **sem peso**.
- `ReduceLROnPlateau`: fator 0,5; paciência 5; `min_lr = 1e-6`.
- Otimizador Adam, `learning_rate = 1e-3`.
- Lote 64.
- Validação: último 10% cronológico dos concursos regulares do treino; nunca misturar treino/validação/teste.
- A validação é sempre **interna ao fold**: nenhum hiperparâmetro é escolhido com dados posteriores ao alvo. No Teste A, a validação é o intervalo 2471–2772; em cada fold do Teste B, é o último 10% dos concursos regulares anteriores à Virada.
- 10 seeds para M3/M4/M5; média ± desvio-padrão.
- Arquitetura de referência da v1 (LSTM 128 → Dropout 0,2 → LSTM 64 → concat → Dense 128 → 6 × Dense 60 softmax) é o ponto de partida; alterações são ablações registradas.

### 13.3. Duas perdas neurais e a regra de comparação

| Perda | Fórmula | O que estima | Comparável com |
|-------|---------|--------------|----------------|
| CCE posicional (pseudo-likelihood) | $\sum_j -\log P_{t,j}(d_{t,j})$ | marginais posicionais | B2a-CCE, M5-CCE |
| Log-score de conjunto | $-\sum_j \log P_{t,j}(d_{t,j}) + \log Z_t$ | likelihood de conjunto M3 | M1, M2, B2a, B2b |

**Regra:** nunca comparar uma NLL de conjunto com uma CCE posicional no mesmo teste. Se o modelo for treinado com CCE, ele é avaliado com CCE e rotulado como pseudo-likelihood; se for treinado com log-score de conjunto, é avaliado com log-score de conjunto.

### 13.4. Peso amostral $\alpha$

O peso da amostra $t$ é:

$$
\omega_t =
\begin{cases}
1, & \text{se o alvo é Virada},\\
\alpha, & \text{caso contrário},
\end{cases}
\qquad 0 < \alpha \le 1.
$$

A perda do lote é $\sum_t \omega_t L_t / \sum_t \omega_t$. O padrão é **$\alpha = 1.0$** (sem diferenciação). A grade $\alpha \in \{0.01; 0.05; 0.10; 0.25\}$ é ablação, não configuração primária.

O tamanho efetivo de amostra (Kish) é:

$$
N_{\text{eff}} = \frac{(\sum_t \omega_t)^2}{\sum_t \omega_t^2}.
$$

Com 3.057 concursos regulares e 17 Viradas:

| $\alpha$ | $N_{\text{eff}}$ aprox. |
|---:|---:|
| 0,01 | 131 |
| 0,05 | 1.171 |
| 0,25 | 2.934 |
| 1,00 | 3.074 |

$N_{\text{eff}}$ deve ser reportado em toda execução com $\alpha \neq 1$.

---

## 14. Protocolo de Validação Temporal

### 14.1. Walk-forward por edição da Mega da Virada

Para cada edição $Y$ (2015–2025):

1. Treina-se com todos os concursos anteriores à Virada $Y$.
2. Valida-se com o último 10% dos concursos regulares do treino.
3. Avalia-se na Virada $Y$ (teste).

A validação nunca contém o alvo nem concursos posteriores a ele.

### 14.2. Conjuntos de avaliação pré-registrados

**Teste A (primário, $n = 300$):** Os 300 concursos regulares mais recentes do snapshot, com `is_virada = 0`:

- Início: concurso 2773 (2024-09-12).
- Fim: concurso 3074 (2026-10-24).
- Excluídos do conjunto: concursos 2810 e 2955 (Viradas).
- Treino do Teste A: concursos 1–2772.
- Validação para seleção de hiperparâmetros: 300 concursos regulares 2471–2772.
- Ajuste de referência (após seleção): concursos 1–2772.

**Teste B (secundário, $n = 11$):** As Viradas de 2015 a 2025, cada uma avaliada no fold walk-forward correspondente (treino estritamente anterior ao concurso da Virada). Viradas de 2009–2014 entram apenas como treino.

Teste A e Teste B **não são independentes** para as Viradas de 2024 e 2025. O Teste A é a análise confirmatória; o Teste B é secundário e descritivo, com poder muito menor.

---

## 15. Métricas e Critérios Estatísticos

### 15.1. Métrica primária: log-score de conjunto

$$
\text{LS} = \frac{1}{n}\sum_{t=1}^{n} \log p(S_t),
\qquad
\text{NLL} = -\text{LS}.
$$

A comparação é pareada:

$$
\Delta_t = \log p_{\text{modelo}}(S_t) - \log p_{\text{baseline}}(S_t).
$$

$\Delta_t > 0$ favorece o modelo. O efeito é reportado em nats por concurso e como razão de verossimilhança geométrica $\exp(\overline{\Delta})$.

### 15.2. Métricas secundárias

- Acertos $k_t = |\hat S_t \cap S_t|$; média e distribuição.
- $A_{\text{bar}}$ e $A_{\text{max}}$ do portfólio de $K$ jogos.
- Contagem de quadras/quinas/senas.
- Brier marginal e calibração (exploratórios).
- PL-CCE para M5-CCE vs B2a-CCE (exploratório).

### 15.3. Proibições

- Acurácia, precisão, revocação, F1, especificidade não são evidência principal.
- NLL de B1/B3 não entra no teste.
- NLL de M5-CCE não é comparada com NLL de conjunto.
- Nenhum resultado de melhor configuração sem Holm.
- Nenhuma métrica por dezena sem correção de multiplicidade.

---

## 16. Testes de Sanidade

Executar antes de qualquer conclusão:

1. **Rótulos permutados:** embaralhar os conjuntos-alvo entre concursos; M1/M3 não podem vencer B0 (nem B2b/B2a) no Teste A.
2. **Entradas aleatórias:** substituir as janelas por multi-hot aleatórios; M3/M5 não podem vencer os baselines.
3. **Sentinela de vazamento:** inserir um concurso futuro fictício; o splitter deve rejeitá-lo.
4. **Paridade de $K$:** todo portfólio comparado tem o mesmo $K$ e o mesmo filtro.
5. **Normalização:** todo modelo probabilístico soma 1 em universo reduzido.
6. **Reprodutibilidade:** duas execuções com a mesma seed produzem o mesmo resultado dentro da tolerância documentada.

Se qualquer sanidade falhar, o veredito é **erro/vazamento**, não evidência.

---

## 17. Seleção de Modelos

Toda seleção é **interna ao fold** e usa apenas dados anteriores ao alvo. Nenhum hiperparâmetro é escolhido com o conjunto de teste do Teste A (2773–3074) nem com as Viradas do Teste B.

- **Teste A:** validação = 300 concursos regulares 2471–2772. Ajuste de referência após a seleção = 1–2772.
- **Cada fold do Teste B:** validação = último 10% dos concursos **regulares** do treino do fold (mínimo de 100 concursos). Ajuste final = todos os concursos anteriores à Virada.
- **M1/M2:** $\lambda \in \{0; 10^{-4}; 10^{-3}\}$, escolhido por log-score de validação.
- **M3/M5:** $W \in \{10,20,40\}$, escolhido por log-score de validação com 3 seeds; a configuração escolhida é reajustada com 10 seeds.
- **Desempate:** em caso de empate numérico, prefira o modelo mais simples — menor $W$ e maior $\lambda$.
- $\alpha = 1.0$ é a configuração primária. $\alpha < 1$ é ablação exploratória, restrita a M3/M5.
- $\kappa = 1.0$ é a suavização primária de B2a; $\{0.1; 0.5; 2.0\}$ é sensibilidade.
- A grade de $\alpha$, a grade de $\kappa$ e a grade de $W$ não usada na seleção são exploratórias.

---

## 18. Treinamento Final

O modelo primário é treinado com todos os concursos até o último disponível antes do sorteio prospectivo (2026). Gerar $K = 10$ jogos com a configuração pré-registrada, seed fixa e filtro de diversidade. Gravar `artifacts/generations/<timestamp>_<hash>.json` com modelo, dataset, config, commit, seed, $T$, $K$, diversidade e jogos. Após o sorteio, avaliar descritivamente; $n = 1$, sem teste de significância. Nenhum ajuste posterior ao sorteio.

---

## 19. Inferência

A inferência utiliza artefatos treinados salvos em disco (formato `.keras`). O modelo é carregado e aplicado sobre a janela histórica mais recente, sem exigir retreinamento da rede a cada execução. A janela de inferência é construída a partir dos últimos $W$ concursos válidos, com `alvo_virada = 1`.

---

## 20. Geração de Jogos

### 20.1. Requisitos dos jogos

Cada jogo deverá:
- Ter exatamente seis dezenas.
- Utilizar dezenas entre 1 e 60.
- Não conter repetições.
- Ser armazenado em formato consistente.
- Poder ser validado automaticamente.

### 20.2. Estratégias de decodificação e amostragem

#### Estratégia 1: Decodificação de Máxima Verossimilhança (Top-1 / Best Sequence)

Utiliza programação dinâmica em $O(6 \times 60)$ para encontrar a combinação única que maximiza a soma das log-probabilidades sob a restrição estrita de ordenação:

$$
\max_{1 \le d_1 < d_2 < \dots < d_6 \le 60} \sum_{i=1}^6 \log P_i(d_i).
$$

É expressamente proibido utilizar o argmax isolado de cada cabeça, pois isso gera repetições e violações de ordem.

#### Estratégia 2: Amostragem ponderada sequencial para múltiplos jogos

Para gerar $K$ jogos distintos, adota-se amostragem autorregressiva/condicionada na ordem das cabeças:

1. Para a primeira cabeça ($i=1$), a distribuição sobre $d_1$ é truncada no intervalo $[1, 55]$ e normalizada:

$$
P(d_1 = c) \propto P_{1, c}^{1/T}, \quad c \in [1, 55].
$$

2. Para cada posição subsequente $i \in \{2, \dots, 6\}$, dada a dezena $d_{i-1}$ sorteada na posição anterior, trunca-se a distribuição da cabeça $i$ estritamente para o intervalo $[d_{i-1} + 1, 60 - (6 - i)]$:

$$
P(d_i = c \mid d_{i-1}) \propto P_{i, c}^{1/T}, \quad c \in [d_{i-1} + 1, 60 - (6 - i)],
$$

onde $T > 0$ é a temperatura de amostragem configurável ($T = 1.0$ por padrão; $T < 1.0$ concentra nas maiores probabilidades; $T > 1.0$ aumenta a exploração).

Por construção, toda combinação gerada satisfaz $1 \le d_1 < d_2 < \dots < d_6 \le 60$ e não possui dezenas repetidas.

### 20.3. Mecanismo de diversidade controlada

Um jogo candidato $J_{\text{candidato}}$ só é aceito se:

$$
|J_{\text{candidato}} \cap J_m| \le \text{max\_shared\_numbers}, \quad \forall m < k.
$$

O parâmetro `max_shared_numbers` $\in \{2, 3, 4, 5\}$ (padrão: 4) define o teto de dezenas coincidentes entre qualquer par de jogos da mesma geração.

### 20.4. Manifesto de geração

Cada geração grava `artifacts/generations/<timestamp>_<hash>.json` com: modelo (hash), dataset (hash), config (hash), commit, seed, $T$, modelo probabilístico, $K$, diversidade, tentativas, jogos e flag de saturação. Nada é sobrescrito.

---

## 21. Amostragem e Decodificação

### 21.1. M0, M1 e B2b

- **M0:** amostrar 6 dezenas uniformes sem reposição e ordenar.
- **M1/B2b:** amostrar uma sequência PL sucessiva (peso $w_i$ entre as restantes), depois ordenar. Isso é exato para a distribuição de conjuntos de M1, porque cada sequência ordenada tem a probabilidade $P_{\text{ord}}$ e a ordenação é uma função determinística.
- **Temperatura:** $w_i^{(1/T)}$, renormalizados. $T=1$ na avaliação científica.

### 21.2. M2 (Fisher)

Amostragem recursiva exata. Para um conjunto de tamanho $k$ restante e pesos $w$:

1. Calcular $e_{k-1}(w_{-i})$ para cada $i$ por DP.
2. Amostrar o primeiro elemento $i$ com probabilidade:

$$
\pi_i = \frac{w_i\, e_{k-1}(w_{-i})}{k\, e_k(w)}.
$$

3. Remover $i$, decrementar $k$ e repetir.

Esse esquema produz cada conjunto $S$ com probabilidade $\prod_{i\in S} w_i / e_k(w)$.

### 21.3. M3/M5 (produto posicional)

Amostragem backward exata pelo DP:

1. Calcular $F_1(c) = P_1(c)$ e $F_j(c) = P_j(c)\sum_{c'<c} F_{j-1}(c')$ para $j=2,\dots,6$.
2. $Z = \sum_c F_6(c)$.
3. Amostrar $d_6$ com probabilidade $F_6(c)/Z$.
4. Para $j=5,\dots,1$, amostrar $d_j$ com probabilidade proporcional a $F_j(c)$ para $c < d_{j+1}$.

Isso é exato para $P_{\text{M3}}(S) = \prod_j P_j(d_j)/Z$. A amostragem sequencial truncada do ADR-001 **não** é exata e não deve ser usada como se fosse.

### 21.4. M1 e M4 — modo e decodificação pontual

- **M2:** modo exato = 6 maiores pesos.
- **M3/M5:** modo exato = DP atual.
- **M1:** o modo do conjunto não é, em geral, o modo da sequência. Não há DP simples. Usar beam search sobre sequências ordenadas, coletar os melhores conjuntos candidatos e avaliar $P_{\text{M1}}(S)$ exatamente para cada candidato. Rotular como **heurístico** e reportar o limitante superior $6! \cdot \max_\pi P_{\text{ord}}(\pi)$.
- **M4:** beam search análogo, condicionado no prefixo.

Nenhum ponto preditivo é usado como evidência de sinal; ele é uma saída operacional.

---

## 22. Diversidade de Portfólio

O filtro `max_shared_numbers` é aplicado após a amostragem:

$$
|J_{\text{cand}} \cap J_m| \le \text{max\_shared\_numbers}, \qquad \forall m < k.
$$

Ele **altera a distribuição** do portfólio. Para avaliação científica:

- desativar o filtro; ou
- simular a distribuição nula com o mesmo filtro e o mesmo $K$.

Nunca comparar um portfólio filtrado com um baseline não filtrado. Se o filtro saturar, o comportamento deve ser explícito: retornar parcial com `saturated = True` (não lançar exceção).

Para avaliação científica de log-score, o filtro é desativado (a amostra é i.i.d.).

---

## 23. Manifestos e Rastreabilidade

### 23.1. Manifesto de dados (RT-004)

Cada execução grava um manifesto JSON com:

- hash SHA-256 do raw e do processado;
- hash do config YAML;
- hash do modelo salvo;
- commit Git (ou `dirty` + diff hash);
- versões de Python, NumPy, Pandas, SciPy, TensorFlow/Keras;
- seeds;
- data/hora UTC;
- `virada_ano` e folds usados.

### 23.2. Manifesto de geração

Cada geração grava `artifacts/generations/<timestamp>_<hash>.json` com:

- modelo (hash);
- dataset (hash);
- config (hash);
- commit;
- seed;
- $T$;
- modelo probabilístico;
- $K$;
- diversidade;
- tentativas;
- jogos;
- flag de saturação.

---

## 24. Arquitetura do Software

### 24.1. Estrutura de pacotes (`src/lstm_virada/`)

```text
src/lstm_virada/
├── __init__.py
├── baselines.py          # B0, B1, B2a, B2b, B3
├── data.py               # Ingestão, validação RT-003, virada_ano, manifesto
├── decode.py             # Decodificação ótima por DP, exceções em vez de assert
├── encoding.py           # Multi-hot 61 features, janelas W, targets_to_draw
├── generation.py         # Geração de jogos com filtros de diversidade e manifesto
├── metrics.py            # Baselines B0–B3, métricas M1–M10
├── splits.py             # TemporalSplitter, WalkForwardSplitter, InferenceWindowExtractor
├── stats.py              # Testes de hipótese, bootstrap BCa, veredito
├── train_walkforward.py  # Orquestração do treino walk-forward
└── models/               # Modelos probabilísticos
    ├── __init__.py
    ├── plackett_luce.py    # M1 — Plackett–Luce marginalizado
    ├── fisher.py           # M2 — Fisher (produto condicional)
    ├── positional.py       # M3 — Produto posicional normalizado
    ├── seq_pl.py           # M4 — PL Neural Sequencial
    └── lstm_seis_cabecas.py # M5 — LSTM de 6 cabeças (M5-SET e M5-CCE)
```

`train_walkforward.py` está na raiz do `src/` e orquestra o treino walk-forward.

### 24.2. Convenções de código

- **RC-001**: Python 3.11, TensorFlow 2.x / Keras.
- **RC-002**: Tipagem estática (`mypy`), linter e formatação (`ruff`), testes (`pytest`) com cobertura mínima de 85% em dados e métricas.
- **RC-003**: Configuração em arquivos YAML versionados (sem constantes mágicas no código).
- **RC-004**: Funções puras e isoladas para codificação, decodificação, amostragem e validação de jogos. Operações com I/O (disco/rede) são estritamente isoladas.
- **RC-005**: Apresentação dos jogos e relatórios no terminal e em tabelas do Jupyter Notebook.
- **RC-006**: Logs, mensagens e docstrings em português; nomes de identificadores em inglês.

---

## 25. Estrutura do Repositório

```
Mega da virada LSTM/
├── spec/                                       # Especificação única e ADRs
│   ├── PROJECT_SPEC.md                         # Especificação única (este documento)
│   ├── math_spec.md                            # Histórico (v0.3-draft)
│   ├── pre_registration.md                     # Histórico (pré-registro)
│   ├── spec_lstm_mega_virada_v0.2.md          # Histórico (v0.2)
│   ├── spec_lstm_mega_virada_v1.md            # Histórico (v0.1)
│   └── adr/                                    # Decisões arquiteturais
│       ├── ADR-001-geracao-de-jogos-e-diversidade.md
│       ├── ADR-002-modelo-probabilistico-primario.md
│       └── ADR-003-likelihood-e-comparabilidade.md
├── scripts/                                    # Scripts de execução
│   ├── activar_treino.ps1                     # Ativa .venv-treino (TensorFlow, Windows)
│   ├── activar_treino_wsl.sh                  # Define LD_LIBRARY_PATH para CUDA (WSL2)
│   ├── check_gpu.py                            # Verificação de GPU (Windows/DirectML)
│   ├── check_gpu_wsl.py                        # Verificação de GPU (WSL/TensorFlow)
│   ├── run_treino.py                           # Ponto de entrada do treino
│   ├── run_generation.py                       # Ponto de entrada da geração de jogos
│   └── configurar_venv_wsl_gpu_native.sh       # Configuração do venv WSL (referência)
├── configs/                                    # Arquivos YAML versionados
│   ├── README.md
│   └── experiments/                            # Configurações de treinos e ablações
├── data/
│   ├── raw/                                    # Dados brutos imutáveis
│   │   ├── megasena-download-resultados.csv
│   │   └── mega_da_virada.csv
│   ├── processed/                              # Datasets unificados com hash SHA-256
│   │   └── dataset_megasena_virada_processed.csv
│   └── manifests/                              # Manifestos com hashes SHA-256
│       └── manifest.json
├── notebooks/                                  # Ambiente interativo Jupyter
│   ├── 00_format_megasena_geral.ipynb
│   ├── 01_format_virada.ipynb
│   ├── 02_inspecao_e_eda.ipynb
│   └── 03_experimentos.ipynb
├── src/
│   └── lstm_virada/                            # Pacote de lógica de produção pura
│       ├── __init__.py
│       ├── baselines.py
│       ├── data.py
│       ├── decode.py
│       ├── encoding.py
│       ├── generation.py
│       ├── metrics.py
│       ├── splits.py
│       ├── stats.py
│       ├── train_walkforward.py
│       └── models/
│           ├── __init__.py
│           ├── plackett_luce.py
│           ├── fisher.py
│           ├── positional.py
│           ├── seq_pl.py
│           └── lstm_seis_cabecas.py
├── tests/                                      # Bateria de testes unitários com rastreabilidade
│   ├── conftest.py
│   ├── data/test_data.py
│   ├── encoding/test_encoding.py
│   ├── splits/test_splits.py
│   ├── baselines/test_baselines.py
│   ├── metrics/test_metrics.py
│   ├── stats/test_stats.py
│   ├── models/test_plackett_luce.py
│   ├── models/test_fisher.py
│   ├── models/test_positional.py
│   ├── models/test_seq_pl.py
│   ├── models/test_lstm_seis_cabecas.py
│   └── train_walkforward/test_walkforward.py
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-base.txt
└── requirements-gpu.txt
```

### 24.3. Diretórios ignorados pelo Git

- `.venv/`, `.venv-treino/`, `.venv-treino-wsl/`, `python311/` — ambientes virtuais não versionados.
- `artifacts/` — artefatos gerados pelo pipeline (modelos, gerações).
- `data/processed/` e `data/manifests/` — dados processados e manifestos (gerados).
- `__pycache__/`, `*.py[cod]`, `*.egg-info/` — cache Python.
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` — cache de ferramentas.

---

## 26. Ambientes e Dependências

### 26.1. Ambiente base (Windows)

Para desenvolvimento e testes que **não** dependem do TensorFlow:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 26.2. Ambiente de treino (Windows, CPU/DirectML)

Para treinar modelos com TensorFlow:

```powershell
.\scripts\activar_treino.ps1
python scripts/check_gpu.py
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10
deactivate
```

### 26.3. Ambiente de treino (WSL2 Ubuntu 22.04, GPU/CUDA)

O WSL2 permite aceleração por GPU NVIDIA via CUDA:

```bash
source /home/<usuario>/.venv-treino-wsl/bin/activate
source scripts/activar_treino_wsl.sh   # define LD_LIBRARY_PATH para CUDA
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python scripts/run_treino.py --modelo M5-SET --epocas 100 --seeds 10
deactivate
```

O script `scripts/configurar_venv_wsl_gpu_native.sh` documenta como criar este venv do zero (instala TensorFlow + bibliotecas CUDA via pip, sem depender de `apt`).

---

## 27. Comandos de Execução

### 27.1. Testes (não-TensorFlow)

```powershell
pytest tests\data\ tests\encoding\ tests\splits\ tests\baselines\ tests\metrics\ tests\stats\
```

### 27.2. Treino (Windows)

```powershell
.\scripts\activar_treino.ps1
python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 42 43 44 --window-size 20
deactivate
```

### 27.3. Treino (WSL/GPU)

```bash
source /home/<usuario>/.venv-treino-wsl/bin/activate
source scripts/activar_treino_wsl.sh
python scripts/run_treino.py --modelo M5-SET --epocas 100 --seeds 10 --window-size 20
deactivate
```

### 27.4. Geração de jogos

```bash
source /home/<usuario>/.venv-treino-wsl/bin/activate
source scripts/activar_treino_wsl.sh
python scripts/run_generation.py --modelo M1 --K 10 --seed 42 --diversidade 4
deactivate
```

---

## 28. Testes de Software

| Grupo | Requisitos cobertos | Status |
|-------|---------------------|--------|
| `tests/data/` | RT-001 a RT-004, P0-01 a P0-04 | ✅ 18 testes |
| `tests/encoding/` | RT-005, RT-006, P0-06, P0-10 | ✅ 25 testes |
| `tests/splits/` | P0-05 | ✅ 15 testes |
| `tests/baselines/` | P0-07 | ✅ 16 testes |
| `tests/metrics/` | P0-08 | ✅ 23 testes |
| `tests/stats/` | P0-09 | ✅ 16 testes |
| `tests/models/` | P1-01 a P1-05 | ✅ 44 testes |
| `tests/train_walkforward/` | P1-06 | ✅ 9 testes |
| `tests/integration/` | E2E | ✅ 2 testes |

**Total de testes não-TF aprovados: 111.**

---

## 29. Estado de Implementação

| Fase | Item | Escopo | Status |
|------|------|--------|--------|
| Fase 0 | `spec/math_spec.md` | Modelos M0–M5, likelihoods, normalizações, baselines, avaliação | ✅ Concluída (Gate 0 aprovado) |
| Fase 0 | `spec/adr/ADR-002` | Modelo primário M1, M2, M3/M4/M5, LSTM como ablação | ✅ Concluída |
| Fase 0 | `spec/adr/ADR-003` | Likelihood, normalização, comparabilidade | ✅ Concluída |
| Fase 0 | `spec/pre_registration.md` | Hipóteses H1–H7, splits, métricas, testes, veredito | ✅ Concluída |
| Fase 1 | P0-01 a P0-04 | Proveniência, `virada_ano`, manifesto, testes invariantes | ✅ Concluída (Gate 1 aprovado) |
| Fase 2 | P0-05 a P0-10 | Splitter, baselines, métricas, estatística, contratos | ✅ Concluída (Gate 2 aprovado) |
| Fase 3 | P1-01 | M1 Plackett–Luce | ✅ Concluído (9 testes) |
| Fase 3 | P1-02 | M2 Fisher | ✅ Concluído (10 testes) |
| Fase 3 | P1-03 | M3 Produto Posicional | ✅ Concluído (11 testes) |
| Fase 3 | P1-04 | M4 PL Neural Sequencial | ✅ Concluído (5 testes) |
| Fase 3 | P1-05 | M5 LSTM 6 cabeças (M5-SET e M5-CCE) | ✅ Concluído (7 testes) |
| Fase 3 | P1-06 | Treino walk-forward | ✅ Concluído (9 testes) |
| Fase 4 | P4-1 a P4-4 | Inferência e geração com artefatos treinados | ✅ Concluído (`generation.py`) |

### Pendentes (Fases P3–P5)

- Walk-forward completo (M1 vs B0 para H1, M1 vs B2b para S2, M3 vs B2a para S1).
- Testes de sanidade (rótulos permutados, entradas aleatórias, sentinela).
- Correção de Holm às hipóteses secundárias.
- Veredito padronizado.
- Seleção de modelo.
- Treino com todo histórico.
- Geração prospectiva 2026.

---

## 30. Limitações Científicas

1. **Independência estocástica:** Sorteios da Mega-Sena são projetados e auditados para serem processos i.i.d. (independentes e identicamente distribuídos).
2. **Equiprobabilidade teórica:** Em um globo justo com 60 bolas, todas as $\binom{60}{6} = 50\,063\,860$ combinações possíveis possuem exatamente a mesma probabilidade teórica de ocorrência ($1 / 50\,063\,860 \approx 2 \times 10^{-8}$).
3. **Ausência de garantia de ganho:** O treinamento de modelos neurais sobre séries temporais de loterias visa capturar eventuais desvios empíricos ou artefatos mecânicos sutis de forma experimental, mas **não confere aumento comprovado de probabilidade nem garantia de ganho financeiro**.
4. **Alerta de aposta:** A geração de jogos destina-se a fins estritamente educacionais, técnicos e de pesquisa em Machine Learning.

---

## 31. Riscos e Decisões Pendentes

| ID | Decisão | Recomendação | Onde registrar |
|----|---------|--------------|----------------|
| D-01 | M1 ou M2 como primário | **M1** (interpretação física de urna sequencial); M2 como sensibilidade | `ADR-002` |
| D-02 | M3 ou M4 como ablação neural | **M3** primeiro (custo baixo); M4 só se M3 mostrar algo | `ADR-002` |
| D-03 | M5-CCE ou M5-SET | **M5-SET** para a comparação principal; M5-CCE como ablação de custo | `ADR-003` |
| D-04 | Saturação do filtro de diversidade | Retornar parcial com flag `saturated=True` (não lançar exceção) | `ADR-001` (emenda) |
| D-05 | Recuperação do raw original | Tentar a fonte oficial; se falhar, documentar a perda e marcar o dataset como "proveniência parcial" | Fase 1 |

---

## 33. Pré-Registro Científico Original

O pré-registro estatístico está preservado integralmente no arquivo `spec/pre_registration.md` e foi incorporado a este documento. Resumo:

- **Hipótese primária (H1):** No Teste A ($n = 300$ concursos regulares), o log-score de conjunto de M1 é maior que o de **B0 (uniforme)**. Teste: Wilcoxon signed-rank unilateral. Nível: $\alpha = 0,05$. Efeito mínimo de relevância prática: $\overline{\Delta} \ge 0,05$ nats/concurso.
- **Hipóteses secundárias (S1–S5):** submetidas à correção de Holm.
- **Critérios de veredito:** Evidência / Inconclusivo / Sem evidência / Erro-vazamento.
- **Prospectivo 2026:** treinar o modelo primário com todos os concursos até o último disponível antes do sorteio; gerar $K = 10$ jogos com configuração pré-registrada, seed fixa e filtro de diversidade; avaliar descritivamente após o sorteio ($n = 1$, sem teste de significância).

---

## 34. Emendas e Desvios Metodológicos

Nenhuma emenda foi registrada até o momento. Todas as decisões metodológicas estão refletidas nos documentos originais (`math_spec.md`, `pre_registration.md`, `ADR-001`, `ADR-002`, `ADR-003`) e foram incorporadas a este documento único.

---

## 35. Histórico de Decisões Arquiteturais

### ADR-001 — Geração de jogos e diversidade

- Decisão 1: Desacoplamento entre treinamento e inferência (módulo `generation.py`).
- Decisão 2: Estratégia de amostragem sequencial condicionada (weighted sampling).
- Decisão 3: Mecanismo de diversidade por limite de dezenas compartilhadas (`max_shared_numbers`).
- Decisão 4: Paridade amostral estrita na avaliação científica.

### ADR-002 — Modelo probabilístico primário

- M1 (Plackett–Luce marginalizado) é o modelo primário.
- M2 (Fisher) é o modelo de sensibilidade.
- M3, M4, M5 são ablações neurais.
- Nenhuma ablação neural é modelo principal.

### ADR-003 — Likelihood e comparabilidade

- Todo modelo probabilístico é normalizado sobre $\Omega$.
- A métrica primária é o log-score de conjunto.
- Modelos pontuais (B1, B3) não têm NLL.
- Pseudo-likelihood é uma família separada.
- B2 tem duas versões (B2a e B2b), ambas válidas.
- Suavização é obrigatória em B2a.
- A unidade de análise é o concurso.
- A comparação é sempre contra o mesmo $K$ e o mesmo filtro.
- M5 tem duas variantes, com comparações distintas.

---

## 36. Critérios de Conclusão

O projeto está concluído quando:

1. Todos os modelos pré-registrados foram executados.
2. As sanidades foram verificadas e aprovadas.
3. O veredito padronizado foi emitido (Evidência / Inconclusivo / Sem evidência / Erro-vazamento).
4. O prospectivo 2026 foi gerado e registrado.
5. Não há "parada por resultado promissor" nem seleção de seed.

A conclusão "sem evidência de sinal" é um resultado científico válido e esperado.

---

## Apêndice A — Derivações

### A.1 Normalização de M1

O conjunto de todas as sequências ordenadas de 6 dezenas distintas é $\Pi_6$. A construção PL define uma probabilidade sobre $\Pi_6$ que soma 1. Cada $S \in \Omega$ corresponde a exatamente $6!$ sequências. Logo, $\sum_S P_{\text{M1}}(S) = 1$.

### A.2 DP de $e_k$

A recorrência $e_j^{(i)} = e_j^{(i-1)} + w_i e_{j-1}^{(i-1)}$ calcula o polinômio simétrico elementar de grau $j$.

### A.3 DP de $Z$ em M3

Define $F_j(c) = \sum_{d_1 < \dots < d_{j-1} < c} \prod_{l=1}^{j} P_l(d_l)$. Então $F_1(c) = P_1(c)$ e $F_j(c) = P_j(c) \sum_{c' < c} F_{j-1}(c')$. Logo, $Z = \sum_c F_6(c)$.

### A.4 Amostragem backward de M3

A marginal de $d_6$ é $p(d_6 = c) = F_6(c)/Z$. Dado $d_{j+1}$, a condicional de $d_j$ é proporcional a $F_j(c)$ para $c < d_{j+1}$. Isso permite amostrar de trás para frente, exatamente.

### A.5 Amostragem de M2

A probabilidade de inclusão de $i$ é $P(i \in S) = \frac{w_i e_{k-1}(w_{-i})}{e_k(w)}$. Para amostrar sem reposição, escolhe-se o primeiro elemento com probabilidade $\pi_i = \frac{w_i e_{k-1}(w_{-i})}{k e_k(w)}$, remove-se $i$, decrementa-se $k$ e repete-se.

---

## Apêndice B — Vetores de teste (universo reduzido)

Para validar a implementação, use $N = 5$, $k = 2$, $w = (1,2,3,4,5)$, $W = 15$.

### B.1 M2 — Fisher

$e_2(w) = 85$. As 10 probabilidades são: {1,2}=2/85, {1,3}=3/85, {1,4}=4/85, {1,5}=5/85, {2,3}=6/85, {2,4}=8/85, {2,5}=10/85, {3,4}=12/85, {3,5}=15/85, {4,5}=20/85. Soma = 1. O modo é {4,5}.

### B.2 M1 — Plackett–Luce marginalizado

As 10 probabilidades de conjunto são: {1,2}=9/455, {1,3}=13/420, {1,4}=10/231, {1,5}=2/35, {2,3}=5/78, {2,4}=64/715, {2,5}=23/195, {3,4}=23/165, {3,5}=11/60, {4,5}=14/55. Soma = 1. O modo é {4,5}.

### B.3 Caso uniforme

Com $w = (1,1,1,1,1)$, $N=5$, $k=2$: $P_{\text{M0}} = P_{\text{M1}} = P_{\text{M2}} = 1/10$ para todo $S$.

### B.4 Teste de normalização

Para qualquer implementação, as somas de $P_{\text{M1}}$, $P_{\text{M2}}$ e $P_{\text{M3}}$ sobre todos os $\binom{5}{2}=10$ conjuntos devem ser 1 dentro de $10^{-10}$.

---

## Apêndice C — Glossário de decisões

| Termo | Significado neste documento |
|-------|----------------------------|
| **Conjunto** | 6 dezenas distintas, em ordem crescente (representação canônica) |
| **Sequência** | ordem de escolha das 6 dezenas; não observada no dataset |
| **Likelihood de conjunto** | $P(S)$, normalizada sobre $\Omega$ |
| **Pseudo-likelihood** | soma de CCEs posicionais; não é $P(S)$ |
| **Normalização** | garantir $\sum_S p(S) = 1$ |
| **Modo** | conjunto $S$ de maior $p(S)$ |
| **Walk-forward** | treinar apenas com concursos anteriores ao alvo |
| **Teste A** | 300 concursos regulares 2773–3074 (excluindo Viradas) |
| **Teste B** | 11 Viradas 2015–2025, uma por fold |

---

*Documento gerado a partir da consolidação de `spec/math_spec.md`, `spec/pre_registration.md`, `spec/spec_lstm_mega_virada_v1.md`, `spec/spec_lstm_mega_virada_v0.2.md` e `spec/adr/ADR-001/002/003.md`. Os documentos originais foram arquivados e não são mais específicos ativos deste projeto.*
