# Spec SDD: LSTM para a Mega da Virada

**Versão:** 0.1 · **Status:** Rascunho consolidado · **Data:** 04/11/2026  
**Stack:** Python 3.11 + TensorFlow/Keras · **Idioma:** pt-BR

> [!IMPORTANT]
> Fonte de verdade do projeto. Código, testes, configurações e prompts para agentes de IA devem derivar desta especificação. Se houver divergência entre a implementação e a spec, **a spec prevalece** até que seja alterada por revisão explícita e registrada em commit separado.

---

## 1. Contexto e objetivo

**Objetivo.** Treinar e avaliar um modelo LSTM que, dado o histórico de concursos, produz uma sequência de 6 dezenas (em ordem crescente, de 1 a 60) para a Mega da Virada. O treinamento utiliza todos os concursos da Mega-Sena regular, atribuindo-lhes peso menor que o atribuído às amostras cujo alvo é uma Mega da Virada.

**Natureza do projeto.** Sorteios são processos desenhados para serem independentes e uniformes. A hipótese de que um LSTM consiga extrair sinal desses dados é experimental. A especificação considera "sem evidência de sinal" um resultado válido e esperado. O sucesso do projeto é construir um pipeline correto, reproduzível e estatisticamente honesto — não acertar dezenas.

**Fora de escopo.** Estratégias de apostas, garantia de retorno, outras loterias, interface de usuário, serviço remoto, API pública e publicação ou implantação do modelo.

### 1.1. Escopo operacional da versão 1 (v1)

A v1 será executada localmente, na máquina do desenvolvedor. O modelo será treinado localmente e utilizado somente para validação técnica do pipeline, da implementação e das métricas experimentais. Não haverá treinamento em nuvem, disponibilização como serviço, API, interface gráfica, aplicação web ou publicação do modelo.

Ao término da execução, os resultados e o relatório da seção 10 serão devolvidos diretamente no próprio terminal, em tabelas de texto. A execução deverá apresentar as métricas, comparações com os baselines, verificações estatísticas e o veredito padronizado. Arquivos de relatório, CSV/JSON e figuras ficam fora do escopo da v1, salvo revisão explícita desta especificação.

- **RF-007** — A v1 deve executar treinamento e avaliação localmente, exclusivamente para validação técnica. Ao final da execução, deve imprimir no terminal os resultados e o relatório previstos na seção 10.  
  *Critério de aceite:* a execução completa pode ser iniciada localmente e apresenta o relatório no terminal, sem depender de interface gráfica, serviço remoto, API ou publicação do modelo.

---

## 2. Princípios AI First e SDD

- **RF-001** — Todo requisito tem ID rastreável, com prefixo da categoria correspondente (`RF-`, `RT-`, `RM-`, `RV-` ou `RC-`) e critério de aceite verificável.
- **RF-002** — Agentes de IA implementam tarefas a partir desta spec e não podem alterá-la. Mudanças na especificação entram em commit separado, acompanhadas de justificativa.
- **RF-003** — Deve existir rastreabilidade entre spec, teste e código. Todo teste cita o ID do requisito que cobre (por exemplo: `# cobre: RV-004`).
- **RF-004** — Testes são escritos antes da implementação ou junto dela, sempre derivados dos critérios de aceite.
- **RF-005** — Todo prompt enviado a um agente inclui: ID da tarefa, seções relevantes da spec, arquivos permitidos para edição e definição de pronto (seção 9).
- **RF-006** — A saída de cada agente é revisada por uma pessoa antes do merge. Métricas reportadas por agentes são recomputadas por script versionado, nunca simplesmente copiadas.

---

## 3. Dados

| Fonte | Conteúdo | Uso |
|-------|----------|-----|
| Histórico oficial da Mega-Sena (Caixa) | Todos os concursos: número, data e seis dezenas | Série completa de treino e validação |
| `mega_da_virada.csv` | Mega da Virada de 2009 a 2025 | Identificação das Viradas por `is_virada` e avaliação principal |

- **RT-001** — Deve existir uma série única, ordenada por número de concurso. As Mega da Virada já fazem parte do histórico da Mega-Sena e não devem ser duplicadas: os concursos correspondentes apenas recebem `is_virada = 1`.
- **RT-002** — Os números dos concursos de 2023 e 2024 estão vazios no CSV de Mega da Virada. Devem ser preenchidos usando a data (31/12) e a fonte oficial da Caixa. Os demais registros do CSV também devem ser conferidos contra a fonte oficial.
- **RT-003** — As validações de qualidade devem interromper o pipeline em caso de falha: exatamente seis dezenas distintas por concurso; todas as dezenas no intervalo [1, 60]; números de concurso estritamente crescentes e sem lacunas; datas crescentes; ausência de concursos duplicados.
- **RT-004** — Os dados brutos são imutáveis. O dataset processado deve ser versionado por hash SHA-256, registrado em cada execução.

---

## 4. Codificação

**Representação do concurso.** Cada concurso é representado pelas dezenas ordenadas $d_1 < d_2 < \ldots < d_6$.

- **Entrada por passo da janela (61 features):** vetor multi-hot de 60 posições (1 se a dezena saiu no concurso daquele passo; 0 caso contrário) acrescido da variável `is_virada` daquele concurso.
- **Entrada estática (1 feature):** `alvo_virada`, que indica se o concurso a prever é uma Mega da Virada. Essa informação é conhecida antes do sorteio e, portanto, não constitui vazamento do resultado.
- **Janela:** W concursos consecutivos imediatamente anteriores ao concurso-alvo. W é hiperparâmetro, com padrão 20.
- **Alvo:** seis inteiros $y_i = d_i - 1$, em [0, 59], um por posição ordenada.
- **Features derivadas opcionais:** soma das dezenas, quantidade de pares e quantidade por faixa de dezena (1–10, 11–20 etc.). Devem ser padronizadas com média e desvio calculados apenas no conjunto de treino. Entram somente como ablação (RV-006).

- **RT-005** — A codificação deve ser uma função pura e determinística, com testes de ida e volta: codificar e depois decodificar deve devolver exatamente as dezenas originais.
- **RT-006** — Nenhuma janela pode conter o concurso-alvo nem concursos posteriores a ele. Deve existir teste automatizado obrigatório de vazamento temporal.

**Decodificação.** A saída do modelo consiste em seis distribuições softmax de 60 classes. A sequência final é a que maximiza a soma das log-probabilidades sob a restrição $d_1 < d_2 < \ldots < d_6$, calculada por programação dinâmica em O(6 · 60). É proibido escolher o argmax de cada cabeça isoladamente, pois isso pode gerar uma sequência inválida, com dezenas repetidas ou fora de ordem.

---

## 5. Modelo e treinamento

**Arquitetura de referência:**

```
sequência (W, 61)
    → LSTM(128, return_sequences=True)
    → Dropout(0.2)
    → LSTM(64)
    → concat(alvo_virada)
    → Dense(128, relu)
    → 6 × Dense(60, softmax)   # uma cabeça por posição ordenada
```

### Peso por amostra

- **RT-007** — Cada amostra (um concurso-alvo) recebe peso $w = 1.0$ se o alvo é Mega da Virada e $w = \alpha$ se o alvo é Mega-Sena regular, com $0 < \alpha < 1$.
- **RT-008** — A perda por amostra é a soma da entropia cruzada categórica esparsa das seis cabeças. A perda do lote é calculada por $\Sigma(w_i \cdot L_i) / \Sigma(w_i)$, mantendo a escala estável.
- **RT-009** — Os pesos são aplicados somente no treino. Validação e teste utilizam métricas sem peso.
- **RT-010** — Deve ser reportada a participação efetiva das amostras de Mega da Virada no peso total: $\Sigma w_{\text{virada}} / \Sigma w$. Com aproximadamente 17 Viradas e 2.900 concursos regulares, os valores de referência são:

| α    | Participação efetiva aproximada da Virada |
|------|-------------------------------------------|
| 0,25 | 2%                                        |
| 0,10 | 6%                                        |
| 0,05 | 10%                                       |
| 0,01 | 37%                                       |

> [!NOTE]
> Se o objetivo é que as Viradas influenciem de fato o ajuste, α precisa ser pequeno.

- **RT-011** — A grade de α é {0,01; 0,05; 0,10; 0,25}, com o controle α = 1,0 (sem diferenciação de peso).

### Hiperparâmetros e regras

| Item | Valor |
|------|-------|
| Otimizador | Adam, `learning_rate = 1e-3`, com `ReduceLROnPlateau` (fator 0,5; paciência 5) |
| Lote / épocas | 64 / até 100 |
| Parada antecipada | Paciência 10 sobre a perda de validação sem peso, restaurando os melhores pesos |
| Validação | Último 10% cronológico do conjunto de treino, sem embaralhar |
| Regularização | Dropout 0,2; `recurrent_dropout` testado na grade |
| Grade de busca | No máximo 20 configurações; W ∈ {10, 20, 40} combinado com os valores de α definidos para o experimento. Todas as configurações executadas devem ser registradas |
| Sementes | No mínimo 10 por configuração, fixando Python, NumPy e TensorFlow |

- **RT-012** — Embaralhar é permitido apenas entre lotes de treino já formados por janelas válidas. Nunca se deve embaralhar ou misturar amostras entre treino, validação e teste.
- **RT-013** — Os hiperparâmetros são escolhidos usando apenas dados anteriores ao ano de teste (validação aninhada) ou são fixados antes de consultar o teste. É proibido selecionar hiperparâmetros com base no desempenho do próprio teste.

---

## 6. Validação

- **RV-001** — O esquema principal é **walk-forward por ano de Mega da Virada**. Para cada ano Y de 2015 a 2025 (11 eventos), treina-se com todos os concursos anteriores à Mega da Virada Y e testa-se exclusivamente na Mega da Virada Y. As Viradas de 2009 a 2014 servem apenas como dados de treino no primeiro passo.
- **RV-002** — A avaliação secundária usa uma amostra maior: os últimos 300 concursos regulares como teste temporal, para verificar se o modelo aprende algo mensurável fora da pequena amostra de Viradas.
- **RV-003** — A previsão experimental para a Mega da Virada de 2026 (31/12/2026) usa o modelo treinado com toda a série disponível e deve ser registrada antes do sorteio, junto com o hash do modelo e do dataset. Não existe avaliação *ex ante* dessa previsão antes da realização do concurso.

### 6.1. Baselines obrigatórios

| ID | Baseline | Definição |
|----|----------|-----------|
| B0 | Aleatório uniforme | Seis dezenas uniformes, sem reposição, sorteadas por semente |
| B1 | Mais frequentes | As seis dezenas mais frequentes no conjunto de treino |
| B2 | Distribuição marginal empírica por posição | Distribuição empírica por posição ordenada, com a mesma decodificação do modelo |
| B3 | Concurso anterior | Repetir as seis dezenas do concurso imediatamente anterior |

### 6.2. Métricas principais

| ID | Métrica | Referência sob H₀ / comparação |
|----|---------|--------------------------------|
| M1 (primária) | Acertos por concurso: previsão ∩ sorteio | Distribuição hipergeométrica sob H₀ |
| M2 | NLL por cabeça e total, no teste, sem peso | Comparar com B2 |
| M3 | Acerto exato por posição ordenada | ≈ 1/60 por cabeça; comparação empírica obrigatória com B2 |
| M4 | Recall@10: quantidade de dezenas sorteadas entre as 10 maiores pontuações $p(d) = \Sigma_i\, p_i(d)$ | Esperado 1,0 sob H₀ |

**Distribuição exata de M1 sob H₀** (seis dezenas previstas e seis sorteadas, ambas sem reposição, em um universo de 60):

| Acertos | Probabilidade aproximada |
|---------|--------------------------|
| 0       | 51,59%                   |
| 1       | 37,90%                   |
| 2       | 9,48%                    |
| 3       | 0,99%                    |
| 4       | 0,043%                   |
| 5       | 0,00065%                 |
| 6       | 0,000002%                |

### 6.3. Métricas clássicas de classificação

Para calcular matriz de confusão e métricas derivadas, cada concurso é convertido em 60 classificações binárias: a dezena "saiu" ou "não saiu". A sequência decodificada marca exatamente seis dezenas como positivas.

- **RM-001** — A unidade de análise é o concurso. Calcula-se uma matriz 2×2 por concurso e agregam-se as células somando-as (micro) sobre os concursos de teste. Intervalos de confiança usam bootstrap sobre concursos, nunca sobre dezenas, porque as 60 dezenas de um mesmo concurso não são independentes.

**Matriz de confusão de um concurso com *k* acertos:**

|                     | Real: saiu   | Real: não saiu |
|---------------------|:------------:|:--------------:|
| **Previsto: saiu**      | VP = *k*     | FP = 6 − *k*  |
| **Previsto: não saiu**  | FN = 6 − *k* | VN = 48 + *k*  |

Como previsão e sorteio têm seis dezenas cada, FP = FN e as métricas clássicas tornam-se função de *k*:

| Métrica | Fórmula | Valor esperado sob H₀ (*k* = 0,6) |
|---------|---------|-----------------------------------|
| Precisão = Revocação = F1 | *k* / 6 | 0,100 |
| Especificidade | (48 + *k*) / 54 | 0,900 |
| Acurácia | (48 + 2*k*) / 60 | 0,820 |
| MCC (igual ao kappa de Cohen neste caso) | (5*k* − 3) / 27 | 0 |

- **RM-002** — Essas métricas devem ser reportadas, mas interpretadas somente em comparação com os baselines. Acurácia e especificidade são dominadas pelos verdadeiros negativos: o acaso já produz acurácia esperada de 82%. Precisão, revocação e F1 são idênticas, e MCC e kappa são transformações lineares de *k*. Portanto, nenhuma delas constitui um teste independente além de M1.

### 6.4. Métricas baseadas em probabilidades

Definir $p(d) = \min(1,\; \Sigma_i\, p_i(d))$, em que $p_i(d)$ é a probabilidade atribuída à dezena $d$ pela cabeça correspondente à posição $i$.

| ID | Métrica | Referência sob H₀ / comparação |
|----|---------|--------------------------------|
| M5 | ROC-AUC por concurso (60 scores), seguida da média | 0,5 |
| M6 | PR-AUC (precisão média) | 0,10 (prevalência de 6/60) |
| M7 | Brier score de $p(d)$ e Brier skill score: $1 - \text{Brier} / 0{,}09$ | 0,09 e 0 |
| M8 | Calibração: diagrama de confiabilidade (10 faixas) e ECE | Comparar com B2 |
| M9 | Acurácia top-1, top-3 e top-5 por cabeça (posição ordenada) | Comparar com B2 |
| M10 | Matriz de confusão posicional 60×60 por cabeça (classe prevista × classe real), em mapa de calor, e precisão/revocação por dezena | Descritiva |

- **RM-003** — Somente M1 e M2 entram no veredito RV-011 (hipóteses primárias). M3 a M10 são exploratórias e não podem ser usadas para declarar evidência de sinal.
- **RV-012** — Toda métrica clássica deve ser apresentada junto com o valor correspondente de B0 a B3, calculado do mesmo modo, IC de 95% por bootstrap sobre concursos e p-valor de permutação corrigido por Holm junto com RV-005. Métrica sem baseline não entra no relatório.
- **RV-013** — Métricas por dezena e matrizes por classe são descritivas: observar 60 dezenas gera falsos positivos por acaso. Uma "dezena favorita do modelo" só pode ser considerada após correção de múltiplas comparações e replicação na avaliação RV-002.
- **RV-014** — M5 a M8 nas 11 Viradas têm intervalos de confiança muito largos, pois são apenas 11 concursos, com seis dezenas positivas cada. Conclusões sobre discriminação e calibração devem vir da avaliação RV-002, com 300 concursos. A matriz posicional M10 só é interpretável nessa avaliação secundária.

---

## 7. Validade estatística

**Hipóteses pré-registradas.**

- Para **M1:** $H_0$: $E[\text{acertos}] = 0{,}6$; $H_1$: $E[\text{acertos}] > 0{,}6$ (unilateral). Nível de significância $\alpha_{\text{teste}} = 0{,}05$.
- Para **M2:** $H_0$ é que a NLL do modelo é maior ou igual à NLL de B2; a hipótese alternativa é que a NLL do modelo é menor que a de B2.

- **RV-004** — *Testes.* Usar teste de permutação ou teste exato contra a distribuição hipergeométrica para M1; teste de Wilcoxon pareado entre o modelo e cada baseline; e intervalo de confiança de 95% por bootstrap BCa para a diferença de médias. Reportar tamanho de efeito, não apenas p-valor.
- **RV-005** — *Múltiplas comparações.* Aplicar correção de Holm sobre todas as configurações testadas (W, α e controle). Todas as configurações devem ser reportadas. Escolher a melhor configuração depois de observar o teste invalida o resultado.
- **RV-006** — *Ablações.* Comparar α = 1,0 (sem diferenciação de peso) e versões com/sem features derivadas, usando os mesmos testes e correções.
- **RV-007** — *Sementes.* Reportar média ± desvio-padrão sobre no mínimo 10 sementes. A inferência utiliza a previsão agregada por semente (média das métricas), nunca a melhor semente isolada.
- **RV-008** — *Poder estatístico.* Com desvio de M1 aproximado de 0,70 por concurso e aproximação normal, o menor efeito detectável (80% de poder, $\alpha = 0{,}05$ bilateral) é aproximadamente:

| Amostra de teste | Efeito mínimo detectável em M1 |
|------------------|-------------------------------|
| 11 Viradas (RV-001) | ≈ 0,59 (acerto médio ≈ 1,19, quase o dobro do acaso) |
| 17 Viradas | ≈ 0,48 |
| 300 concursos regulares (RV-002) | ≈ 0,11 |

> [!WARNING]
> Conclusão obrigatória no relatório: com poucas Viradas, o teste principal só detecta efeitos enormes. Resultado não significativo **não prova** ausência de sinal.

- **RV-009** — *Verificações de sanidade.* Devem passar antes de qualquer conclusão:
  1. Treinar com rótulos permutados — M1 e M2 devem colapsar para o nível dos baselines;
  2. Treinar com entradas aleatórias — idem;
  3. Executar o teste unitário de vazamento RT-006.

  Se o modelo "vencer" e uma verificação falhar, deve-se assumir vazamento até prova em contrário.

- **RV-010** — *EDA prévia da aleatoriedade.* Calcular qui-quadrado de uniformidade das frequências por dezena, teste de sequências (runs) na soma dos concursos e autocorrelação das indicadoras por dezena, com correção para múltiplos testes. Reportar os resultados como contexto da hipótese, não como evidência preditiva isolada.
- **RV-011** — *Veredito padronizado.* O relatório classifica o resultado em uma destas três categorias:
  - **Evidência de sinal:** p-valor corrigido < 0,05 em M1 e M2, verificações de sanidade aprovadas e resultado replicado na avaliação secundária RV-002.
  - **Inconclusivo:** efeito observado maior que o acaso, mas abaixo do poder de detecção disponível.
  - **Sem evidência de sinal:** demais casos.

---

## 8. Convenções de código e reprodutibilidade

- **RC-001** — Python 3.11, TensorFlow 2.x / Keras e `tf.data` para janelas e lotes.
- **RC-002** — Tipagem estática (`mypy`), formatação e lint (`ruff`), testes (`pytest`) e cobertura mínima de 85% no pacote de dados e métricas.
- **RC-003** — Configuração em YAML versionado (sem constantes mágicas). Cada execução grava a configuração utilizada, hash do dataset, versões das bibliotecas, sementes e hash do commit.
- **RC-004** — Funções puras para codificação, decodificação, janelas e métricas. Efeitos colaterais (disco e rede) são isolados em módulos de I/O.
- **RC-005** — Na v1, o relatório da seção 10 é impresso no próprio terminal, em tabelas de texto, ao fim da execução. Arquivos CSV/JSON e figuras ficam para versões futuras.
- **RC-006** — Mensagens, logs e docstrings em português; nomes de identificadores em inglês.

**Estrutura sugerida do repositório:**

```
spec/                    # esta spec e decisões (ADRs)
configs/                 # YAML dos experimentos
src/lstm_virada/         # data.py, encoding.py, model.py, train.py,
                         #   decode.py, metrics.py, stats.py
tests/                   # espelha src/, com tags de requisito
data/
  raw/                   # dados brutos imutáveis
  processed/             # dataset processado e versionado por hash
reports/                 # reservado para versões futuras; não é saída da v1
```

---

## 9. Definição de pronto e tarefas

**Definição de pronto de uma tarefa:** testes que citam os IDs dos requisitos passam; `ruff` e `mypy` não apresentam erros; verificações de sanidade RV-009 passam; a execução é reproduzível com a mesma semente (diferença numérica dentro de tolerância documentada); a spec permanece inalterada ou é alterada em commit separado.

| Tarefa | Cobre | Entrega |
|--------|-------|---------|
| T1 — Ingestão e validação dos dados | RT-001 a RT-004 | `data.py` e testes de qualidade |
| T2 — Codificação e decodificação | RT-005 e RT-006 | `encoding.py`, `decode.py`, testes de ida e volta e de vazamento |
| T3 — Baselines B0 a B3 | RV-001 e RV-004 | `metrics.py` e resultados de referência |
| T4 — Modelo e treino com pesos | RT-007 a RT-013 | `model.py`, `train.py` e configurações |
| T5 — Walk-forward e métricas | RV-001 a RV-003, RM-001 a RM-003, RV-012 a RV-014 | Tabelas por ano e por semente |
| T6 — Estatística e sanidade | RV-004 a RV-011 | `stats.py` e relatório com veredito |
| T7 — Execução local e saída no terminal | RF-007, RC-003 e RC-005 | Orquestração da execução v1 e impressão do relatório completo no terminal |

---

## 10. Critérios de aceite do projeto

1. Todos os requisitos acima têm teste ou verificação automática associada.
2. O relatório final da v1, impresso no terminal, apresenta M1 a M10, incluindo as matrizes de confusão, para o modelo e para B0 a B3, por ano e em resultados agregados; apresenta ICs de 95%, p-valores corrigidos, tabela de poder estatístico e veredito RV-011.
3. O pipeline reproduz o relatório a partir do dataset versionado, sem etapas manuais não documentadas.
4. A conclusão do relatório respeita o poder estatístico informado em RV-008, sem afirmar além do que as amostras sustentam.
5. O treinamento e a avaliação da v1 são executados localmente, exclusivamente para validação técnica. Ao concluir, os resultados são apresentados no próprio terminal. A v1 não exige nem entrega interface gráfica, aplicação web, API, serviço remoto, treinamento em nuvem ou publicação/implantação do modelo.

---

## 11. Premissas e pontos em aberto

- A fonte e o formato exatos do histórico completo da Mega-Sena (download oficial da Caixa ou API) devem ser confirmados antes da implementação de T1. Os valores de α, W e a grade de busca são pontos de partida; alterações passam por revisão explícita da spec.
- A escolha de seis cabeças softmax é a decisão de projeto atual. Alternativas (por exemplo, saída multi-hot de 60 posições) podem entrar como ablação futura, mediante revisão da spec.
- A previsão para 31/12/2026 é um registro experimental, sem valor preditivo garantido. A v1 é um experimento local de validação técnica; os resultados não devem ser apresentados como recomendação de aposta ou evidência de capacidade de prever sorteios futuros sem que os critérios estatísticos desta spec sejam satisfeitos.

> [!CAUTION]
> Antes de congelar o pré-registro estatístico, reconciliar o teste unilateral definido para M1 ($H_1$: $E[\text{acertos}] > 0{,}6$) com o cálculo de poder de RV-008, atualmente descrito com nível bilateral. Também deve ser confirmada a referência teórica de M3 por posição ordenada: como as seis dezenas são ordenadas, cada cabeça não possui distribuição uniforme de classes. A comparação empírica com B2 deve prevalecer até que seja definida uma referência nula posicional adequada.
