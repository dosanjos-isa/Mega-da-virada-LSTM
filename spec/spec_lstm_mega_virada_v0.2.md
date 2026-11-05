# Spec SDD: LSTM para a Mega da Virada

**Versão:** 0.2 · **Status:** Proposta de Evolução Aprovada — Geração de Jogos e Inferência · **Data:** 04/11/2026  
**Stack:** Python 3.11 + TensorFlow/Keras + Pandas + JupyterLab · **Idioma:** pt-BR

> [!IMPORTANT]
> **Fonte de Verdade do Projeto:** Código, testes, notebooks, configurações e prompts para agentes de IA devem derivar estritamente desta especificação. Se houver divergência entre qualquer implementação e esta spec, **a spec prevalece** até que seja alterada por revisão formal registrada em commit separado.

---

## Histórico de Revisões e Changelog

| Versão | Data | Autor | Resumo das Alterações | Requisitos Afetados |
|:---:|:---:|:---:|---|---|
| **0.1** | 04/11/2026 | Matheus Pinheiro | Rascunho inicial consolidado: foco em treinamento, validação estatística walk-forward e decodificação determinística de sequência única. | RF-001 a RF-007, RT-001 a RT-013, RV-001 a RV-014, RM-001 a RM-003, RC-001 a RC-006. |
| **0.2** | 04/11/2026 | Matheus Pinheiro | **Evolução Operacional: Geração de Jogos e Inferência.** Formalização da geração de múltiplos jogos válidos e diversificados como produto operacional do modelo; desacoplamento entre treino e inferência; amostragem sequencial ponderada com controle de diversidade (`max_shared_numbers`); configuração YAML de geração; expansão dos critérios de avaliação científica para conjuntos de $K$ jogos; adição das tarefas T7 a T14. | Inclusão de **RG-001 a RG-012**; atualização de RC-004 e estrutura de módulos (`src/lstm_virada/generation/`); acréscimo de T7 a T14. |

---

## 1. Contexto e Objetivo

### 1.1. Objetivo Principal
Treinar uma rede neural LSTM utilizando o histórico completo de sorteios da Mega-Sena (atribuindo peso diferenciado aos concursos da Mega da Virada) e **utilizá-la operacionalmente para gerar jogos válidos de 6 dezenas para a Mega da Virada**.

A geração de jogos é o **resultado operacional esperado** do projeto. A modelagem não se encerra em métricas descritivas ou diagnósticos estáticos: o artefato final entregue pelo pipeline é um conjunto parametrizável de combinações geradas a partir das distribuições de probabilidade aprendidas pelo modelo.

### 1.2. Natureza Científica e Honestidade Estatística
Sorteios de loteria são processos estocásticos desenhados para serem independentes e espacialmente uniformes. A hipótese de que uma rede LSTM consiga extrair sinal preditivo persistente de tais séries é experimental.
- A especificação considera a conclusão **"sem evidência de sinal"** um resultado científico perfeitamente válido e esperado.
- A geração dos jogos reflete as representações e preferências de probabilidade aprendidas pela rede sobre os dados históricos, **sem qualquer garantia de acerto ou vantagem probabilística comprovada**.
- O sucesso do projeto é construir um pipeline correto, reprodutível, modular e estatisticamente honesto.

### 1.3. Fora de Escopo
Estratégias comerciais de apostas, promessa de retorno financeiro, outras modalidades de loteria, desenvolvimento de interfaces web/mobile, serviços em nuvem, APIs públicas e implantação em servidores remotos.

### 1.4. Escopo Operacional Local (v0.2)
Toda a execução do projeto é estritamente **local**, na máquina do desenvolvedor:
- **Treinamento e inferência:** Executados localmente via scripts Python ou orquestrados via Jupyter Notebooks.
- **Saídas:** Os resultados da validação e os jogos gerados são exibidos diretamente no terminal e renderizados em tabelas no Jupyter Notebook, além de gravados em arquivos de log/artefato para fins de auditoria e reprodutibilidade.

- **RF-007** — A execução de treino, inferência e geração de jogos é exclusivamente local. A geração deve apresentar os jogos no terminal e no ambiente Jupyter, sem depender de serviços externos ou infraestrutura remota.  
  *Critério de aceite:* A execução completa pode ser disparada localmente via linha de comando ou notebook e conclui a geração apresentando as combinações e validações na tela.

---

## 2. Princípios AI First e SDD

- **RF-001** — Todo requisito possui identificador rastreável com prefixo de categoria (`RF-` Funcionais, `RT-` Técnicos de Dados/Modelo, `RG-` Geração de Jogos, `RM-` Métricas, `RV-` Validação Estatística, `RC-` Convenções de Código) e critério de aceite verificável.
- **RF-002** — Agentes de IA implementam tarefas estritamente a partir desta spec e não podem alterá-la de forma autônoma. Alterações na especificação entram em commit separado com changelog.
- **RF-003** — Toda implementação e teste possui rastreabilidade com a spec. Testes unitários e de integração citam os requisitos cobertos (exemplo: `# cobre: RG-002, RG-003`).
- **RF-004** — Os testes derivam diretamente dos critérios de aceite e são desenvolvidos antes ou paralelamente à implementação da funcionalidade.
- **RF-005** — Todo prompt de trabalho para agentes inclui: ID da tarefa, seções relevantes da spec, arquivos permitidos para edição e definição de pronto (DoD).
- **RF-006** — Códigos e saídas produzidos por agentes são validados por revisão humana e por testes automatizados antes do merge.

---

## 3. Dados

| Fonte | Conteúdo | Uso no Pipeline |
|---|---|---|
| `data/raw/megasena-download-resultados.csv` | Histórico oficial de todos os concursos regulares e especiais da Caixa (1 a $N$) | Série temporal completa de treino, validação e inferência |
| `data/raw/mega_da_virada.csv` | Registros específicos dos concursos da Mega da Virada de 2009 a 2025 | Identificação das Viradas (`is_virada = 1`) e validação temporal walk-forward |
| `data/processed/dataset_megasena_virada_processed.csv` | Dataset consolidado, ordenado e validado, gerado pela Tarefa T1 | Fonte de entrada única para janelamento, treino e inferência |

- **RT-001** — Existe uma série temporal única, estritamente ordenada por número de concurso. As Mega da Virada já integram o histórico e não são duplicadas: os concursos correspondentes apenas recebem `is_virada = 1` e os regulares `is_virada = 0`.
- **RT-002** — Concursos da Mega da Virada foram conferidos contra a base da Caixa. Registros sem número de concurso (ex: 2023 e 2024) foram reconciliados pela data de 31/12 e dezenas sorteadas.
- **RT-003** — Validações de integridade barram o pipeline em caso de falha: exatamente 6 dezenas distintas por concurso; dezenas em $[1, 60]$; concursos estritamente contínuos e sem lacunas (1 a $N$); datas em ordem cronológica; ausência de duplicatas e nulos.
- **RT-004** — Os dados brutos em `data/raw/` são imutáveis. O dataset processado em `data/processed/` é versionado com hash SHA-256 obrigatório registrado em cada execução.

---

## 4. Codificação e Janelas Temporais

**Representação do sorteio:** Cada concurso é representado pelas 6 dezenas ordenadas $d_1 < d_2 < \ldots < d_6$.

- **Entrada sequencial por passo da janela (61 features):** Vetor multi-hot de 60 posições indicando a presença das dezenas naquele sorteio (1 se a dezena saiu, 0 caso contrário) acrescido do indicador binário `is_virada` daquele concurso.
- **Entrada estática de contexto (1 feature):** `alvo_virada`, indicando se o concurso a prever/gerar é uma Mega da Virada. Essa informação é conhecida antes do sorteio e não constitui vazamento de dados.
- **Janela temporal:** $W$ concursos consecutivos imediatamente anteriores ao concurso-alvo. $W \in \{10, 20, 40\}$, com padrão $W = 20$.
- **Alvos de treinamento:** 6 inteiros $y_i = d_i - 1$, onde $y_i \in [0, 59]$, representando a classe de cada uma das seis posições ordenadas.
- **RT-005** — A codificação e decodificação básica devem ser funções puras e determinísticas, com testes de ida e volta comprovando a restauração exata das dezenas.
- **RT-006** — Nenhuma janela temporal pode conter o concurso-alvo ou concursos posteriores a ele. O teste unitário contra vazamento temporal é de execução obrigatória no pipeline.

---

## 5. Arquitetura do Modelo e Treinamento com Pesos

### Arquitetura de Referência
```text
Entrada Sequencial: (batch_size, W, 61)
    → LSTM(128, return_sequences=True)
    → Dropout(0.2)
    → LSTM(64)
    → Concatenação com Entrada Estática alvo_virada (1 feature)
    → Dense(128, activation='relu')
    → 6 × Dense(60, activation='softmax')  # Uma cabeça independente por posição ordenada
```

### Treinamento Ponderado por Amostra
- **RT-007** — Cada amostra recebe peso $w_i = 1.0$ se o alvo é Mega da Virada e $w_i = \alpha$ se o alvo é concurso regular, com $\alpha \in \{0.01, 0.05, 0.10, 0.25\}$ e controle $\alpha = 1.0$.
- **RT-008** — A perda por amostra é a soma das entropias cruzadas categóricas esparsas das 6 cabeças: $L_i = \sum_{k=1}^6 \text{CCE}(y_{i,k}, \hat{p}_{i,k})$. A perda do lote é a média ponderada $\sum (w_i L_i) / \sum w_i$.
- **RT-009** — Os pesos $\alpha$ são aplicados exclusivamente durante o treinamento. Validação e testes utilizam métricas não ponderadas.
- **RT-010** — A participação efetiva das Viradas no peso total do treino deve ser reportada no relatório ($\Sigma w_{\text{virada}} / \Sigma w$).
- **RT-011** — Grade de busca de hiperparâmetros limitada a 20 combinações ($W \times \alpha$).
- **RT-012** — Embaralhamento restrito a lotes de treino já formados. Amostras temporais de treino, validação e teste jamais são misturadas.
- **RT-013** — Hiperparâmetros são fixados antes do teste ou ajustados via validação aninhada temporal estrita.

---

## 6. Geração de Jogos e Inferência — RG

Esta seção formaliza a evolução introduzida na versão 0.2: a utilização da LSTM treinada para **produzir conjuntos de jogos para a Mega da Virada**.

### 6.1. Filosofia e Desacoplamento entre Treino e Inferência
- **Desacoplamento estrito:** A inferência e geração de jogos não exigem novo treinamento da rede. O módulo de geração consome um modelo previamente treinado e salvo em disco (formato `.keras`), aplicando-o sobre a janela histórica mais recente ou sobre uma janela histórica de teste.
- **Entrada da inferência:** Janela mais recente de $W$ concursos validada em `data/processed/`, com `alvo_virada = 1`.
- **Saída da rede:** Matriz de probabilidades $\mathbf{P} \in \mathbb{R}^{6 \times 60}$, onde $\mathbf{P}_{i, d}$ representa a probabilidade atribuída pela cabeça $i$ à dezena $d \in \{1, \dots, 60\}$ (com índice $d-1$).

### 6.2. Estratégias de Decodificação e Amostragem

#### Estratégia 1: Decodificação de Máxima Verossimilhança (Top-1 / Best Sequence)
Utiliza programação dinâmica em $\mathcal{O}(6 \times 60)$ para encontrar a combinação única que maximiza a soma das log-probabilidades sob a restrição estrita de ordenação:
$$\max_{1 \le d_1 < d_2 < \ldots < d_6 \le 60} \sum_{i=1}^6 \log \mathbf{P}_{i, d_i}$$
*É expressamente proibido utilizar o argmax isolado de cada cabeça, pois isso gera repetições e violações de ordem.*

#### Estratégia 2: Amostragem Ponderada Sequencial (Weighted Sampling) para Múltiplos Jogos
Para gerar $K$ jogos distintos, adota-se amostragem autorregressiva/condicionada na ordem das cabeças:
1. Para a primeira cabeça ($i=1$), a distribuição sobre $d_1$ é truncada no intervalo $[1, 55]$ e normalizada:
   $$P_{\text{amostragem}}(d_1 = c) \propto \mathbf{P}_{1, c}^{1/T}, \quad c \in [1, 55]$$
2. Para cada posição subsequente $i \in \{2, \ldots, 6\}$, dada a dezena $d_{i-1}$ sorteada na posição anterior, trunca-se a distribuição da cabeça $i$ estritamente para o intervalo $[d_{i-1} + 1, 60 - (6 - i)]$:
   $$P_{\text{amostragem}}(d_i = c \mid d_{i-1}) \propto \mathbf{P}_{i, c}^{1/T}, \quad c \in [d_{i-1} + 1, 60 - (6 - i)]$$
   onde $T > 0$ é a temperatura de amostragem configurável ($T = 1.0$ por padrão; $T < 1.0$ concentra nas maiores probabilidades; $T > 1.0$ aumenta a exploração).
3. **Propriedade matemática:** Por construção, toda combinação gerada satisfaz $1 \le d_1 < d_2 < \ldots < d_6 \le 60$ e não possui dezenas repetidas.

### 6.3. Mecanismo de Diversidade Controlada
Para evitar que uma geração de $K = 20$ jogos produza apostas praticamente idênticas (com 5 dezenas iguais, variando apenas uma), o sistema implementa um filtro configurável de **distância entre jogos por contagem de dezenas compartilhadas**:
- **Critério de aceitação:** Um novo jogo candidato $J_{\text{candidato}}$ só é aceito no conjunto de jogos gerados se:
  $$|J_{\text{candidato}} \cap J_m| \le \text{max\_shared\_numbers}, \quad \forall J_m \in \{J_1, \ldots, J_{k-1}\}$$
- **Configuração:** O parâmetro `max_shared_numbers` $\in \{2, 3, 4, 5\}$ (padrão: 4) define o teto de dezenas coincidentes entre qualquer par de jogos da mesma geração.
- **Controle de teto de tentativas (`max_attempts_factor`):** Se as restrições de diversidade impedirem a geração de novas combinações após $\text{quantity} \times \text{max\_attempts\_factor}$ tentativas (padrão fator 200), o gerador emite um aviso estruturado e devolve as combinações válidas encontradas até o momento, prevenindo laços infinitos.

### 6.4. Requisitos Funcionais de Geração (RG)

- **RG-001** — O sistema deve gerar jogos utilizando um modelo LSTM treinado previamente carregado de disco, sem exigir retreinamento da rede a cada execução de inferência.  
  *Critério de aceite:* A chamada do gerador carrega o modelo serializado e executa a geração em tempo inferior a 5 segundos para até 50 jogos.
- **RG-002** — Cada jogo gerado deve conter exatamente seis dezenas distintas no intervalo fechado $[1, 60]$.  
  *Critério de aceite:* Nenhuma combinação contém menos ou mais de 6 números, nenhum número repete dentro do jogo e todas as dezenas pertencem a $\{1, 2, \ldots, 60\}$.
- **RG-003** — As dezenas de cada jogo gerado devem ser estritamente ordenadas de forma crescente ($N1 < N2 < N3 < N4 < N5 < N6$).  
  *Critério de aceite:* A verificação `all(jogo[i] < jogo[i+1] for i in range(5))` retorna `True` para 100% dos jogos gerados.
- **RG-004** — A quantidade de jogos a gerar ($K$) deve ser configurável via parâmetro e arquivo YAML (ex: 5, 10, 20), não podendo ser um valor fixado no código-fonte.  
  *Critério de aceite:* Executar o gerador com $K=5$ produz 5 jogos; com $K=20$ produz 20 jogos válidos.
- **RG-005** — O processo de geração deve utilizar diretamente as probabilidades produzidas pelas seis cabeças softmax da rede neural.  
  *Critério de aceite:* A amostragem utiliza as matrizes de saída da LSTM ponderadas pela temperatura configurada.
- **RG-006** — O sistema deve impedir a geração de jogos duplicados na mesma rodada.  
  *Critério de aceite:* O conjunto de jogos gerados possui exatamente $K$ tuplas únicas; `len(set(jogos)) == K`.
- **RG-007** — O processo de geração deve permitir controle estrito por semente aleatória (`seed`).  
  *Critério de aceite:* Duas execuções com a mesma seed, mesmo modelo e mesma configuração geram exatamente a mesma lista de jogos em ordem idêntica.
- **RG-008** — O sistema deve oferecer mecanismo configurável de diversidade baseado no limite máximo de dezenas compartilhadas (`max_shared_numbers`).  
  *Critério de aceite:* Com `max_shared_numbers = 4`, nenhum par de jogos no conjunto gerado compartilha 5 ou 6 dezenas.
- **RG-009** — O sistema deve registrar metadados completos de auditoria para cada geração de jogos (timestamp, hash do modelo, hash do dataset, seed, parâmetros de diversidade e jogos gerados).  
  *Critério de aceite:* O artefato de log/YAML gravado permite rastrear a origem exata de cada combinação gerada.
- **RG-010** — O sistema deve permitir reproduzir com exatidão qualquer geração anterior informando os mesmos metadados e seed.  
  *Critério de aceite:* Execução de teste de idempotência confirma reprodução exata dos bilhetes gerados no passado.
- **RG-011** — O sistema deve validar matematicamente todos os jogos antes de apresentá-los ou persisti-los, levantando exceção caso alguma combinação seja inválida.  
  *Critério de aceite:* Função de validação pura rejeita qualquer lista de jogos com tamanho inadequado, dezenas fora da faixa ou duplicatas.
- **RG-012** — O sistema deve apresentar os jogos formatados de maneira legível e profissional no terminal e no Jupyter Notebook (tabelas formatadas e destaque das dezenas).  
  *Critério de aceite:* Impressão no terminal exibe formato padronizado `JOGO XX: D1 - D2 - D3 - D4 - D5 - D6` e relatório de validação.

### 6.5. Configuração YAML de Geração (`configs/generation_default.yaml`)
A configuração da geração é totalmente separada da configuração de treinamento:

```yaml
# configs/generation_default.yaml
# Configuração oficial para geração de jogos da Mega da Virada (RG-001 a RG-012)

generation:
  model_path: "models/lstm_virada_best.keras"
  dataset_path: "data/processed/dataset_megasena_virada_processed.csv"
  quantity: 10                     # Quantidade de jogos a gerar (RG-004)
  seed: 42                         # Semente para reprodutibilidade (RG-007)
  strategy: "weighted_sampling"    # Opções: "weighted_sampling" ou "dynamic_programming_top1"
  temperature: 1.0                 # Calibração estocástica (> 0.0)
  avoid_duplicates: true           # Garantia de zero jogos repetidos (RG-006)
  
  diversity:
    enabled: true                  # Ativação do filtro de diversidade (RG-008)
    max_shared_numbers: 4          # Máximo de dezenas compartilhadas entre quaisquer 2 jogos
    max_attempts_factor: 200       # Fator multiplicador de tentativas máximas de amostragem

  output:
    terminal: true                 # Exibição formatada no terminal
    notebook: true                 # Renderização em tabela formatada no Jupyter
    save_to_file: true             # Gravação de artefato de auditoria
    output_dir: "reports/generated_games"
```

---

## 7. Validação Walk-Forward e Avaliação Científica

### 7.1. Esquema Principal Walk-Forward (RV-001)
- Walk-forward cronológico por ano de Mega da Virada de 2015 a 2025 (11 eventos de validação).
- Para cada ano $Y$, treina-se exclusivamente com concursos anteriores à Mega da Virada de $Y$ e avalia-se na Mega da Virada $Y$. As Viradas de 2009 a 2014 são usadas apenas como treino inicial.

### 7.2. Baselines Obrigatórios (RV-004)
- **B0 (Aleatório uniforme):** Amostragem de 6 dezenas uniformes sem reposição por semente.
- **B1 (Mais frequentes):** As 6 dezenas mais frequentes na história até aquele momento.
- **B2 (Distribuição marginal empírica por posição):** Decodificação por DP da frequência histórica por posição ordenada.
- **B3 (Concurso anterior):** Repetição das 6 dezenas do concurso regular imediatamente anterior.

### 7.3. Avaliação Científica de Conjuntos de $K$ Jogos (Métrica Expandida para RG)
Quando o sistema gera um conjunto de $K$ jogos (ex: $K=10$ ou $K=20$), a avaliação científica contra o concurso de teste deve seguir regras rígidas:
1. **Paridade Amostral Estrita:** É expressamente proibido comparar um conjunto de $K$ jogos da LSTM contra apenas 1 jogo aleatório. Todo baseline comparador deve receber **exatamente a mesma quantidade $K$ de jogos gerados sob as mesmas restrições**.
2. **Métricas de Conjunto:**
   - **Média de acertos do portfólio:** $\bar{A} = \frac{1}{K} \sum_{j=1}^K |J_j \cap S|$, onde $S$ é o sorteio real.
   - **Melhor acerto do portfólio:** $A_{\max} = \max_{j=1 \dots K} |J_j \cap S|$.
   - **Contagem de faixas premiadas:** Quantidade de Quadras (4 acertos), Quinas (5) e Senas (6) obtidas no conjunto de $K$ bilhetes.
3. **Prevenção contra Pseudorreplicação:** Os $K$ jogos gerados para um mesmo concurso não são eventos estatisticamente independentes. O teste de hipótese pareado deve ser conduzido sobre os concursos de validação temporal (walk-forward), agregando o desempenho do portfólio por evento, e nunca tratando os $11 \times K$ jogos como observações independentes.

### 7.4. Previsão Prospectiva para a Mega da Virada de 2026 (RV-003 / RG-010)
A geração experimental de jogos para o sorteio de 31/12/2026 deve ser executada com o modelo treinado em toda a base disponível e seus metadados (jogos, seed, hash do dataset, hash do modelo e configuração) devem ser **gravados e versionados antes da realização do sorteio**. É vedado qualquer ajuste retrospectivo pós-sorteio.

---

## 8. Convenções de Código e Arquitetura Modular

### Estrutura de Pacotes (`src/lstm_virada/`)
A lógica de negócio é organizada em módulos independentes de responsabilidade única:

```text
src/lstm_virada/
├── __init__.py
├── data.py              # Ingestão, validação de integridade e exportação (T1 - concluída)
├── encoding.py          # Janelas temporais W, vetorização multi-hot e features (T2)
├── decode.py            # Decodificação ótima de máxima verossimilhança por DP (T2)
├── model.py             # Arquitetura LSTM 6 cabeças, compilação, save/load (T4)
├── train.py             # Loop de treinamento com pesos α, callbacks e walk-forward (T4)
├── generation.py         # Geração de múltiplos jogos, amostragem e diversidade (T8)
├── metrics.py           # Baselines B0..B3 e métricas M1..M10 (T3, T5)
└── stats.py             # Testes de permutação, bootstrap, Holm e veredito RV-011 (T6)
```

- **RC-001** — Python 3.11, TensorFlow 2.x / Keras e `tf.data`.
- **RC-002** — Tipagem estática rigorosa (`mypy`), linter e formatação (`ruff`), suíte de testes (`pytest`) com cobertura mínima de 85% em dados, decodificação, geração e métricas.
- **RC-003** — Configurações em arquivos YAML versionados (sem constantes mágicas no código).
- **RC-004** — Funções puras e isoladas para codificação, decodificação, amostragem e validação de jogos. Operações com I/O (disco/rede) são estritamente isoladas.
- **RC-005** — Apresentação dos jogos e relatórios no terminal e em tabelas do Jupyter Notebook.
- **RC-006** — Logs, mensagens e docstrings em português; identificadores e variáveis em inglês.

---

## 9. Limitações Estatísticas Obrigatórias

Toda documentação, tela de execução no terminal e célula de conclusão nos notebooks deve explicitar formalmente:
1. **Independência estocástica:** Sorteios da Mega-Sena são projetados e auditados para serem processos i.i.d. (independentes e identicamente distribuídos).
2. **Equiprobabilidade teórica:** Em um globo justo com 60 bolas, todas as $\binom{60}{6} = 50.063.860$ combinações possíveis possuem exatamente a mesma probabilidade teórica de ocorrência ($1 / 50.063.860 \approx 2 \times 10^{-8}$).
3. **Ausência de garantia de ganho:** O treinamento de modelos neurais sobre séries temporais de loterias visa capturar eventuais desvios empíricos ou artefatos mecânicos sutis de forma experimental, mas **não confere aumento comprovado de probabilidade nem garantia de ganho financeiro**.
4. **Alerta de aposta:** A geração de jogos destina-se a fins estritamente educacionais, técnicos e de pesquisa em Machine Learning.

---

## 10. Matriz de Tarefas do Projeto (T1 a T14)

A esteira de desenvolvimento do projeto é organizada em tarefas rastreáveis e sequenciais:

| ID | Tarefa | Requisitos Cobertos | Entregas Principais | Status |
|:---:|---|---|---|:---:|
| **T0** | Estruturação inicial do projeto e ambiente Jupyter | RF-001, RC-001, RC-002 | Diretórios, notebooks, pyproject.toml, requirements.txt, .gitignore | ✅ Concluída |
| **T1** | Ingestão e validação estrita dos dados | RT-001 a RT-004 | `data.py`, `test_data.py` (18 testes), dataset processado com SHA-256 | ✅ Concluída |
| **T2** | Codificação e decodificação determinística | RT-005, RT-006 | `encoding.py`, `decode.py`, testes de ida-e-volta e teste contra vazamento | ⏳ Pendente |
| **T3** | Implementação dos baselines B0 a B3 | RV-001, RV-004 | `metrics.py` com geradores de referência B0, B1, B2 e B3 | ⏳ Pendente |
| **T4** | Arquitetura LSTM e treinamento com pesos | RT-007 a RT-013 | `model.py`, `train.py`, suporte a pesos $\alpha$ por amostra | ⏳ Pendente |
| **T5** | Validação Walk-Forward e métricas primárias | RV-001 a RV-003, RM-001 a RM-003, RV-012 a RV-014 | Pipeline walk-forward 2015–2025, tabelas de métricas M1 a M10 | ⏳ Pendente |
| **T6** | Testes estatísticos, sanidade e veredito | RV-004 a RV-011 | `stats.py`, testes de permutação, bootstrap BCa, sanidade e veredito | ⏳ Pendente |
| **T7** | Especificação formal dos requisitos de geração (RG) | RG-001 a RG-012 | Spec v0.2, `ADR-001`, matriz de rastreabilidade | 🎯 Em Conclusão |
| **T8** | Implementação do módulo de geração e diversidade | RG-002 a RG-008, RG-011 | `src/lstm_virada/generation.py` (amostragem ponderada e filtro de diversidade) | ⏳ Pendente |
| **T9** | Integração da geração com o modelo treinado | RG-001, RG-005 | Pipeline de inferência desacoplado alimentado por modelo salvo | ⏳ Pendente |
| **T10** | Configuração YAML e registro de gerações | RG-004, RG-007, RG-009, RG-010 | `configs/generation_default.yaml`, logger estruturado de auditoria | ⏳ Pendente |
| **T11** | Suíte de testes automatizados para geração | RG-001 a RG-012 | `tests/test_generation.py` cobrindo 14 cenários com tags rastreáveis | ⏳ Pendente |
| **T12** | Integração da geração no Jupyter Notebook | RG-012, RF-007 | `notebooks/03_experimentos.ipynb` executando inferência e exibição | ⏳ Pendente |
| **T13** | Avaliação walk-forward de conjuntos de jogos | RV-001, RV-004, RG-005 | Avaliação comparativa de $K$ jogos da LSTM vs $K$ jogos aleatórios | ⏳ Pendente |
| **T14** | Geração prospectiva oficial para a Mega da Virada 2026 | RV-003, RG-009, RG-010 | Relatório assinado com jogos gerados, seed e hashes antes de 31/12/2026 | ⏳ Pendente |

---

## 11. Bateria de Testes Automatizados para Geração (T11)

A implementação de `generation.py` e `test_generation.py` deve conter cobertura obrigatória para os seguintes 14 cenários de teste:

1. **Validação de 6 dezenas por jogo:** Garante `len(jogo) == 6` para todos os jogos (`RG-002`).
2. **Impedimento de dezenas repetidas:** Garante `len(set(jogo)) == 6` (`RG-002`).
3. **Intervalo estrito $[1, 60]$:** Nenhuma dezena $< 1$ ou $> 60$ (`RG-002`).
4. **Ordenação crescente:** Garante $N1 < N2 < N3 < N4 < N5 < N6$ (`RG-003`).
5. **Zero duplicatas na geração:** Garante que todas as $K$ combinações sejam únicas (`RG-006`).
6. **Controle de quantidade $K$:** Testa geração com $K=5$, $K=10$ e $K=20$ (`RG-004`).
7. **Reprodutibilidade por seed:** Mesma seed produz jogos idênticos (`RG-007, RG-010`).
8. **Variabilidade entre seeds:** Seeds diferentes produzem conjuntos distintos (`RG-007`).
9. **Validade da amostragem ponderada:** Dezenas com probabilidade zero não são sorteadas (`RG-005`).
10. **Validação do filtro de diversidade:** Nenhum par de jogos compartilha mais do que `max_shared_numbers` dezenas (`RG-008`).
11. **Imutabilidade do modelo:** A inferência e geração não alteram os pesos do modelo (`RG-001`).
12. **Inexistência de vazamento temporal:** A inferência utiliza apenas a janela anterior ao concurso de teste (`RT-006, RG-001`).
13. **Robustez a probabilidades degeneradas:** Lançamento de erro claro se matriz de probabilidades contiver nulos, NaNs ou soma nula (`RG-011`).
14. **Tratamento de saturação do filtro de diversidade:** O gerador emite aviso seguro e devolve as combinações válidas se o teto de tentativas for atingido (`RG-008`).

---

## 12. Critérios de Aceite Globais da Versão 0.2

1. Todos os requisitos da v0.1 continuam plenamente preservados e rastreáveis.
2. Os novos requisitos **RG-001 a RG-012** possuem especificações funcionais e técnicas claras com critérios de aceite determinísticos.
3. O módulo de geração opera de forma desacoplada do treinamento, podendo carregar qualquer modelo `.keras` previamente treinado.
4. O sistema gera a quantidade solicitada de jogos ($K$), garantindo ausência de duplicatas, respeito à diversidade (`max_shared_numbers`) e ordenação $N1 < \ldots < N6$.
5. Cada geração é determinística sob semente (`seed`) e registra metadados completos em arquivo para auditoria.
6. A avaliação walk-forward do portfólio de jogos respeita a paridade amostral ($K$ jogos da LSTM vs $K$ jogos de cada baseline) e veda pseudorreplicação.
7. As limitações estatísticas e alertas éticos constam explicitamente no documento e nas saídas do sistema.

