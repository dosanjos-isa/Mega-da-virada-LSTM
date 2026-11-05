# ADR-001: Arquitetura de Geração de Jogos e Mecanismo de Diversidade

**Status:** Aprovado  
**Data:** 04/11/2026  
**Contexto:** Especificação SDD v0.2 — Requisitos RG-001 a RG-012

---

## 1. Contexto do Problema
O objetivo operacional do projeto evoluiu de uma análise puramente preditiva/estatística para a **geração efetiva de conjuntos de jogos (ex: 5, 10, 20 combinações) para a Mega da Virada**, orientados pelas representações aprendidas pela rede LSTM.

A arquitetura do modelo fornece 6 distribuições softmax sobre 60 classes (uma por posição ordenada $d_1 < \dots < d_6$). Enfrentamos dois desafios fundamentais:
1. **Validade Matemática Estrita:** Uma combinação de loteria não pode conter dezenas repetidas e deve ser estritamente crescente ($N1 < N2 < N3 < N4 < N5 < N6$).
2. **Diversidade Operacional:** Na geração de múltiplos bilhetes (ex: $K=20$), uma amostragem ingênua baseada apenas nas maiores probabilidades geraria apostas quase idênticas (trocando apenas 1 dezena), o que reduz a utilidade prática do conjunto de bilhetes.

---

## 2. Decisões Arquiteturais

### Decisão 1: Desacoplamento entre Treinamento e Inferência
- **Decisão:** A inferência e a geração de jogos residem em módulo próprio (`src/lstm_virada/generation.py`), consumindo modelos treinados serializados em formato `.keras`.
- **Justificativa:** Permite gerar novos jogos, testar diferentes quantidades ($K$), sementes e hiperparâmetros de diversidade em segundos, sem exigir novo treinamento dispendioso da rede a cada execução.

### Decisão 2: Estratégia de Amostragem Sequencial Condicionada (Weighted Sampling)
- **Decisão:** A geração estocástica de múltiplos jogos utiliza amostragem autorregressiva/condicionada na ordem das posições:
  - $d_1$ é amostrado da cabeça 1 truncada em $[1, 55]$.
  - $d_i$ é amostrado da cabeça $i$ truncada estritamente no intervalo $[d_{i-1} + 1, 60 - (6 - i)]$.
- **Justificativa:** Elimina a possibilidade de combinações inválidas (com dezenas repetidas ou fora de ordem) por construção matemática direta, preservando o formato de probabilidades aprendido pelo modelo em cada cabeça posicional.

### Decisão 3: Mecanismo de Diversidade por Limite de Dezenas Compartilhadas (`max_shared_numbers`)
- **Decisão:** Cada novo jogo candidato $J_{\text{candidato}}$ só é aceito no conjunto de jogos se não compartilhar mais do que `max_shared_numbers` dezenas (padrão: 4) com qualquer outro jogo já gerado na mesma rodada:
  $$|J_{\text{candidato}} \cap J_m| \le \text{max\_shared\_numbers}, \quad \forall m < k$$
- **Justificativa:** É uma métrica interpretável diretamente no contexto da Mega-Sena (compartilhar 4 números significa que os bilhetes coincidem no máximo em uma Quadra), garantindo espalhamento inteligente das apostas sem distorcer as probabilidades da rede com penalizações heurísticas arbitrárias.

### Decisão 4: Paridade Amostral Estrita na Avaliação Científica
- **Decisão:** Ao avaliar o portfólio de $K$ jogos da LSTM contra sorteios históricos ou prospectivos, a comparação é feita obrigatoriamente contra baselines recebendo **exatamente o mesmo número $K$ de bilhetes** (ex: 20 jogos da LSTM vs 20 jogos aleatórios).
- **Justificativa:** Impede comparações desonestas onde um conjunto volumoso de palpites é comparado contra um palpite isolado, mantendo a integridade metodológica da SDD.

---

## 3. Consequências e Trade-offs

### Positivas:
- **Reprodutibilidade:** Garantida por semente (`seed`) determinística.
- **Validade garantida:** 100% dos jogos gerados cumprem as regras da Caixa sem necessidade de correções manuais.
- **Flexibilidade:** Quantidade $K$ e temperatura $T$ totalmente parametrizáveis via arquivo YAML.

### Trade-offs:
- Valores excessivamente restritivos de diversidade (ex: `max_shared_numbers = 2` para $K=50$) podem saturar o espaço de combinações de alta probabilidade. Nesses casos, o sistema conta com um limitador de tentativas (`max_attempts_factor`) que encerra a busca de forma segura com aviso estruturado.

