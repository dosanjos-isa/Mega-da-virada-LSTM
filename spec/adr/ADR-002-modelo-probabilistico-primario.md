# ADR-002: Modelo Probabilístico Primário e Papel da LSTM

**Status:** Proposto — aguardando aprovação (Gate 0)  
**Data:** 05/11/2026  
**Contexto:** `spec/math_spec.md` v0.3.0-draft; correções ao parecer anterior e à `spec_lstm_mega_virada_v1.md`  
**Substitui:** a decisão implícita de usar a LSTM de 6 cabeças como modelo principal  
**Não substitui:** `ADR-001` (geração e diversidade), que permanece válido para a operação, mas tem sua Decisão 2 corrigida pela §4.4 e §9.3 da `math_spec.md`

---

## 1. Contexto

### 1.1 O que estava sendo proposto

A `spec_lstm_mega_virada_v1.md` define como arquitetura de referência uma LSTM que produz seis distribuições softmax independentes, uma por posição ordenada $d_1 < \dots < d_6$. O treinamento minimiza a soma das entropias cruzadas categóricas das seis cabeças. A decodificação usa programação dinâmica para escolher a sequência crescente de maior produto de probabilidades.

O parecer anterior recomendou substituir esse modelo por Plackett–Luce (PL). A revisão técnica solicitada pelo usuário mostrou que essa recomendação, como estava escrita, era matematicamente incompleta.

### 1.2 Os três problemas que motivam esta decisão

**(a) A ordem real do sorteio não é observada.** O dataset guarda as dezenas em ordem crescente. A fórmula sequencial de PL aplicada diretamente a $(d_1,\dots,d_6)$ calcula a probabilidade de uma ordem que não foi observada. A verossimilhança correta do conjunto exige marginalizar sobre as $6! = 720$ permutações.

**(b) A LSTM de 6 cabeças não é PL.** Cada cabeça é condicionada apenas no histórico, não no prefixo de dezenas já escolhidas. Portanto, a LSTM não é um modelo sequencial de Plackett–Luce. Ela é um **produto posicional** (M3), e só se torna uma distribuição válida sobre conjuntos depois de normalizada por $Z$. O treinamento com CCE estima marginais posicionais, não a likelihood do conjunto.

**(c) Comparar NLL sem normalização é inválido.** Produtos de marginais não somam 1 sobre o espaço de conjuntos. Qualquer comparação de NLL entre modelos precisa garantir o mesmo $\Omega$ e a mesma normalização.

### 1.3 O que o projeto precisa

Um modelo primário que:

1. seja uma distribuição válida sobre conjuntos de 6 dezenas;
2. tenha likelihood de conjunto computável e comparável;
3. seja simples o bastante para não overfittar com 3.074 concursos;
4. tenha interpretação clara;
5. permita um baseline da mesma família, para isolar o ganho do ajuste;
6. não dependa de uma arquitetura neural para existir.

---

## 2. Decisão

### 2.1 Modelo primário: M1 — Plackett–Luce com marginalização sobre permutações

Adota-se como modelo primário o **M1** da `math_spec.md`:

$$
P_{\text{M1}}(S)
=
\sum_{\pi \in \operatorname{Perm}(S)}
\prod_{j=1}^{6}
\frac{w_{\pi_j}}{W - \sum_{l<j} w_{\pi_l}},
\qquad
W = \sum_{i=1}^{60} w_i.
$$

- 60 pesos ($\theta_i = \log w_i$, com $\sum_i \theta_i = 0$; 59 graus de liberdade efetivos).
- Sem uso de histórico: é um modelo estático de pesos por dezena.
- Ajuste por máxima verossimilhança de conjunto, com soma em log-espaço sobre 720 permutações.
- Baseline da mesma família: **B2b** (M1 com $w_i \propto$ frequência empírica).
- Piso: **M0** (uniforme).
- Amostragem exata: sequência PL sucessiva, seguida de ordenação.
- Modo: não tem DP exato; usar beam search rotulado como heurístico.
- **Escopo da inferência:** M1 é estático. H1 (M1 vs B0) testa estrutura global de frequência contra o uniforme; H2 (M1 vs B2b) testa se o ajuste PL agrega sobre frequências empíricas; ambos não testam dependência temporal. A hipótese temporal/neural é H3/S1 (M3 vs B2a), secundária.

### 2.2 Modelo de sensibilidade: M2 — Fisher

Adota-se **M2** como análise de sensibilidade:

$$
P_{\text{M2}}(S) = \frac{\prod_{i \in S} w_i}{e_6(w)}.
$$

- Mesmos 60 pesos (59 graus de liberdade efetivos).
- Normalização por $e_6$ em $O(60 \cdot 6)$.
- Modo exato: as 6 maiores dezenas por peso.
- Amostragem exata por recursão condicional.
- Serve para verificar se a conclusão sobre sinal depende da forma funcional de M1.

### 2.3 Ablações neurais: M3, M4 e M5

- **M3** — produto posicional normalizado por $Z$, com $P_j$ estimadas por rede. É a formalização correta da LSTM de 6 cabeças.
- **M4** — PL neural sequencial, condicionado no prefixo. É a alternativa neural mais próxima de M1, mas cara (720 avaliações de rede por concurso).
- **M5** — a LSTM de 6 cabeças da v1, mantida como ablação:
  - **M5-SET**: treinada com a likelihood de conjunto ($-\sum_j \log P_{t,j}(d_{t,j}) + \log Z_t$), comparável a M1/B2b por NLL.
  - **M5-CCE**: treinada com soma de CCEs, comparável apenas a B2a-CCE (pseudo-likelihood).

Nenhuma ablação neural é modelo principal. Elas existem para responder: "uma arquitetura de Deep Learning extrai algo além do modelo estático e dos baselines?"

### 2.4 O que não muda

- A LSTM continua no projeto, mas como experimento de Deep Learning e ablação.
- O walk-forward por Virada, o pré-registro, a correção de Holm, as sanidades e a paridade de $K$ continuam obrigatórios.
- A conclusão "sem evidência de sinal" continua sendo um resultado válido.

---

## 3. Alternativas consideradas

| Alternativa | Prós | Contras | Decisão |
|---|---|---|---|
| **M0 uniforme** | Trivial, é o piso | Não modela nada | Piso, não modelo |
| **M1 PL marginalizado** | Interpretação física (urna com pesos), likelihood de conjunto, 60 parâmetros, amostragem exata | 720 permutações por concurso; modo sem DP exato | **Primário** |
| **M2 Fisher** | Likelihood de conjunto barata, modo exato, amostragem exata | Interpretação menos direta do mecanismo físico; forma funcional diferente | **Sensibilidade** |
| **M3 produto posicional** | Formaliza a LSTM de 6 cabeças; DP exato para $Z$ e para o modo; barato | Não modela dependência entre posições; se treinado com CCE, é pseudo-likelihood | **Ablação** |
| **M4 PL neural sequencial** | Modela dependência entre escolhas e histórico; é o análogo neural de M1 | Caro; 720 avaliações de rede por concurso; risco de overfit | **Ablação opcional** |
| **M5 LSTM 6 cabeças (v1)** | Já especificada; familiar | Não é PL; CCE é pseudo-likelihood; não modela dependência entre posições; 201.576 parâmetros para 3.054 amostras | **Ablação** |
| **Modelo multi-hot de 60 saídas** | Uma única distribuição sobre dezenas | Não garante exatamente 6 dezenas; precisaria de normalização e decodificação | Fora de escopo nesta fase |

### 3.1 Por que M1 e não M2 como primário

M1 corresponde ao mecanismo físico de uma urna com bolas de pesos diferentes, retiradas sem reposição. Essa interpretação é mais direta para uma loteria. M2 corresponde a uma inclusão condicionada ao tamanho, que é uma construção matemática válida, mas menos natural como descrição do sorteio.

Além disso, M1 tem um baseline natural da mesma família (B2b), o que permite separar "frequência empírica" de "ajuste por verossimilhança". M2 não tem um baseline igualmente natural.

O custo de M1 (720 permutações) é aceitável para 60 parâmetros e 3.074 concursos.

### 3.2 Por que não M3 como primário

M3 é a formalização correta da LSTM de 6 cabeças, mas:

- ele não modela a dependência entre as posições (as seis cabeças são independentes);
- se treinado com CCE, otimiza uma pseudo-likelihood;
- se treinado com a likelihood de conjunto, o custo e acomplexidade aumentam sem uma justificativa clara;
- ele não tem uma interpretação física direta.

M3 é uma ablação útil, não o modelo principal.

### 3.3 Por que não M4 como primário

M4 é o análogo neural mais próximo de M1 e, em princípio, o modelo mais expressivo. Mas:

- o custo de 720 avaliações de rede por concurso é alto;
- com 3.074 concursos, o risco de overfit é grande;
- a comparação com M1 precisa ser feita com a mesma likelihood de conjunto, o que exige marginalização nas duas pontas;
- o objetivo do projeto é testar sinal, não maximizar a complexidade.

M4 fica como ablação opcional, condicionada ao custo e aos resultados de M3.

---

## 4. Consequências

### 4.1 Positivas

- **Comparabilidade:** todos os modelos probabilísticos ficam normalizados sobre $\Omega$, com NLL pareada por concurso.
- **Interpretação:** M1 tem uma leitura física clara; M2 é uma checagem de forma funcional.
- **Simplicidade:** 60 parâmetros reduzem o risco de overfit e tornam o ajuste rápido.
- **Honestidade:** se M1 não superar B0, a conclusão é "sem evidência de sinal global" sem precisar de uma rede; se superar B0 mas não B2b, o sinal é capturado por frequências empíricas e o ajuste PL não agrega.
- **Ablação limpa:** a LSTM deixa de ser o modelo principal e passa a ser uma pergunta separada.

### 4.2 Negativas e custos

- **Custo de M1:** 720 permutações por concurso; a implementação precisa de logsumexp estável e de autodiferenciação.
- **Modo de M1:** não há DP exato; a decodificação pontual é heurística e deve ser rotulada como tal.
- **M3/M5 com CCE:** não são comparáveis a M1 por NLL; exigem uma tabela separada de pseudo-likelihood.
- **M4:** pode ser inviável no orçamento atual; se for, deve ser explicitamente adiado.
- **Expectativa:** com dados de loteria, o resultado mais provável é que M1 não supere B0 (nem B2b). Isso é um resultado válido, não um fracasso.

### 4.3 Impacto no `ADR-001`

A Decisão 2 do `ADR-001` (amostragem sequencial truncada) permanece válida como **modelo operacional M3-seq**, mas não é a amostragem exata de M3. Para a avaliação científica:

- usar amostragem backward pelo DP (§9.3 da `math_spec.md`);
- ou declarar explicitamente que o portfólio foi gerado por M3-seq e simular a nulidade com o mesmo esquema.

A Decisão 3 (diversidade) permanece, com a ressalva de que o filtro distorce a distribuição e deve ser desativado ou incluído na simulação nula.

---

## 5. Critérios de aceite desta decisão

- [ ] A `math_spec.md` foi aprovada.
- [ ] O `ADR-003` foi aprovado.
- [ ] O `pre_registration.md` foi congelado.
- [ ] A implementação de M1 reproduz os vetores de teste do Apêndice B da `math_spec.md`.
- [ ] A implementação de M2 reproduz os vetores de teste do Apêndice B.
- [ ] M3 normalizado por $Z$ soma 1 em um universo reduzido.
- [ ] O modo de M1 é rotulado como heurístico e reporta o limitante superior.
- [ ] Nenhum resultado é declarado antes das sanidades.

---

## 6. Riscos e mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| M1 e M2 produzirem conclusões diferentes | Média | Médio | Reportar ambos; decisão primária em M1; sensibilidade em M2 |
| O modo de M1 por beam search não ser o verdadeiro | Média | Baixo | Rotular como heurístico; reportar limitante; não usar como evidência |
| Custo de M4 inviabilizar a ablação | Alta | Médio | Adiar M4; executar M3 primeiro |
| Overfit de M3/M5 com 3.074 amostras | Média | Alto | Walk-forward, early stopping, 10 seeds, regularização |
| Interpretação indevida de "sem evidência" | Média | Alto | Relatório com poder estatístico e linguagem padronizada |

---

## 7. Decisões relacionadas

- `ADR-001` — geração e diversidade (permanece, com ressalvas).
- `ADR-003` — likelihood, normalização e comparabilidade.
- `spec/math_spec.md` — definições normativas.
- `spec/pre_registration.md` — hipóteses e critérios de veredito.

---

## 8. Aprovação

| Papel | Nome | Data | Decisão |
|---|---|---|---|
| Autor | — | 05/11/2026 | Proposto |
| Revisor técnico | — | — | Pendente |
| Aprovação final | — | — | Pendente |
