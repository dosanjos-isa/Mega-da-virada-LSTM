# ADR-003: Likelihood, Normalização e Comparabilidade entre Modelos

**Status:** Proposto — aguardando aprovação (Gate 0)  
**Data:** 05/11/2026  
**Contexto:** `spec/math_spec.md` v0.3.0-draft; `ADR-002`  
**Relaciona-se:** `ADR-001` (geração), `spec_lstm_mega_virada_v1.md` (métricas M1–M10)

---

## 1. Contexto

### 1.1 O problema

O parecer anterior recomendava comparar a LSTM com o baseline B2 por NLL. A revisão técnica mostrou que essa comparação, como estava escrita, era inválida por três motivos:

1. **B2 era definido como produto de marginais posicionais**, que não soma 1 sobre o espaço de conjuntos. Sem normalização, a NLL não é uma log-verossimilhança.
2. **A LSTM de 6 cabeças treinada com CCE** minimiza uma soma de entropias cruzadas posicionais, não a NLL do conjunto. Comparar essa pseudo-likelihood com a NLL de um modelo de conjunto seria comparar objetivos diferentes.
3. **B1 e B3 são modelos pontuais**, que produzem um conjunto, não uma distribuição. Não têm NLL.

### 1.2 O que este ADR fixa

Este ADR fixa as regras para que qualquer comparação de NLL/log-score seja válida, e separa explicitamente:

- modelos probabilísticos de conjunto;
- pseudo-likelihoods posicionais;
- modelos pontuais.

---

## 2. Decisões

### D1 — Todo modelo probabilístico é normalizado sobre $\Omega$

Todo modelo que se apresenta como distribuição sobre conjuntos deve satisfazer

$$
p(S) \ge 0,
\qquad
\sum_{S \in \Omega} p(S) = 1,
\qquad
\Omega = \{S \subset \{1,\dots,60\} : |S| = 6\}.
$$

- M0: trivial.
- M1: normalização automática pela construção PL (§A.1 da `math_spec.md`).
- M2: normalização por $e_6(w)$ (§A.2).
- M3/M5: normalização por $Z$ (§A.3).
- M4: normalização automática pela construção sequencial, com marginalização sobre permutações.
- B2a: normalização por $Z$.
- B2b: normalização automática (é M1 com pesos empíricos).

**Critério de verificação:** em um universo reduzido ($N=5$, $k=2$), a soma das probabilidades de todos os $\binom{5}{2}=10$ conjuntos deve ser 1 dentro de $10^{-10}$. No universo completo, a soma não é enumerável por força bruta; a verificação é feita por DP e por testes de consistência.

### D2 — A métrica primária é o log-score de conjunto

$$
\text{LS} = \frac{1}{N_{\text{test}}} \sum_{t=1}^{N_{\text{test}}} \log p(S_t),
\qquad
\text{NLL} = -\text{LS}.
$$

A comparação é pareada por concurso:

$$
\Delta_t = \log p_{\text{modelo}}(S_t) - \log p_{\text{baseline}}(S_t).
$$

Reportar:

- $\overline{\Delta}$ em nats por concurso;
- $\exp(\overline{\Delta})$ (razão de verossimilhança geométrica);
- IC 95% por bootstrap BCa com reamostragem por concurso;
- $p$-valor do teste pré-registrado.

### D3 — Modelos pontuais não têm NLL

B1 (mais frequentes), B3 (concurso anterior) e qualquer decodificação top-1 produzem **um** conjunto. Eles são avaliados por:

- acertos $k_t = |\hat S_t \cap S_t|$;
- média e máximo de acertos em portfólio de $K$ jogos;
- contagem de quadras/quinas/senas contra a distribuição nula simulada.

Se uma NLL for reportada para um modelo pontual, ela deve ser explicitamente definida como "ponto + ruído uniforme" e **não** entra no teste principal.

### D4 — Pseudo-likelihood é uma família separada

Uma pseudo-likelihood posicional é

$$
\text{PL-CCE} = -\frac{1}{N_{\text{test}}} \sum_{t=1}^{N_{\text{test}}} \sum_{j=1}^{6} \log P_{t,j}(d_{t,j}).
$$

Ela só pode ser comparada com outra pseudo-likelihood da mesma família posicional:

- M5-CCE vs B2a-CCE;
- M3-CCE vs B2a-CCE.

Nunca comparar PL-CCE com NLL de conjunto (M1, M2, B2b) no mesmo teste.

### D5 — B2 tem duas versões, ambas válidas

| Baseline | Definição | Família | Normalização |
|---|---|---|---|
| **B2a** | M3 com $P_j$ empíricas suavizadas | posicional | $Z$ |
| **B2b** | M1 com $w_i \propto$ frequência global | PL | automática |

- **B2a** é o baseline da mesma família para M3/M5.
- **B2b** é o baseline da mesma família para M1/M2.
- **B0 (uniforme)** é o piso para o teste de **sinal** (H1). B2b/B2a são baselines de **valor do ajuste** (H2/S1): eles já podem conter o sinal, portanto não servem como null de "não há sinal".
- Não existe "B2" genérico. Toda tabela deve dizer B2a ou B2b.

### D6 — Suavização é obrigatória em B2a

$$
\hat P_j(c) = \frac{n_{j,c} + \kappa}{N_{\text{tr}} + 55\kappa},
\qquad c \in \operatorname{supp}_j = \{j,\dots,60-6+j\}.
$$

- $\kappa = 1{,}0$ (Laplace) é o valor primário.
- $\kappa \in \{0{,}1;\,0{,}5;\,2{,}0\}$ é sensibilidade.
- Sem suavização, uma dezena nunca observada em uma posição produz NLL infinita. Isso não é um resultado empírico; é um artefato de suporte.

### D7 — A unidade de análise é o concurso

- O bootstrap reamostra **concursos**, não dezenas.
- As 60 dezenas de um concurso são dependentes.
- Os testes pareados usam um $\Delta_t$ por concurso.
- Métricas por dezena são descritivas e exigem correção de multiplicidade; não entram no veredito.

### D8 — A comparação é sempre contra o mesmo $K$ e o mesmo filtro

Para portfólios:

- todos os modelos recebem exatamente $K$ jogos;
- o filtro de diversidade é idêntico;
- a distribuição nula é simulada com o mesmo $K$ e o mesmo filtro;
- nunca comparar $K=20$ do modelo com $K=1$ do baseline.

### D9 — M5 tem duas variantes, com comparações distintas

| Variante | Treino | Comparável com | Papel |
|---|---|---|---|
| M5-SET | $-\sum_j \log P_{t,j}(d_{t,j}) + \log Z_t$ | M1, M2, B2a, B2b (NLL de conjunto) | ablação principal |
| M5-CCE | $\sum_j -\log P_{t,j}(d_{t,j})$ | B2a-CCE (pseudo-likelihood) | ablação de custo |

A variante primária é **M5-SET**. M5-CCE só é reportada como pseudo-likelihood, em tabela separada.

### D10 — Reportar em nats e em razão de verossimilhança

Toda diferença de log-score deve ser reportada em nats por concurso e como $\exp(\Delta)$. Exemplo: $\overline{\Delta} = 0{,}05$ nats/concurso corresponde a uma razão de verossimilhança geométrica de $e^{0{,}05} \approx 1{,}051$ por concurso, ou $\approx 3{,}3 \times 10^6$ em 300 concursos. Isso evita que diferenças pequenas em nats sejam interpretadas como irrelevantes ou como enormes sem contexto.

### D11 — O veredito não mistura métricas

O veredito pré-registrado é baseado em **uma** métrica primária (H1: log-score de M1 vs B0 no Teste A). H2 (M1 vs B2b) e H3 (M3 vs B2a) são secundárias e entram na família Holm. Métricas de acerto, Brier, calibração e portfólio são secundárias ou exploratórias. Nenhuma combinação post hoc de métricas pode produzir "evidência".

---

## 3. Matriz de comparabilidade

| Modelo | Normalizado? | Likelihood de conjunto? | Pode entrar em teste de log-score? | Métricas permitidas |
|---|---|---|---|---|
| M0 | sim | sim | como piso | LS, NLL, Brier, calibração |
| M1 | sim | sim | **sim** | LS, NLL, Brier, calibração, acertos, portfólio |
| M2 | sim | sim | sim (sensibilidade) | LS, NLL, Brier, calibração, acertos, portfólio |
| M3 | sim (com $Z$) | sim | sim (H3/S1) | LS, NLL, Brier, calibração, acertos, portfólio |
| M4 | sim | sim | exploratório | LS, NLL, Brier, calibração, acertos, portfólio |
| M5-SET | sim (com $Z$) | sim | sim (ablação) | LS, NLL, Brier, calibração, acertos, portfólio |
| M5-CCE | não (pseudo) | não | não | PL-CCE, acertos, portfólio |
| B0 | sim | sim | baseline | LS, NLL, Brier, calibração, acertos, portfólio |
| B1 | não (pontual) | não | baseline de acertos | acertos, portfólio |
| B2a | sim (com $Z$) | sim | baseline | LS, NLL, Brier, calibração, acertos, portfólio |
| B2b | sim | sim | baseline | LS, NLL, Brier, calibração, acertos, portfólio |
| B3 | não (pontual) | não | baseline de acertos | acertos, portfólio |

---

## 4. Consequências

### 4.1 Positivas

- **Validade:** toda NLL reportada é, de fato, uma log-verossimilhança sobre o mesmo espaço amostral.
- **Clareza:** a tabela separa modelos probabilísticos, pseudo-likelihoods e modelos pontuais.
- **Reprodutibilidade:** as regras de normalização e suavização são explícitas e testáveis.
- **Honestidade:** a LSTM não é comparada de forma injusta nem favorecida por uma métrica inadequada.

### 4.2 Negativas e custos

- **Custo de implementação:** M1 exige soma sobre 720 permutações; M3/B2a exigem $Z$ por DP; M5-SET exige $Z$ diferenciável.
- **Tabelas separadas:** o relatório terá uma tabela de NLL de conjunto e outra de pseudo-likelihood; não haverá um "ranking único" que misture as duas.
- **B1/B3 sem NLL:** a comparação com esses baselines é apenas por acertos/portfólio, o que é menos informativo para o log-score.
- **Sensibilidade à suavização:** B2a depende de $\kappa$; a escolha primária e a sensibilidade precisam ser reportadas.

### 4.3 Riscos

| Risco | Mitigação |
|---|---|
| Alguém comparar M5-CCE com M1 por NLL | A tabela de comparabilidade e os testes de aceite bloqueiam; o relatório separa as famílias |
| B2a com $\kappa$ escolhido após ver o teste | $\kappa$ é pré-registrado; a sensibilidade é reportada |
| NLL infinita por suporte zero | Suavização obrigatória; teste de suporte |
| Portfólio filtrado comparado com baseline não filtrado | D8 e simulação nula com o mesmo filtro |
| "Sem evidência" interpretado como "prova de ausência" | Relatório com poder estatístico e linguagem padronizada |

---

## 5. Critérios de aceite

- [ ] Todo modelo probabilístico soma 1 em universo reduzido.
- [ ] B2a e B2b implementados separadamente, com normalização verificada.
- [ ] M5-SET e M5-CCE separados no código e no relatório.
- [ ] Nenhuma NLL de modelo pontual no teste principal.
- [ ] Comparações pareadas por concurso, com bootstrap por concurso.
- [ ] Tabela de comparabilidade impressa no relatório.
- [ ] Testes de suporte zero para B2a.
- [ ] Teste de que M1 com $w$ uniforme é idêntico a M0.
- [ ] Teste de que M2 com $w$ uniforme é idêntico a M0.
- [ ] Teste de que M3 com $P_j$ uniformes é idêntico a M0.

---

## 6. Relação com a spec v1

Este ADR **substitui** as seguintes partes da `spec_lstm_mega_virada_v1.md`:

- §6.1, B2: passa a ser B2a/B2b.
- §6.2, M2: a NLL por cabeça deixa de ser a métrica primária; a métrica primária é o log-score de conjunto.
- §6.4, M5–M10: permanecem como métricas secundárias/exploratórias, mas não entram no veredito.
- §6.2, M1: permanece como métrica de acertos, mas deixa de ser a única métrica primária.

Este ADR **não substitui**:

- `RV-001` a `RV-014` (walk-forward, sanidade, poder, veredito), que permanecem válidos;
- `ADR-001` (geração e diversidade), que permanece válido com as ressalvas da §4.3 do `ADR-002`.

---

## 7. Aprovação

| Papel | Nome | Data | Decisão |
|---|---|---|---|
| Autor | — | 05/11/2026 | Proposto |
| Revisor técnico | — | — | Pendente |
| Aprovação final | — | — | Pendente |
