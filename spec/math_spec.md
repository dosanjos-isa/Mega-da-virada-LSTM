# Especificação Matemática — Modelos Probabilísticos para Conjuntos da Mega-Sena

**Versão:** 0.3.0-draft  
**Status:** Proposto — aguardando revisão e aprovação (Gate 0)  
**Data:** 05/11/2026  
**Idioma:** pt-BR  
**Substitui, no que conflitar:** seções 4 a 7 da `spec_lstm_mega_virada_v1.md` e a Decisão 2 do `ADR-001`.

> [!IMPORTANT]
> Este documento é **normativo** para a Fase 0. Nenhum código de modelagem, treino, avaliação ou geração deve ser implementado antes da aprovação explícita deste documento, do `ADR-002`, do `ADR-003` e do `pre_registration.md`. A aprovação é um gate humano, não automático.

> [!NOTE]
> Este documento corrige dois erros do parecer anterior: (i) o Plackett–Luce **não** pode ser aplicado diretamente às dezenas ordenadas, porque a ordem real do sorteio não é observada; (ii) o produto de distribuições marginais por posição **não** é, por si só, uma distribuição conjunta sobre conjuntos. As seções 1.2 e 5 formalizam as correções.

---

## 1. Contexto, objetivo e convenções

### 1.1 Objetivo

O projeto é um **experimento reproduzível** para testar, de forma estatisticamente honesta, se existe evidência de sinal preditivo nos sorteios da Mega-Sena/Mega da Virada. O objetivo **não** é produzir seis números com uma LSTM. A LSTM permanece no projeto como experimento de Deep Learning e ablação, não como modelo principal nem como evidência de capacidade preditiva.

Um sorteio de loteria é, por desenho, um processo aleatório. A hipótese de sinal é experimental e a conclusão "sem evidência de sinal" é um resultado válido e esperado.

### 1.2 Duas correções conceituais obrigatórias

**(a) Ordem real versus ordem crescente.** O dataset armazena as seis dezenas em ordem crescente, $d_1 < \dots < d_6$. Isso é uma **ordenação canônica do conjunto**, não a ordem em que as bolas foram fisicamente sorteadas. Um modelo que atribui probabilidade a uma sequência ordenada $\pi$ (por exemplo, Plackett–Luce) precisa ser convertido em uma distribuição sobre **conjuntos** marginalizando sobre as $6! = 720$ permutações de cada conjunto:

$$
P_{\text{set}}(S) = \sum_{\pi \in \operatorname{Perm}(S)} P_{\text{ord}}(\pi).
$$

Aplicar a fórmula sequencial diretamente a $(d_1,\dots,d_6)$ calcula a probabilidade de uma ordem específica que **não foi observada** e, portanto, não é a verossimilhança do dado.

**(b) Marginais posicionais não são uma conjunta.** Seis distribuições $P_1,\dots,P_6$ sobre posições ordenadas definem uma função de escore $\prod_j P_j(d_j)$, mas essa função **não** soma 1 sobre o espaço de conjuntos. Para se tornar uma distribuição válida, precisa ser normalizada:

$$
P_{\text{set}}(S) = \frac{\prod_{j=1}^{6} P_j(d_j)}{Z},
\qquad
Z = \sum_{d_1 < \dots < d_6} \prod_{j=1}^{6} P_j(d_j).
$$

Sem $Z$, comparações de NLL entre modelos são inválidas. A seção 5 formaliza as duas versões corretas do baseline B2.

### 1.3 Notação

| Símbolo | Significado |
|---|---|
| $N = 60$ | Universo de dezenas |
| $k = 6$ | Dezenas por concurso |
| $\Omega$ | Espaço amostral de conjuntos de 6 dezenas |
| $S \in \Omega$ | Conjunto observado, $S = \{d_1,\dots,d_6\}$ com $d_1 < \dots < d_6$ |
| $\pi$ | Sequência ordenada (permutação de $S$) |
| $w_i > 0$ | Peso da dezena $i$ |
| $W = \sum_{i=1}^{N} w_i$ | Soma dos pesos |
| $P_j(c)$ | Probabilidade de a $j$-ésima menor dezena ser $c$ |
| $e_k(w)$ | Polinômio simétrico elementar de grau $k$ |
| $Z$ | Constante de normalização sobre $\Omega$ |
| $\operatorname{supp}_j$ | Suporte admissível da posição $j$: $\{j,\dots,N-k+j\}$ |
| $\kappa$ | Parâmetro de suavização das marginais empíricas (não confundir com $\alpha$ de peso amostral) |
| $\alpha$ | Peso amostral das Viradas no treino (padrão 1,0) |
| $T$ | Temperatura de geração (padrão 1,0) |

O espaço amostral é

$$
\Omega = \{S \subset \{1,\dots,60\} : |S| = 6\},
\qquad
|\Omega| = \binom{60}{6} = 50\,063\,860.
$$

Uma **distribuição válida** sobre $\Omega$ é qualquer $p(S) \ge 0$ com $\sum_{S \in \Omega} p(S) = 1$.

---

## 2. Dados e proveniência (normativo para a Fase 1)

### 2.1 Snapshot e arquivos

| Arquivo | Papel | SHA-256 atual |
|---|---|---|
| `data/raw/megasena-download-resultados.csv` | Histórico completo da Mega-Sena | `043a16a7044d54eed9388629de3205ea2b8340c8f86b14525d3bed99ac3a6cc8` |
| `data/raw/mega_da_virada.csv` | Lista das Viradas 2009–2025 | `b0f60e6d325e591af190d8d1f60852a6cb0988931a1aab93dbf782d6822274b6` |
| `data/processed/dataset_megasena_virada_processed.csv` | Dataset unificado | `247c36d7e1ba8b84e416b2e84ff7aa25b1db1972ef416c000ef212531e762974` |

Snapshot: **3.074 concursos**, de 1 a 3074, contínuos, sem duplicatas, com **17 Viradas**. Último concurso disponível: 3074, em 2026-10-24.

> [!WARNING]
> **Problema de proveniência conhecido.** O notebook `00_format_megasena_geral.ipynb` registra um hash inicial `088db01f...` para o arquivo bruto, mas o arquivo atual tem hash `043a16a7...` e já está normalizado. O commit `8efe765` contém a versão normalizada; o arquivo original não está no Git. A Fase 1 deve recuperar o original na fonte oficial ou documentar formalmente a perda. Até que isso seja resolvido, nenhuma afirmação confirmatória sobre proveniência pode ser feita.

### 2.2 Rotulagem temporal das Viradas

O campo `ano` do dataset processado é o ano civil da data do sorteio. Isso é inadequado para walk-forward, porque a Virada de 2025 ocorreu em **2026-01-01** (concurso 2955) e foi rotulada como `ano = 2026`. Além disso, as Viradas de 2009–2014 ocorreram em **30/12**, não em 31/12.

A partir da Fase 1, o dataset deve conter uma coluna explícita **`virada_ano`** (edição da Virada, 2009–2025), independente do ano civil:

| `virada_ano` | Concurso | Data | Observação |
|---:|---:|---|---|
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

O fold de teste da Virada $Y$ é o concurso `c_Y` correspondente. O treino do fold $Y$ usa **apenas** concursos com número estritamente menor que `c_Y`. Isso é walk-forward no eixo do concurso, não no eixo do ano civil.

### 2.3 Conjuntos de avaliação pré-registrados

**Teste A (primário, $n = 300$).** Os 300 concursos regulares mais recentes do snapshot, com `is_virada = 0`:

- Início: concurso 2773 (2024-09-12).
- Fim: concurso 3074 (2026-10-24).
- Excluídos do conjunto: concursos 2810 e 2955 (Viradas).
- Treino do Teste A: concursos 1–2772.
- Validação para seleção de hiperparâmetros: 300 concursos regulares 2471–2772.
- Ajuste de referência (após seleção): concursos 1–2772.

**Teste B (secundário, $n = 11$).** As Viradas de 2015 a 2025, cada uma avaliada no fold walk-forward correspondente (treino estritamente anterior ao concurso da Virada). Viradas de 2009–2014 entram apenas como treino.

> [!NOTE]
> Teste A e Teste B **não são independentes** para as Viradas de 2024 e 2025, porque os folds dessas Viradas treinam com concursos que pertencem ao período de teste do Teste A. O Teste A é a análise confirmatória; o Teste B é secundário e descritivo, com poder muito menor.

---

## 3. Requisitos de comparabilidade

Para que duas distribuições possam ser comparadas por log-score/NLL:

- **MC-01** — Ambas devem estar definidas no mesmo $\Omega$ (conjuntos de 6 dezenas).
- **MC-02** — Ambas devem estar normalizadas: $\sum_{S \in \Omega} p(S) = 1$.
- **MC-03** — Ambas devem ser avaliadas nos mesmos concursos, com o mesmo pré-processamento e a mesma política de suavização.
- **MC-04** — A unidade de análise é o **concurso**, não a dezena. As 60 dezenas de um concurso são dependentes.
- **MC-05** — Comparações de NLL são **pareadas por concurso**: $\Delta_t = \log p_{\text{modelo}}(S_t) - \log p_{\text{baseline}}(S_t)$.
- **MC-06** — Modelos pontuais (B1, B3, decodificação top-1) **não** têm NLL. Eles são avaliados por acertos e métricas de portfólio. Se uma NLL for reportada para um modelo pontual, ela deve ser explicitamente rotulada como "ponto suavizado" e não entra no teste principal.
- **MC-07** — Pseudo-likelihoods (soma de CCEs por posição) só podem ser comparadas com outras pseudo-likelihoods da mesma família posicional (ver §6.4).

---

## 4. Modelos probabilísticos

### 4.1 M0 — Uniforme

$$
p_{\text{M0}}(S) = \frac{1}{\binom{60}{6}}.
$$

É o piso absoluto. Todo modelo deve ser comparado a M0, e nenhum resultado é interpretável sem essa referência.

### 4.2 M1 — Plackett–Luce com marginalização sobre permutações (modelo primário proposto)

Pesos $w_i > 0$, $W = \sum_i w_i$. Para uma sequência ordenada de escolhas sem reposição $\pi = (\pi_1,\dots,\pi_6)$:

$$
P_{\text{ord}}(\pi)
=
\prod_{j=1}^{6}
\frac{w_{\pi_j}}{W - \sum_{l<j} w_{\pi_l}}.
$$

Como a ordem real do sorteio não é observada, a probabilidade do conjunto $S$ é a soma sobre as $6! = 720$ permutações:

$$
\boxed{
P_{\text{M1}}(S)
=
\sum_{\pi \in \operatorname{Perm}(S)}
\prod_{j=1}^{6}
\frac{w_{\pi_j}}{W - \sum_{l<j} w_{\pi_l}}
}
$$

**Por que essa é a verossimilhança do dado.** A ordenação crescente é uma bijeção entre o conjunto $S$ e a tupla observada $(d_1,\dots,d_6)$. Portanto, a probabilidade de observar essa tupla é exatamente $P_{\text{M1}}(S)$; não é preciso modelar a ordem real do sorteio, apenas somar sobre as ordens compatíveis.

**Normalização.** Cada sequência ordenada tem probabilidade $P_{\text{ord}}(\pi)$, e o conjunto de todas as sequências ordenadas de 6 dezenas é uma partição de $\Omega$ segundo a ordem. Logo,

$$
\sum_{S \in \Omega} P_{\text{M1}}(S)
=
\sum_{\text{sequências } \pi} P_{\text{ord}}(\pi)
=
1.
$$

**Forma computacional.** O numerador é independente da ordem:

$$
P_{\text{M1}}(S)
=
\left(\prod_{i \in S} w_i\right)
\cdot
\sum_{\pi \in \operatorname{Perm}(S)}
\prod_{j=1}^{6}
\frac{1}{W - \sum_{l<j} w_{\pi_l}}.
$$

Em log-espaço:

$$
\log P_{\text{M1}}(S)
=
\sum_{i \in S} \log w_i
+
\operatorname{logsumexp}_{\pi \in \operatorname{Perm}(S)}
\left[
-\sum_{j=1}^{6}
\log\left(W - \sum_{l<j} w_{\pi_l}\right)
\right].
$$

**Custo.** $720 \times 6$ termos por concurso. Para 3.074 concursos, $\approx 2{,}2$ milhões de permutações por época ($\approx 13{,}3$ milhões de termos); viável. O modelo tem 60 pesos (59 graus de liberdade após a restrição de escala) e é estático (não usa janela temporal).

**Identificabilidade.** $w$ e $c\,w$ produzem a mesma distribuição. Fixa-se $\sum_i \log w_i = 0$ (média geométrica 1). A estimação é feita em $\theta_i = \log w_i$ com a restrição $\sum_i \theta_i = 0$.

**Caso de controle.** Se todos $w_i = 1$, então $P_{\text{ord}}(\pi) = 1/(60\cdot59\cdots55)$ e $P_{\text{M1}}(S) = 6!/(60)_6 = 1/\binom{60}{6} = P_{\text{M0}}(S)$.

**Modo.** O conjunto mais provável sob M1 é

$$
\arg\max_{S \in \Omega} P_{\text{M1}}(S),
$$

que **não** é, em geral, igual ao modo da sequência $\arg\max_\pi P_{\text{ord}}(\pi)$. Não existe DP simples para o modo de M1; ver §9.4.

### 4.3 M2 — Fisher (produto condicional, sensibilidade)

Pesos $w_i > 0$. Para $S = \{d_1,\dots,d_6\}$:

$$
P_{\text{M2}}(S)
=
\frac{\prod_{i \in S} w_i}{e_6(w)},
\qquad
e_6(w) = \sum_{|T|=6} \prod_{i \in T} w_i.
$$

**Normalização.** Por definição de $e_6$, $\sum_{S \in \Omega} \prod_{i \in S} w_i = e_6(w)$. Logo, $\sum_S P_{\text{M2}}(S) = 1$.

**Cálculo de $e_6$.** DP em $O(Nk)$:

```text
e[0] = 1
e[j] = 0 para j = 1..k
para i = 1..N:
    para j = min(i, k) até 1:
        e[j] += w[i] * e[j-1]
```

Ao final, $e_k(w) = e[k]$.

**Estabilidade numérica.** Normalize os pesos antes do DP (por exemplo, divida por $\max_i w_i$) ou use DP em log-espaço. Sem isso, $e_6$ pode estourar para pesos com grande variação.

**Identificabilidade.** Mesma invariança de escala de M1; fixa-se $\sum_i \log w_i = 0$.

**Caso de controle.** Se todos $w_i = 1$, então $e_6 = \binom{60}{6}$ e $P_{\text{M2}}(S) = 1/\binom{60}{6}$.

**Modo.** Como $e_6$ é constante, o modo de M2 é simplesmente o conjunto das 6 dezenas de maior peso:

$$
\arg\max_{S \in \Omega} P_{\text{M2}}(S)
=
\{\text{6 maiores } w_i\}.
$$

Isso é exato e barato.

**Relação com M1.** M1 e M2 são distribuições diferentes, mas ambas válidas e ambas redutíveis a M0 quando os pesos são iguais. M1 corresponde a uma urna sequencial com pesos; M2 corresponde a uma inclusão condicionada ao tamanho (produto de pesos). A escolha é uma decisão de modelagem, registrada no `ADR-002`.

### 4.4 M3 — Produto posicional normalizado (order statistics)

Distribuições $P_j$ sobre a $j$-ésima menor dezena, com suporte $\operatorname{supp}_j = \{j,\dots,60-6+j\}$ e $\sum_{c \in \operatorname{supp}_j} P_j(c) = 1$. Para $S = \{d_1<\dots<d_6\}$:

$$
q(S) = \prod_{j=1}^{6} P_j(d_j),
\qquad
Z = \sum_{d_1 < \dots < d_6} \prod_{j=1}^{6} P_j(d_j),
\qquad
P_{\text{M3}}(S) = \frac{q(S)}{Z}.
$$

**Cálculo de $Z$.** DP em $O(6 \cdot 60)$:

```text
F[1][c] = P_1(c)
para j = 2..6:
    F[j][c] = P_j(c) * soma_{c' < c} F[j-1][c']
Z = soma_c F[6][c]
```

**Normalização.** Por construção, $Z = \sum_S q(S)$ e $P_{\text{M3}}$ soma 1 sobre $\Omega$.

**Caso de controle.** Se todas as $P_j$ forem uniformes sobre seus suportes, então $q(S) = (1/55)^6$ para todo $S$ e $Z = \binom{60}{6}(1/55)^6$, logo $P_{\text{M3}} = P_{\text{M0}}$.

**Modo.** O DP atual `decode_top1_dp` resolve exatamente

$$
\arg\max_{d_1<\dots<d_6} \prod_{j=1}^{6} P_j(d_j),
$$

que é o modo de M3, pois $Z$ é constante. O DP está correto **para M3**.

**Amostragem exata.** A amostragem sequencial truncada do `ADR-001` (cabeça $j$ truncada em $[d_{j-1}+1,\,60-(6-j)]$ e renormalizada) **não** é exata para M3. Ela define outra distribuição, doravante **M3-seq**:

$$
P_{\text{M3-seq}}(S)
=
\prod_{j=1}^{6}
\frac{P_j(d_j)}{Z_j(d_{<j})},
\qquad
Z_j(d_{<j}) = \sum_{c = d_{j-1}+1}^{60-(6-j)} P_j(c),
\qquad d_0 = 0,
$$

que não é proporcional a $\prod_j P_j(d_j)$ porque os normalizadores condicionais $Z_j$ variam com o prefixo. Para amostrar exatamente de M3, usa-se amostragem backward pelo DP (ver §9.3).

**Relação com a LSTM de 6 cabeças.** A arquitetura de 6 softmax independentes, condicionadas apenas no histórico e não no prefixo de dezenas, é exatamente M3 quando normalizada por $Z$. Sem $Z$, é uma função de escore, não uma probabilidade sobre $\Omega$.

### 4.5 M4 — Plackett–Luce neural sequencial (ablação)

Uma rede (LSTM/GRU) condicionada no histórico e no prefixo de dezenas já escolhidas produz, a cada passo $j$, pesos $w_{j,i}$ para as dezenas restantes. A verossimilhança de uma sequência é

$$
P_{\text{ord}}(\pi \mid \text{histórico})
=
\prod_{j=1}^{6}
\frac{w_{j,\pi_j}}{\sum_{i \notin \{\pi_1,\dots,\pi_{j-1}\}} w_{j,i}}.
$$

A probabilidade do conjunto é novamente a soma sobre as 720 permutações:

$$
P_{\text{M4}}(S \mid \text{histórico})
=
\sum_{\pi \in \operatorname{Perm}(S)} P_{\text{ord}}(\pi \mid \text{histórico}).
$$

**Custo.** \(720 \times 6\) avaliações da rede por concurso, o que é caro. Alternativas: (i) treinar com a verossimilhança sequencial e marginalizar apenas na avaliação; (ii) usar uma aproximação de Monte Carlo ou importância; (iii) restringir a ablação a M3. A decisão de custo é registrada no `ADR-002`.

### 4.6 M5 — LSTM de 6 cabeças como M3 / pseudo-likelihood

A arquitetura de referência da v1 produz seis distribuições $P_{t,j}$ por concurso $t$, condicionadas no histórico. Ela é uma instância de M3 com $P_j = P_{t,j}$, desde que normalizada por $Z_t$. Duas formas de treino são possíveis:

- **M5-CCE (pseudo-likelihood):** minimizar $\sum_j -\log P_{t,j}(d_{t,j})$. Isso estima as marginais posicionais, não a likelihood do conjunto. A comparação justa é contra a pseudo-likelihood de B2a (soma de CCEs posicionais), não contra a NLL de M1.
- **M5-SET (likelihood de conjunto):** minimizar $-\log P_{\text{M3}}(S_t) = -\sum_j \log P_{t,j}(d_{t,j}) + \log Z_t$. Isso é comparável a M1/B2b por NLL de conjunto.

A escolha entre M5-CCE e M5-SET é registrada no `ADR-003`. A recomendação é M5-SET para a comparação principal e M5-CCE como ablação de custo.

### 4.7 Quadro-resumo dos modelos

| Modelo | Parâmetros | Usa histórico? | Likelihood de conjunto | Normalização | Papel |
|---|---|---|---|---|---|
| M0 | 0 | não | sim | trivial | piso |
| M1 | 60 (59 gl) | não | sim (720 permutações) | automática | **primário** |
| M2 | 60 (59 gl) | não | sim ($e_6$) | automática | sensibilidade |
| M3 | $6 \times 55$ (ou rede) | opcional | sim ($Z$ DP) | $Z$ | ablação |
| M4 | rede | sim | sim (720 permutações) | automática | ablação neural |
| M5 | rede | sim | sim (se M5-SET) ou pseudo | $Z$ | ablação DL |

---

## 5. Baselines

### 5.1 Definições

| ID | Baseline | Definição | Tipo | Tem NLL? |
|---|---|---|---|---|
| B0 | Uniforme | $p(S) = 1/\binom{60}{6}$ | probabilístico | sim |
| B1 | Mais frequentes | Conjunto das 6 dezenas de maior frequência no treino; desempate por menor dezena | pontual | não (sem suavização explícita) |
| B2a | Produto posicional empírico | M3 com $P_j$ estimadas por frequência posicional suavizada | probabilístico | sim |
| B2b | PL com pesos empíricos | M1 com $w_i \propto$ frequência global da dezena $i$ no treino | probabilístico | sim |
| B3 | Concurso anterior | Repetir as 6 dezenas do concurso imediatamente anterior ao alvo | pontual | não |

### 5.2 B2a — produto posicional com suavização

Seja $n_{j,c}$ o número de concursos de treino em que a $j$-ésima menor dezena é $c$, e $N_{\text{tr}}$ o número de concursos de treino. Para $c \in \operatorname{supp}_j$:

$$
\hat P_j(c)
=
\frac{n_{j,c} + \kappa}{N_{\text{tr}} + \kappa \cdot 55},
\qquad
\hat P_j(c) = 0 \text{ se } c \notin \operatorname{supp}_j.
$$

O denominador usa 55 porque $|\operatorname{supp}_j| = 55$ para todo $j$. A suavização é **obrigatória**: sem ela, uma dezena nunca observada em uma posição teria probabilidade zero e a NLL de um conjunto de teste seria infinita.

- **Suavização primária:** $\kappa = 1{,}0$ (Laplace).
- **Sensibilidade:** $\kappa \in \{0{,}1;\,0{,}5;\,2{,}0\}$.
- A escolha de $\kappa$ é feita na validação pré-teste e registrada.

### 5.3 B2b — PL com pesos empíricos

$$
w_i \propto \hat f_i = \frac{\#\{t \in \text{treino} : i \in S_t\}}{N_{\text{tr}}},
\qquad
P_{\text{B2b}} = P_{\text{M1}}(w).
$$

Como todas as 60 dezenas aparecem em algum concurso, todos os $w_i > 0$ e a likelihood é finita. B2b é o baseline **da mesma família** para M1, porque compartilha a mesma forma probabilística e isola a diferença entre "usar a frequência diretamente" e "ajustar os pesos pela verossimilhança PL". B2b **não** é o null de "não há sinal": ele já pode conter desvio de frequência. O null de sinal é B0 (uniforme).

### 5.4 B1 e B3 não têm NLL

B1 e B3 produzem **um** conjunto, não uma distribuição. Não faz sentido calcular $-\log p(S_t)$ de um ponto. Eles são avaliados por:

- acertos por concurso ($k_t$);
- média e máximo de acertos em um portfólio de $K$ jogos;
- contagem de quadras/quinas/senas, comparada com a distribuição nula simulada.

Se for indispensável reportar uma NLL para B1/B3, define-se explicitamente uma distribuição "ponto + ruído uniforme" e rotula-se como artificial; essa NLL **não** entra no teste principal.

### 5.5 Versões de portfólio ($K$ jogos)

Para comparações de portfólio, todos os modelos recebem exatamente $K$ jogos e o mesmo filtro de diversidade:

- **B0-K:** $K$ conjuntos i.i.d. uniformes.
- **B1-K:** o conjunto top-6, mais os $K-1$ conjuntos obtidos substituindo a dezena de menor frequência do top-6 pelas próximas dezenas mais frequentes, em ordem decrescente (regra determinística).
- **B2a-K:** $K$ conjuntos i.i.d. de B2a (amostragem exata, §9.3).
- **B2b-K:** $K$ conjuntos i.i.d. de B2b (amostragem PL, §9.2).
- **B3-K:** os $K$ conjuntos dos $K$ concursos anteriores distintos (o mais recente primeiro).

O filtro de diversidade é aplicado **após** a amostragem e é idêntico para todos os modelos. Para a avaliação científica, o filtro deve ser desativado ou a distribuição nula deve ser simulada com o mesmo filtro; ver §9.5.

---

## 6. Estimação e treino

### 6.1 M1 — máxima verossimilhança de conjunto

Objetivo:

$$
\ell_{\text{M1}}(\theta)
=
\sum_{t=1}^{N_{\text{tr}}}
\log P_{\text{M1}}(S_t;\, e^\theta),
\qquad
\sum_i \theta_i = 0.
$$

O gradiente é calculado por autodiferenciação (TensorFlow `GradientTape` ou equivalente) sobre a soma em log-espaço das 720 permutações. Regularização L2 opcional:

$$
\ell_{\text{M1}}^{\text{reg}} = \ell_{\text{M1}} - \lambda \sum_i \theta_i^2.
$$

- Otimizador: Adam, `learning_rate = 1e-2`, full-batch ou mini-batch 64.
- `max_epochs = 100`.
- Early stopping: paciência 10 sobre o log-score de validação (sem peso).
- Restauração dos melhores pesos.
- `λ ∈ {0; 1e-4; 1e-3}`, escolhido na validação **interna ao fold** (nunca com dados posteriores ao alvo).
- 3 reinicializações para confirmar convergência; relatar a dispersão. Não são necessárias 10 seeds para um modelo de 60 parâmetros, mas as 3 reinicializações devem ser registradas.

### 6.2 M2 — máxima verossimilhança

$$
\ell_{\text{M2}}(\theta)
=
\sum_{t=1}^{N_{\text{tr}}}
\left[
\sum_{i \in S_t} \theta_i
-
\log e_6(e^\theta)
\right],
\qquad
\sum_i \theta_i = 0.
$$

O objetivo é côncavo em $\theta$; a otimização converge para o ótimo global. Mesmo esquema de regularização e validação de M1. Como M2 é convexo, 3 reinicializações são suficientes para verificar convergência.

### 6.3 M3/M4/M5 — treino neural

- `max_epochs = 100`.
- Early stopping: paciência 10 sobre a perda de validação **sem peso**.
- `ReduceLROnPlateau`: fator 0,5; paciência 5; `min_lr = 1e-6`.
- Otimizador Adam, `learning_rate = 1e-3`.
- Lote 64.
- Validação: último 10% cronológico dos concursos regulares do treino; nunca misturar treino/validação/teste.
- A validação é sempre **interna ao fold**: nenhum hiperparâmetro é escolhido com dados posteriores ao alvo. No Teste A, a validação é o intervalo 2471–2772; em cada fold do Teste B, é o último 10% dos concursos regulares anteriores à Virada.
- 10 seeds para M3/M4/M5; média ± desvio-padrão.
- Arquitetura de referência da v1 (LSTM 128 → Dropout 0,2 → LSTM 64 → concat → Dense 128 → 6 × Dense 60 softmax) é o ponto de partida; alterações são ablações registradas.

### 6.4 Duas perdas neurais e a regra de comparação

| Perda | Fórmula | O que estima | Comparável com |
|---|---|---|---|
| CCE posicional (pseudo-likelihood) | $\sum_j -\log P_{t,j}(d_{t,j})$ | marginais posicionais | B2a-CCE, M5-CCE |
| Log-score de conjunto | $-\sum_j \log P_{t,j}(d_{t,j}) + \log Z_t$ | likelihood de conjunto M3 | M1, M2, B2a, B2b |

**Regra:** nunca comparar uma NLL de conjunto com uma CCE posicional no mesmo teste. Se o modelo for treinado com CCE, ele é avaliado com CCE e rotulado como pseudo-likelihood; se for treinado com log-score de conjunto, é avaliado com log-score de conjunto.

### 6.5 Peso amostral $\alpha$

O peso da amostra $t$ é

$$
\omega_t =
\begin{cases}
1, & \text{se o alvo é Virada},\\
\alpha, & \text{caso contrário},
\end{cases}
\qquad 0 < \alpha \le 1.
$$

A perda do lote é $\sum_t \omega_t L_t / \sum_t \omega_t$. O padrão é **$\alpha = 1{,}0$** (sem diferenciação). A grade $\alpha \in \{0{,}01; 0{,}05; 0{,}10; 0{,}25\}$ é ablação, não configuração primária.

O tamanho efetivo de amostra (Kish) é

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

`N_eff` deve ser reportado em toda execução com $\alpha \neq 1$.

### 6.6 Custo computacional

| Modelo | Custo por época (3.074 concursos) | Observação |
|---|---|---|
| M1 | $\approx 13{,}3$M termos (720 × 6 × 3.074) | viável em TF/NumPy |
| M2 | $O(60 \cdot 6)$ por avaliação | trivial |
| M3 | $O(6 \cdot 60)$ por concurso | trivial |
| M4 | $720 \times 6$ avaliações de rede por concurso | caro; ver §4.5 |
| M5 | $O(6 \cdot 60)$ por concurso | trivial |

---

## 7. Avaliação

### 7.1 Métrica primária: log-score de conjunto

$$
\text{NLL} = -\frac{1}{N_{\text{test}}} \sum_{t=1}^{N_{\text{test}}} \log p(S_t),
\qquad
\text{LS} = -\text{NLL}.
$$

A comparação é pareada:

$$
\Delta_t = \log p_{\text{modelo}}(S_t) - \log p_{\text{baseline}}(S_t).
$$

$\Delta_t > 0$ favorece o modelo. O efeito é reportado em nats por concurso e como razão de verossimilhança geométrica $\exp(\overline{\Delta})$.

### 7.2 Métricas secundárias

**Brier marginal.** Para cada dezena $i$, $p_i = P(i \in S)$ e

$$
\text{Brier}_i = \left(\mathbb{1}\{i \in S_t\} - p_i\right)^2.
$$

$p_i$ é calculado exatamente por DP para M3 e B2a, e pela fórmula $p_i = w_i\,e_5(w_{-i})/e_6(w)$ para M2. Para M1 e B2b (mesma família PL), usa-se Monte Carlo com 1e6 amostras, seed fixa e erro Monte Carlo reportado. O Brier marginal mede calibração das inclusões, não a conjunta.

**Calibração.** Diagrama de confiabilidade de $p_i$ em 10 faixas e ECE. Para $p(S)$, a calibração conjunta é exploratória e limitada por $n = 300$.

**Acertos.** $k_t = |\hat S_t \cap S_t|$. Para modelos pontuais, $\hat S_t$ é o conjunto predito; para modelos probabilísticos, usa-se o modo (top-1) e, separadamente, o portfólio amostrado. Sob $H_0$, $k_t \sim \text{Hipergeométrica}(60, 6, 6)$:

$$
\mathbb{E}[k] = 0{,}6,
\qquad
\operatorname{Var}(k) = 6 \cdot \frac{6}{60} \cdot \frac{54}{60} \cdot \frac{54}{59} \approx 0{,}494,
\qquad
\sigma_k \approx 0{,}703.
$$

**Portfólio.** Para $K$ jogos: $A_{\text{bar}} = \frac{1}{K}\sum_{m=1}^{K} k_{t,m}$, $A_{\max} = \max_m k_{t,m}$, e contagem de quadras/quinas/senas. A distribuição nula é simulada com o mesmo $K$ e o mesmo filtro de diversidade.

### 7.3 O que não pode ser usado como evidência principal

- Acurácia, precisão, revocação, F1, especificidade: são funções de $k$ e dominadas pelos verdadeiros negativos.
- $p$-valores por dezena sem correção de multiplicidade.
- NLL de modelos pontuais sem suavização explícita.
- Comparação de NLL entre modelos com normalizações ou suavizações diferentes.
- Melhor resultado entre muitas configurações sem correção de Holm.

---

## 8. Estatística e veredito

### 8.1 Hipóteses

- **H1 (primária):** M1 tem log-score de conjunto maior que **B0 (uniforme)** no Teste A.
  - $H_0$: $\mathbb{E}[\Delta_t] \le 0$; $H_1$: $\mathbb{E}[\Delta_t] > 0$ (unilateral).
  - $\alpha_{\text{teste}} = 0{,}05$.
- **H2 (secundária):** M1 tem log-score de conjunto maior que B2b no Teste A (valor do ajuste PL sobre frequências empíricas).
- **H3 (secundária):** M3 tem log-score de conjunto maior que B2a no Teste A (sinal temporal/neural).
- **H4 (secundária):** M1 tem média de acertos maior que B0 no Teste B (11 Viradas).
- **H5 (secundária):** M1 tem média de acertos maior que B1 no Teste B.
- **H6 (secundária):** o portfólio de M1 tem $A_{\text{bar}}$ maior que B0-K no Teste B.
- **H7 (exploratória):** M4 tem log-score de conjunto maior que M1 no Teste A.

H2–H6 formam a família secundária, corrigida por Holm. H1 é a única hipótese confirmatória e não entra na correção. H7 é exploratória.

> [!NOTE]
> H1 testa **estrutura global de frequência** (viés de bolas) contra o uniforme, não dependência temporal: M1 é estático e não usa a ordem dos concursos. H2 testa se o ajuste PL agrega sobre frequências empíricas. A hipótese de sinal **temporal/neural** é H3 (M3 vs B2a), secundária. Uma conclusão de "evidência" deve dizer qual dos tipos de sinal foi encontrado; não se pode usar H1 para afirmar que a LSTM prevê o futuro.

### 8.2 Testes

- **Wilcoxon signed-rank unilateral** sobre $\Delta_t$ para H1–H3.
- **Teste de sinal ou permutação** como sensibilidade se houver muitos empates ou assimetria.
- **Bootstrap BCa** com 10.000 reamostragens **por concurso** para o IC 95% de $\overline{\Delta}$.
- **Teste exato hipergeométrico** para H4; **permutação de sinais** para H5.
- **Simulação nula** com o mesmo $K$ e filtro para H6.
- Tamanho de efeito sempre reportado; nunca apenas $p$-valor.

### 8.3 Poder estatístico

Para um teste unilateral com $\alpha = 0{,}05$ e poder de 80%, o efeito mínimo detectável (MDE) para uma média pareada é

$$
\text{MDE} = \frac{(z_{0{,}95} + z_{0{,}80})\,\sigma_d}{\sqrt{n}}
= \frac{2{,}486\,\sigma_d}{\sqrt{n}}.
$$

| Amostra | $n$ | MDE em unidades de $\sigma_d$ |
|---|---:|---:|
| Teste A | 300 | $0{,}144\,\sigma_d$ |
| Teste B (11 Viradas) | 11 | $0{,}750\,\sigma_d$ |

Para acertos no Teste B, $\sigma_k \approx 0{,}703$, logo o MDE é $\approx 0{,}53$ acertos (média sob $H_1$ $\approx 1{,}13$). A formulação bilateral da v1 produzia $\approx 0{,}59$; este documento adota a unilateral por coerência com H1.

> [!WARNING]
> Com 11 Viradas, o Teste B só detecta efeitos enormes. Resultado não significativo **não prova** ausência de sinal. O Teste A, com 300 concursos, é a análise com poder razoável para o log-score.

### 8.4 Sanidade

Antes de qualquer conclusão, devem passar:

1. **Rótulos permutados:** treinar com os conjuntos-alvo embaralhados entre concursos; o modelo não pode vencer B2b/B2a no Teste A.
2. **Entradas aleatórias:** substituir as janelas de histórico por multi-hot aleatórios; o modelo não pode vencer os baselines.
3. **Sentinela de vazamento:** inserir um concurso futuro fictício e verificar que ele é rejeitado por todos os splitters.
4. **Paridade de $K$:** todo portfólio comparado deve ter o mesmo $K$ e o mesmo filtro.

Se qualquer sanidade "vencer" significativamente, o veredito é **erro/vazamento**, não evidência.

### 8.5 Veredito padronizado

O veredito é composto por **tipo de sinal** e **força da evidência**; não existe um "sinal" genérico.

- **Sinal global de frequência:** H1 rejeitada em $\alpha = 0{,}05$ unilateral, com $\overline{\Delta} \ge 0{,}05$ nats/concurso e IC 95% excluindo zero; sanidades aprovadas; sinal consistente em pelo menos 8 dos 11 folds de Virada (critério descritivo, não teste).
- **Valor do ajuste PL:** H2 rejeitada após Holm, na direção pré-registrada.
- **Sinal temporal/neural:** H3 rejeitada após Holm, na direção pré-registrada.
- **Sinal de acerto/portfólio:** H4, H5 ou H6 rejeitada após Holm (evidência secundária, não substitui o log-score).
- **Inconclusivo:** a hipótese relevante não rejeitada, mas o IC 95% inclui efeitos $\ge 0{,}05$ nats/concurso; ou rejeitada mas a robustez falha.
- **Sem evidência de sinal:** a hipótese relevante não rejeitada e o IC 95% exclui efeitos $\ge 0{,}05$ nats/concurso, com sanidades aprovadas.
- **Erro/vazamento:** qualquer sanidade falha ou o modelo vence de forma implausível.

O limiar de 0,05 nats/concurso é uma escolha pré-registrada de relevância prática (razão de verossimilhança $\approx 1{,}05$ por concurso; $\approx 3{,}3 \times 10^6$ em 300 concursos). Ele não deve ser alterado após ver os resultados.

---

## 9. Geração

### 9.1 Requisitos gerais

- **MG-01** — Todo jogo gerado deve ter 6 dezenas distintas, em $[1,60]$, ordenadas crescentemente.
- **MG-02** — A amostragem científica é i.i.d. da distribuição do modelo, com $T = 1{,}0$ e sem filtro de diversidade. O filtro de diversidade é operacional e deve ser declarado separadamente.
- **MG-03** — Toda geração grava metadados imutáveis (§10.3).

### 9.2 M0, M1 e B2b

- **M0:** amostrar 6 dezenas uniformes sem reposição e ordenar.
- **M1/B2b:** amostrar uma sequência PL sucessiva (peso $w_i$ entre as restantes), depois ordenar. Isso é exato para a distribuição de conjuntos de M1, porque cada sequência ordenada tem a probabilidade $P_{\text{ord}}$ e a ordenação é uma função determinística.
- **Temperatura:** $w_i^{(1/T)}$, renormalizados. $T=1$ na avaliação científica.

### 9.3 M2, M3 e M5

**M2 (Fisher).** Amostragem recursiva exata. Para um conjunto de tamanho $k$ restante e pesos $w$:

1. Calcular $e_{k-1}(w_{-i})$ para cada $i$ por DP.
2. Amostrar o primeiro elemento $i$ com probabilidade
   $$
   \pi_i = \frac{w_i\, e_{k-1}(w_{-i})}{k\, e_k(w)}.
   $$
3. Remover $i$, decrementar $k$ e repetir.

Esse esquema produz cada conjunto $S$ com probabilidade $\prod_{i\in S} w_i / e_k(w)$ (cada conjunto tem $k$ ordens igualmente prováveis de "primeiro elemento").

**M3/M5 (produto posicional).** Amostragem backward exata pelo DP:

1. Calcular $F_1(c) = P_1(c)$ e $F_j(c) = P_j(c)\sum_{c'<c} F_{j-1}(c')$ para $j=2,\dots,6$.
2. $Z = \sum_c F_6(c)$.
3. Amostrar $d_6$ com probabilidade $F_6(c)/Z$.
4. Para $j=5,\dots,1$, amostrar $d_j$ com probabilidade proporcional a $F_j(c)$ para $c < d_{j+1}$.

Isso é exato para $P_{\text{M3}}(S) = \prod_j P_j(d_j)/Z$. A amostragem sequencial truncada do `ADR-001` **não** é exata e não deve ser usada como se fosse.

### 9.4 M1 e M4 — modo e decodificação pontual

- **M2:** modo exato = 6 maiores pesos.
- **M3/M5:** modo exato = DP atual.
- **M1:** o modo do conjunto não é, em geral, o modo da sequência. Não há DP simples. Usar **beam search** sobre sequências ordenadas, coletar os melhores conjuntos candidatos e avaliar $P_{\text{M1}}(S)$ exatamente para cada candidato. Rotular como **heurístico** e reportar o limitante superior $6! \cdot \max_\pi P_{\text{ord}}(\pi)$.
- **M4:** beam search análogo, condicionado no prefixo.
- **Regra:** nenhum ponto preditivo é usado como evidência de sinal; ele é uma saída operacional.

### 9.5 Diversidade

O filtro `max_shared_numbers` é aplicado após a amostragem:

$$
|J_{\text{cand}} \cap J_m| \le \text{max\_shared\_numbers},
\qquad \forall m < k.
$$

Ele **altera a distribuição** do portfólio. Para avaliação científica:

- desativar o filtro; ou
- simular a distribuição nula com o mesmo filtro e o mesmo $K$.

Nunca comparar um portfólio filtrado com um baseline não filtrado. Se o filtro saturar, o comportamento deve ser explícito: retornar parcial com `saturated = True` **ou** lançar `GenerationSaturationError` (decisão no `ADR-001`, emenda).

---

## 10. Reprodutibilidade

### 10.1 Manifestos

Cada execução grava um manifesto JSON com:

- hash SHA-256 do raw e do processado;
- hash do config YAML;
- hash do modelo salvo;
- commit Git (ou `dirty` + diff hash);
- versões de Python, NumPy, Pandas, SciPy, TensorFlow/Keras;
- seeds;
- data/hora UTC;
- `virada_ano` e folds usados.

### 10.2 Determinismo

- Fixar `PYTHONHASHSEED`, `numpy.random.seed`, `tf.random.set_seed`.
- `shuffle = "batches_only"`; nunca misturar treino/validação/teste.
- `tf.config.experimental.enable_op_determinism()` quando disponível.
- Tolerância documentada para diferenças numéricas entre execuções.

### 10.3 Artefatos de geração

Cada geração grava `artifacts/generations/<timestamp>_<hash>.json` com: modelo (hash), dataset (hash), config (hash), commit, seed, $T$, modelo probabilístico, $K$, diversidade, tentativas, jogos e flag de saturação. Nada é sobrescrito.

---

## 11. Critérios de aceite da Fase 0

- [ ] Fórmulas de M1, M2, M3, M4 e M5 revisadas e sem ambiguidade de normalização.
- [ ] B2a e B2b definidos, com suavização e normalização explícitas.
- [ ] Regra de comparabilidade (MC-01 a MC-07) aceita.
- [ ] `ADR-002` (modelo primário) aprovado.
- [ ] `ADR-003` (likelihood/comparabilidade) aprovado.
- [ ] `pre_registration.md` congelado, com hipóteses, métricas, testes, seeds, poder e critérios de veredito.
- [ ] Nenhum código de modelagem implementado antes deste gate.
- [ ] Vetores de teste do Apêndice B conferidos por revisão independente.

---

## 12. Decisões abertas

| ID | Decisão | Recomendação | Onde registrar |
|---|---|---|---|
| D-01 | M1 ou M2 como primário | **M1** (interpretação física de urna sequencial); M2 como sensibilidade | `ADR-002` |
| D-02 | M3 ou M4 como ablação neural | **M3** primeiro (custo baixo); M4 só se M3 mostrar algo | `ADR-002` |
| D-03 | M5-CCE ou M5-SET | **M5-SET** para a comparação principal; M5-CCE como ablação de custo | `ADR-003` |
| D-04 | Saturação do filtro de diversidade | Retornar parcial com flag `saturated=True` (não lançar exceção) | `ADR-001` (emenda) |
| D-05 | Recuperação do raw original | Tentar a fonte oficial; se falhar, documentar a perda e marcar o dataset como "proveniência parcial" | Fase 1 |

---

## Apêndice A — Derivações

### A.1 Normalização de M1

O conjunto de todas as sequências ordenadas de 6 dezenas distintas é

$$
\Pi_6 = \{(\pi_1,\dots,\pi_6) : \pi_j \in \{1,\dots,60\},\ \pi_j \neq \pi_l \text{ para } j \neq l\}.
$$

A construção PL define uma probabilidade sobre $\Pi_6$:

$$
\sum_{\pi \in \Pi_6} P_{\text{ord}}(\pi)
=
\sum_{\pi_1} \frac{w_{\pi_1}}{W}
\sum_{\pi_2 \neq \pi_1} \frac{w_{\pi_2}}{W - w_{\pi_1}}
\cdots
\sum_{\pi_6 \notin \{\pi_1,\dots,\pi_5\}} \frac{w_{\pi_6}}{W - \sum_{l<6} w_{\pi_l}}
=
1.
$$

Cada $S \in \Omega$ corresponde a exatamente $6!$ sequências (suas permutações), e a união dessas sequências sobre todos os $S$ é $\Pi_6$. Logo, $\sum_S P_{\text{M1}}(S) = 1$.

### A.2 DP de $e_k$

A recorrência

$$
e_j^{(i)} = e_j^{(i-1)} + w_i\, e_{j-1}^{(i-1)}
$$

com $e_0^{(i)} = 1$ e $e_j^{(0)} = 0$ para $j \ge 1$ calcula o polinômio simétrico elementar de grau $j$ sobre as primeiras $i$ variáveis. Ao final, $e_k^{(N)} = e_k(w)$.

### A.3 DP de $Z$ em M3

Defina

$$
F_j(c) = \sum_{\substack{d_1 < \dots < d_{j-1} < c\\ d_j = c}} \prod_{l=1}^{j} P_l(d_l).
$$

Então $F_1(c) = P_1(c)$ e

$$
F_j(c) = P_j(c) \sum_{c' < c} F_{j-1}(c').
$$

Logo, $Z = \sum_c F_6(c)$.

### A.4 Amostragem backward de M3

A marginal de $d_6$ é

$$
p(d_6 = c) = \frac{F_6(c)}{Z}.
$$

Dado $d_{j+1}$, a condicional de $d_j$ é proporcional a $F_j(c)$ para $c < d_{j+1}$, pois o restante da cadeia (posições $j+1,\dots,6$) é constante nessa condicional. Isso permite amostrar de trás para frente, exatamente.

### A.5 Amostragem de M2

A probabilidade de inclusão de $i$ é

$$
P(i \in S) = \frac{w_i\, e_{k-1}(w_{-i})}{e_k(w)}.
$$

Para amostrar sem reposição, escolhe-se o primeiro elemento com probabilidade

$$
\pi_i = \frac{w_i\, e_{k-1}(w_{-i})}{k\, e_k(w)},
$$

remove-se $i$, decrementa-se $k$ e repete-se. A probabilidade de uma ordem específica $(i_1,\dots,i_k)$ é

$$
\prod_{j=1}^{k} \frac{w_{i_j}\, e_{k-j}(w_{-\{i_1,\dots,i_j\}})}{(k-j+1)\, e_{k-j+1}(w_{-\{i_1,\dots,i_{j-1}\}})}
=
\frac{\prod_{j=1}^{k} w_{i_j}}{k!\, e_k(w)}.
$$

Somando sobre as $k!$ ordens de $S$, obtém-se $\prod_{i\in S} w_i / e_k(w)$. Logo, a amostragem é exata.

---

## Apêndice B — Vetores de teste (universo reduzido)

Para validar a implementação, use $N = 5$, $k = 2$, $w = (1,2,3,4,5)$, $W = 15$.

### B.1 M2 — Fisher

$e_2(w) = 85$. As 10 probabilidades são:

| $S$ | $P_{\text{M2}}(S)$ |
|---|---:|
| {1,2} | 2/85 ≈ 0,023529 |
| {1,3} | 3/85 ≈ 0,035294 |
| {1,4} | 4/85 ≈ 0,047059 |
| {1,5} | 5/85 ≈ 0,058824 |
| {2,3} | 6/85 ≈ 0,070588 |
| {2,4} | 8/85 ≈ 0,094118 |
| {2,5} | 10/85 ≈ 0,117647 |
| {3,4} | 12/85 ≈ 0,141176 |
| {3,5} | 15/85 ≈ 0,176471 |
| {4,5} | 20/85 ≈ 0,235294 |

Soma = 1. O modo é {4,5}.

### B.2 M1 — Plackett–Luce marginalizado

As 10 probabilidades de conjunto são:

| $S$ | $P_{\text{M1}}(S)$ |
|---|---:|
| {1,2} | 9/455 ≈ 0,019780 |
| {1,3} | 13/420 ≈ 0,030952 |
| {1,4} | 10/231 ≈ 0,043290 |
| {1,5} | 2/35 ≈ 0,057143 |
| {2,3} | 5/78 ≈ 0,064103 |
| {2,4} | 64/715 ≈ 0,089510 |
| {2,5} | 23/195 ≈ 0,117949 |
| {3,4} | 23/165 ≈ 0,139394 |
| {3,5} | 11/60 ≈ 0,183333 |
| {4,5} | 14/55 ≈ 0,254545 |

Soma = 1. O modo é {4,5}. Note que o modo coincide com o de M2 neste exemplo, mas as probabilidades são diferentes; isso é suficiente para detectar uma implementação que confunde M1 com M2.

### B.3 Caso uniforme

Com $w = (1,1,1,1,1)$, $N=5$, $k=2$:

- $P_{\text{M0}} = P_{\text{M1}} = P_{\text{M2}} = 1/10$ para todo $S$.
- Se $P_1$ e $P_2$ forem uniformes em seus suportes (4 valores cada), $Z = 10/16$ e $P_{\text{M3}}(S) = 1/10$.

### B.4 Teste de normalização

Para qualquer implementação:

```text
# universo reduzido: N=5, k=2
soma_M1 = sum(P_M1(S) for S in combinations(range(1, 6), 2))
soma_M2 = sum(P_M2(S) for S in combinations(range(1, 6), 2))
soma_M3 = sum(P_M3(S) for S in combinations(range(1, 6), 2))
```

No universo reduzido, as três somas devem ser 1 dentro de $10^{-10}$. No universo completo ($N=60$), a soma também é 1 por construção, mas a verificação por força bruta é inviável (50 milhões de conjuntos); ela é feita por DP e por testes de consistência (por exemplo, M1, M2 e M3 com pesos uniformes devem coincidir com M0 em amostras exatas).

---

## Apêndice C — Glossário de decisões

| Termo | Significado neste documento |
|---|---|
| **Conjunto** | 6 dezenas distintas, em ordem crescente (representação canônica) |
| **Sequência** | ordem de escolha das 6 dezenas; não observada no dataset |
| **Likelihood de conjunto** | $P(S)$, normalizada sobre $\Omega$ |
| **Pseudo-likelihood** | soma de CCEs posicionais; não é $P(S)$ |
| **Normalização** | garantir $\sum_S p(S) = 1$ |
| **Modo** | conjunto $S$ de maior $p(S)$ |
| **Walk-forward** | treinar apenas com concursos anteriores ao alvo |
| **Teste A** | 300 concursos regulares 2773–3074 (excluindo Viradas) |
| **Teste B** | 11 Viradas 2015–2025, uma por fold |
