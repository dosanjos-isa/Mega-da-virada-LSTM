# Pré-Registro Estatístico — Mega da Virada / Mega-Sena

**Versão:** 0.3.0-draft  
**Status:** Proposto — a ser congelado no Gate 0  
**Data:** 05/11/2026  
**Snapshot de dados:** 2026-10-24 (concurso 3074)  
**Documentos normativos:** `spec/math_spec.md`, `spec/adr/ADR-002-modelo-probabilistico-primario.md`, `spec/adr/ADR-003-likelihood-e-comparabilidade.md`

> [!IMPORTANT]
> Este documento congela, **antes** da execução dos modelos principais, as hipóteses, os conjuntos de teste, as métricas, os testes estatísticos, as seeds e os critérios de veredito. Qualquer alteração posterior deve ser registrada como emenda datada, com justificativa, e não pode transformar uma análise exploratória em confirmatória.

---

## 1. Declaração de exposição prévia aos dados

Antes deste pré-registro, foram realizadas análises exploratórias. Elas são declaradas aqui para transparência e **não** contam como evidência confirmatória:

| Análise | Resultado | Status |
|---|---|---|
| Qui-quadrado de uniformidade das 60 dezenas, história completa (3.074 concursos) | $\chi^2 = 81{,}39$, $gl = 59$, $p = 0{,}0283$ (não corrigido) | Exploratória |
| Qui-quadrado de uniformidade, últimos 300 concursos | $\chi^2 = 41{,}40$, $p = 0{,}9603$ | Exploratória |
| Média de acertos in-sample do baseline de frequência (B2b) | 0,612 | Exploratória |
| Inspeção de datas e Viradas | 2009–2014 em 30/12; 2025 em 01/01/2026 | Operacional |
| Contagem de parâmetros da LSTM de referência | 201.576 | Operacional |

O $p$-valor de 0,0283 na história completa é marginal e não corrigido por multiplicidade. O resultado dos últimos 300 concursos é compatível com uniformidade. Nenhum dos dois foi usado para escolher o Teste A, que é definido por recência (os 300 concursos regulares mais recentes do snapshot), não por resultado.

A escolha de M1 como modelo primário foi feita com base em argumentos matemáticos e de interpretação física, antes de qualquer ajuste de M1 ao Teste A.

---

## 2. Pergunta de pesquisa

**Pergunta principal.** Um modelo probabilístico de conjunto, ajustado apenas com dados passados, produz log-score fora da amostra melhor do que o **acaso uniforme (B0)**?

**Perguntas secundárias.** (a) O ajuste PL agrega sobre frequências empíricas (M1 vs B2b)? (b) Modelos neurais (M3/M5) ou um baseline pontual (B0/B1) melhoram alguma métrica pré-registrada?

**Distinção obrigatória.** A pergunta principal (H1) testa **estrutura global de frequência**: alguma dezena é mais provável que outra, contra o uniforme B0? Ela **não** testa dependência temporal, porque M1 é estático. A pergunta "o ajuste PL agrega sobre frequências empíricas?" é S2 (M1 vs B2b). A pergunta sobre sinal **temporal/neural** é S1 (M3 vs B2a). O relatório deve separar explicitamente "desvio global de frequência", "valor do ajuste PL" e "sinal temporal".

**Não é pergunta deste projeto.** "Quais seis números apostar?" ou "é possível garantir retorno?" Não há promessa de capacidade preditiva.

---

## 3. Hipóteses

### 3.1 Hipótese primária (confirmatória)

**H1.** No Teste A ($n = 300$ concursos regulares), o log-score de conjunto de M1 é maior que o de **B0 (uniforme)**.

$$
H_0: \mathbb{E}[\Delta_t] \le 0,
\qquad
H_1: \mathbb{E}[\Delta_t] > 0,
\qquad
\Delta_t = \log p_{\text{M1}}(S_t) - \log p_{\text{B0}}(S_t).
$$

- Teste: Wilcoxon signed-rank unilateral.
- Nível: $\alpha = 0{,}05$.
- Efeito mínimo de relevância prática: $\overline{\Delta} \ge 0{,}05$ nats/concurso.
- IC 95%: bootstrap BCa com 10.000 reamostragens por concurso.

H1 é a **única** hipótese confirmatória. Não há correção de multiplicidade para ela porque é um único teste. B0 é o piso correto para "sinal": comparar M1 apenas com B2b testaria se o ajuste PL agrega sobre frequências empíricas, mas não se há desvio do acaso. Essa segunda pergunta é a hipótese secundária **S2**.

> [!NOTE]
> H1 testa **estrutura global de frequência** (viés de bolas), não dependência temporal: M1 é estático e não usa a ordem dos concursos. A hipótese de sinal **temporal/neural** é S1 (M3 vs B2a), secundária. O veredito deve declarar qual tipo de sinal foi encontrado; H1 não autoriza afirmar que a LSTM prevê o futuro.

### 3.2 Hipóteses secundárias (família Holm, 5 testes)

| ID | Hipótese | Métrica | Amostra | Teste |
|---|---|---|---|---|
| S1 | M3 supera B2a | log-score de conjunto | Teste A | Wilcoxon unilateral |
| S2 | M1 supera B2b | log-score de conjunto | Teste A | Wilcoxon unilateral |
| S3 | M1 supera B0 | média de acertos | Teste B (11 Viradas) | teste exato hipergeométrico |
| S4 | M1 supera B1 | média de acertos | Teste B (11 Viradas) | permutação de sinais exata |
| S5 | portfólio M1-K supera B0-K | $A_{\text{bar}}$, $K = 10$ | Teste B (11 Viradas) | Monte Carlo com 10.000 réplicas |

Correção de Holm sobre os 5 $p$-valores. Se um teste não puder ser executado (por exemplo, M3 não convergir), ele é reportado como "não executado" e a família é corrigida com os testes restantes; a ausência não pode ser interpretada como evidência.

**Equivalência com a `math_spec.md`:** H1 ↔ H1; S1 ↔ H3 (M3 vs B2a); S2 ↔ H2 (M1 vs B2b); S3 ↔ H4 (acertos vs B0); S4 ↔ H5 (acertos vs B1); S5 ↔ H6 (portfólio). A numeração difere apenas para manter a ordem de importância no pré-registro.

### 3.3 Análises exploratórias

Não entram no veredito:

- M2 vs M1;
- M4 vs M1;
- M5-CCE vs B2a-CCE;
- Brier marginal, calibração, ECE;
- métricas por dezena;
- grade de $W \in \{10,20,40\}$;
- grade de $\alpha \in \{0{,}01;0{,}05;0{,}10;0{,}25\}$;
- sensibilidade de $\kappa \in \{0{,}1;0{,}5;2{,}0\}$;
- grades de $K \in \{1,5,10,20,50\}$;
- análise por ano ou por faixa de dezenas.

Toda análise exploratória deve ser rotulada como tal e reportada com IC, sem $p$-valor como afirmação de descoberta.

---

## 4. Dados e splits

### 4.1 Snapshot

- Dataset processado: `data/processed/dataset_megasena_virada_processed.csv`.
- Hash SHA-256: `247c36d7e1ba8b84e416b2e84ff7aa25b1db1972ef416c000ef212531e762974`.
- 3.074 concursos, 1–3074, 17 Viradas.
- Último concurso: 3074, 2026-10-24.
- Nenhum concurso posterior a 3074 entra na análise confirmatória.

### 4.2 Teste A (primário)

| Item | Definição |
|---|---|
| Conjunto de teste | 300 concursos regulares mais recentes: 2773–3074, excluindo 2810 e 2955 |
| Período | 2024-09-12 a 2026-10-24 |
| Treino de referência | concursos 1–2772 |
| Validação para seleção | 300 concursos regulares 2471–2772 |
| Ajuste após seleção | concursos 1–2772 |
| Uso | H1, S1, S2 |

### 4.3 Teste B (secundário)

| `virada_ano` | Concurso | Data | Treino (concurso < alvo) |
|---:|---:|---|---|
| 2015 | 1775 | 2015-12-31 | 1–1774 |
| 2016 | 1890 | 2016-12-31 | 1–1889 |
| 2017 | 2000 | 2017-12-31 | 1–1999 |
| 2018 | 2110 | 2018-12-31 | 1–2109 |
| 2019 | 2220 | 2019-12-31 | 1–2219 |
| 2020 | 2330 | 2020-12-31 | 1–2329 |
| 2021 | 2440 | 2021-12-31 | 1–2439 |
| 2022 | 2550 | 2022-12-31 | 1–2549 |
| 2023 | 2670 | 2023-12-31 | 1–2669 |
| 2024 | 2810 | 2024-12-31 | 1–2809 |
| 2025 | 2955 | 2026-01-01 | 1–2954 |

Cada fold treina apenas com concursos estritamente anteriores ao concurso-alvo. A validação de cada fold é o último 10% dos concursos **regulares** do treino (mínimo de 100 concursos), sem incluir Viradas.

### 4.4 Independência

O Teste A e o Teste B **não são independentes** para as Viradas de 2024 e 2025. O Teste A é confirmatório; o Teste B é secundário e não pode, sozinho, gerar "evidência".

---

## 5. Modelos e baselines

### 5.1 Modelo primário

**M1 — Plackett–Luce marginalizado sobre permutações** (`math_spec.md` §4.2):

$$
P_{\text{M1}}(S) = \sum_{\pi \in \operatorname{Perm}(S)} \prod_{j=1}^{6} \frac{w_{\pi_j}}{W - \sum_{l<j} w_{\pi_l}}.
$$

- 60 pesos, $\sum_i \log w_i = 0$ (59 graus de liberdade efetivos).
- Ajuste por máxima verossimilhança de conjunto.
- Sem uso de histórico (modelo estático).
- Regularização L2: $\lambda \in \{0; 1{,}0\times10^{-4}; 1{,}0\times10^{-3}\}$, escolhida na validação interna ao fold.
- Otimizador Adam, `lr = 1e-2`, `max_epochs = 100`, early stopping `patience = 10` sobre log-score de validação.
- 3 reinicializações; reportar dispersão.

### 5.2 Baselines do modelo primário

**B0 — uniforme (piso de H1):**

$$
p_{\text{B0}}(S) = \frac{1}{\binom{60}{6}}.
$$

**B2b — M1 com pesos empíricos (mesma família, usado em S2):**

$$
w_i \propto \hat f_i = \frac{\#\{t \in \text{treino} : i \in S_t\}}{N_{\text{tr}}}.
$$

Sem hiperparâmetros.

### 5.3 Modelos secundários

- **M3** — produto posicional normalizado por $Z$, com $P_j$ produzidas pela LSTM de 6 cabeças.
- **M5-SET** — a mesma arquitetura, treinada com $-\sum_j \log P_{t,j}(d_{t,j}) + \log Z_t$. É a variante usada em **S1**.
- **M5-CCE** — a mesma arquitetura, treinada com soma de CCEs (exploratória; não entra em S1).

### 5.4 Baselines

| ID | Definição | Tipo |
|---|---|---|
| B0 | Uniforme sobre $\Omega$ | probabilístico |
| B1 | 6 dezenas mais frequentes no treino; desempate por menor dezena | pontual |
| B2a | M3 com marginais posicionais empíricas suavizadas ($\kappa = 1{,}0$) | probabilístico |
| B2b | M1 com pesos empíricos | probabilístico |
| B3 | 6 dezenas do concurso imediatamente anterior | pontual |

### 5.5 Portfólio

- $K = 10$ para S5.
- Filtro de diversidade `max_shared_numbers = 4`, idêntico para todos os modelos.
- Distribuição nula de B0-K simulada com o mesmo $K$ e o mesmo filtro.
- Para a avaliação científica de log-score, o filtro é desativado (a amostra é i.i.d.).

---

## 6. Métricas

### 6.1 Primária

Log-score de conjunto:

$$
\text{LS} = \frac{1}{n}\sum_{t=1}^{n} \log p(S_t).
$$

Reportar $\overline{\Delta}$, IC 95% BCa, $p$-valor unilateral, e $\exp(\overline{\Delta})$.

### 6.2 Secundárias

- Acertos $k_t = |\hat S_t \cap S_t|$; média e distribuição.
- $A_{\text{bar}}$ e $A_{\text{max}}$ do portfólio de $K$ jogos.
- Contagem de quadras/quinas/senas.
- Brier marginal e calibração (exploratórios).
- PL-CCE para M5-CCE vs B2a-CCE (exploratório).

### 6.3 Proibições

- Acurácia, precisão, revocação, F1, especificidade não são evidência principal.
- NLL de B1/B3 não entra no teste.
- NLL de M5-CCE não é comparada com NLL de conjunto.
- Nenhum resultado de melhor configuração sem Holm.
- Nenhuma métrica por dezena sem correção de multiplicidade.

---

## 7. Treino e hiperparâmetros

### 7.1 Configuração primária

```yaml
training:
  max_epochs: 100
  early_stopping:
    monitor: "val_loss_unweighted"
    patience: 10
    restore_best_weights: true
    mode: "min"
  reduce_lr:
    monitor: "val_loss_unweighted"
    factor: 0.5
    patience: 5
    min_lr: 1.0e-6
  optimizer:
    type: "adam"
    learning_rate: 1.0e-3   # M1/M2: 1.0e-2
  batch_size: 64
  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
  sample_weight_alpha: 1.0
  window_size: 20
  validation_fraction: 0.10
  shuffle: "batches_only"
```

### 7.2 Regras

- `max_epochs = 100` para todos os modelos treinados por gradiente.
- Early stopping com paciência 10 sobre a perda de validação **sem peso**.
- Restauração dos melhores pesos.
- 10 seeds para M3/M5; 3 reinicializações para M1/M2.
- Nenhuma seed é escolhida por desempenho; a inferência usa a média das métricas por seed.
- A validação nunca contém o alvo nem concursos posteriores a ele.
- `shuffle = "batches_only"`; nunca misturar treino/validação/teste.

### 7.3 Seleção de hiperparâmetros

Toda seleção é **interna ao fold** e usa apenas dados anteriores ao alvo. Nenhum hiperparâmetro é escolhido com o conjunto de teste do Teste A (2773–3074) nem com as Viradas do Teste B.

- **Teste A:** validação = 300 concursos regulares 2471–2772. Ajuste de referência após a seleção = 1–2772.
- **Cada fold do Teste B:** validação = último 10% dos concursos **regulares** do treino do fold (mínimo de 100 concursos). Ajuste final = todos os concursos anteriores à Virada.
- **M1/M2:** $\lambda \in \{0;\,10^{-4};\,10^{-3}\}$, escolhido por log-score de validação.
- **M3/M5:** $W \in \{10,20,40\}$, escolhido por log-score de validação com 3 seeds; a configuração escolhida é reajustada com 10 seeds.
- **Desempate:** em caso de empate numérico, prefira o modelo mais simples — menor $W$ e maior $\lambda$.
- $\alpha = 1{,}0$ é a configuração primária. $\alpha < 1$ é ablação exploratória, restrita a M3/M5.
- $\kappa = 1{,}0$ é a suavização primária de B2a; $\{0{,}1;\,0{,}5;\,2{,}0\}$ é sensibilidade.
- A grade de $\alpha$, a grade de $\kappa$ e a grade de $W$ não usada na seleção são exploratórias.
- Se o orçamento computacional impedir a grade completa, a grade é reduzida por emenda registrada **antes** de qualquer resultado; a configuração primária $W = 20$ permanece como fallback.

---

## 8. Poder estatístico

Para teste unilateral, $\alpha = 0{,}05$, poder 80%:

$$
\text{MDE} = \frac{2{,}486\,\sigma_d}{\sqrt{n}}.
$$

| Amostra | $n$ | MDE |
|---|---:|---:|
| Teste A (log-score) | 300 | $0{,}144\,\sigma_d$ |
| Teste B (11 Viradas, log-score) | 11 | $0{,}750\,\sigma_d$ |
| Teste B (acertos) | 11 | $\approx 0{,}53$ acertos (média sob $H_1 \approx 1{,}13$) |

> [!WARNING]
> O Teste B tem poder muito baixo. Um resultado não significativo nele **não prova** ausência de sinal. A conclusão principal vem do Teste A.

---

## 9. Sanidade

Executar antes de qualquer conclusão:

1. **Rótulos permutados:** embaralhar os conjuntos-alvo entre concursos; M1/M3 não podem vencer B0 (nem B2b/B2a) no Teste A.
2. **Entradas aleatórias:** substituir as janelas por multi-hot aleatórios; M3/M5 não podem vencer os baselines.
3. **Sentinela de vazamento:** inserir um concurso futuro fictício; o splitter deve rejeitá-lo.
4. **Paridade de $K$:** todo portfólio comparado tem o mesmo $K$ e o mesmo filtro.
5. **Normalização:** todo modelo probabilístico soma 1 em universo reduzido.
6. **Reprodutibilidade:** duas execuções com a mesma seed produzem o mesmo resultado dentro da tolerância documentada.

Se qualquer sanidade falhar, o veredito é **erro/vazamento**, não evidência.

---

## 10. Critérios de veredito

O veredito é composto por **tipo de sinal** e **força da evidência**. Não existe um único "sinal" genérico.

| Componente | Condição |
|---|---|
| **Sinal global de frequência** | H1 rejeitada ($p < 0{,}05$ unilateral), $\overline{\Delta} \ge 0{,}05$ nats/concurso, IC 95% exclui 0, sanidades aprovadas, sinal consistente em $\ge 8/11$ folds de Virada (descritivo) |
| **Valor do ajuste PL** | S2 rejeitada após Holm, com a mesma direção e limiar de 0,05 nats/concurso |
| **Sinal temporal/neural** | S1 rejeitada após Holm, com a mesma direção e limiar de 0,05 nats/concurso |
| **Sinal de acerto/portfólio** | S3, S4 ou S5 rejeitada após Holm, na direção pré-registrada (evidência secundária, não substitui o log-score) |
| **Inconclusivo** | A hipótese relevante não foi rejeitada, mas o IC 95% inclui efeitos $\ge 0{,}05$ nats/concurso; **ou** foi rejeitada mas a robustez falha |
| **Sem evidência** | A hipótese relevante não foi rejeitada e o IC 95% exclui efeitos $\ge 0{,}05$ nats/concurso, com sanidades aprovadas |
| **Erro/vazamento** | Qualquer sanidade falha, ou o modelo vence de forma implausível |

O limiar de 0,05 nats/concurso é pré-registrado e não pode ser alterado após ver os resultados. O relatório deve declarar separadamente cada componente; é proibido resumir tudo em "o modelo funciona" ou "o modelo não funciona".

---

## 11. Regras de parada e emendas

- O experimento para quando todos os modelos pré-registrados forem executados e as sanidades forem verificadas.
- Não há "parada por resultado promissor" nem seleção de seed.
- Qualquer emenda deve ser datada, assinada e registrada antes de olhar o resultado do teste afetado.
- Emenda após o resultado é rotulada **exploratória** e não sustenta a conclusão primária.
- Se um modelo não convergir, isso é reportado; não se substitui por outro sem emenda.

---

## 12. Prospectivo 2026

- Antes de 2026-12-31, treinar o modelo primário com todos os concursos até o último disponível antes do sorteio.
- Gerar $K = 10$ jogos com a configuração pré-registrada, seed fixa e filtro de diversidade.
- Gravar `artifacts/generations/<timestamp>_<hash>.json` com modelo, dataset, config, commit, seed, $T$, $K$, diversidade e jogos.
- Após o sorteio, avaliar descritivamente; $n = 1$, sem teste de significância.
- Nenhum ajuste posterior ao sorteio.

O prospectivo 2026 é um registro operacional, não uma confirmação científica.

---

## 13. Não objetivos

- Não garantir retorno financeiro.
- Não recomendar apostas.
- Não provar que a loteria é previsível.
- Não esconder resultados negativos.
- Não substituir o julgamento humano sobre a aleatoriedade dos sorteios.

---

## 14. Congelamento

Este pré-registro será considerado congelado quando:

- [ ] `math_spec.md` aprovada.
- [ ] `ADR-002` aprovado.
- [ ] `ADR-003` aprovado.
- [ ] Hashes dos três documentos registrados abaixo.
- [ ] Assinatura de aprovação registrada.

| Documento | SHA-256 |
|---|---|
| `spec/math_spec.md` | `cbf41468b2fb2c51ed691d4f130c2dd3c1fca3193fcd164a1ff56a2ff79fea0d` |
| `spec/adr/ADR-002-modelo-probabilistico-primario.md` | `c222b1f7a834a04b4190bda8beb924dd76cf2d36564c94db30a64f7107c05356` |
| `spec/adr/ADR-003-likelihood-e-comparabilidade.md` | `35284742678b833f04930e51f6afc291d2691ad74e253ae2d47b8e30fbeacb8f` |

| Papel | Nome | Data | Decisão |
|---|---|---|---|
| Autor | — | 05/11/2026 | Proposto |
| Revisor técnico | — | — | Pendente |
| Aprovação final | — | — | Pendente |
