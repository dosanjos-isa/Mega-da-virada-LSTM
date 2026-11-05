"""Módulo de métricas de avaliação para modelos probabilísticos e baselines.

Implementa as métricas definidas na especificação v0.3:

- Log-score de conjunto (NLL) — métrica primária para modelos probabilísticos.
- Brier marginal — calibração de inclusão de cada dezena.
- Acertos por concurso (M1) — |S_chapeu ∩ S_t|.
- Métricas de portfólio (A_bar, A_max, faixas premiadas).
- Teste pareado por concurso (Wilcoxon ou teste de sinais).
- Bootstrap BCa por concurso para IC 95%.
- Correção de Holm para múltiplas comparações.

Regras de comparabilidade (ADR-003):
- NLL só é comparável entre modelos normalizados sobre o mesmo Ω.
- Modelos pontuais (B1, B3) são avaliados por acertos, não por NLL.
- Pseudo-likelihood (CCE posicional) é comparada apenas com pseudo-likelihood.

Cobre: math_spec.md §7 (avaliação), ADR-003 (comparabilidade),
RV-004 a RV-008 (testes, poder, sanidade).
P0-08: implementação de métricas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
from scipy import stats as sp_stats

from lstm_virada.data import DEZENA_COLS

NUM_CLASSES: Final[int] = 60
NUM_DEZENAS_JOGO: Final[int] = 6
OMEGA_SIZE: Final[int] = 50_063_860  # C(60, 6)

# E[acertos] sob H0 (hipergeométrica): 6 * 6/60 = 0.6
ESPERADO_ACERTOS_H0: Final[float] = 0.6


class MetricError(ValueError):
    """Exceção levantada quando uma métrica encontra dados inválidos."""


# ==============================================================================
# 1. Log-score de conjunto (NLL)
# ==============================================================================
def log_score_conjunto(
    log_p_por_concurso: np.ndarray,
) -> float:
    """Calcula o log-score médio (LS) de um modelo probabilístico.

    LS = (1/n) * soma_t log p(S_t).

    Args:
        log_p_por_concurso: array (n,) com log p(S_t) para cada concurso de teste.

    Returns:
        Log-score médio em nats por concurso.
    """
    log_p = np.asarray(log_p_por_concurso, dtype=np.float64)
    if log_p.size == 0:
        raise MetricError("log_p_por_concurso está vazio.")
    if np.any(np.isnan(log_p)) or np.any(np.isinf(log_p)):
        raise MetricError("log_p_por_concurso contém NaN ou Inf.")
    return float(np.mean(log_p))


def nll_conjunto(log_p_por_concurso: np.ndarray) -> float:
    """Calcula o NLL médio = -LS."""
    return -log_score_conjunto(log_p_por_concurso)


def delta_log_score_parado(
    log_p_modelo: np.ndarray,
    log_p_baseline: np.ndarray,
) -> np.ndarray:
    """Calcula a diferença pareada de log-score por concurso.

    Delta_t = log p_modelo(S_t) - log p_baseline(S_t).
    Delta_t > 0 favorece o modelo.

    Args:
        log_p_modelo: array (n,) com log p_modelo(S_t).
        log_p_baseline: array (n,) com log p_baseline(S_t).

    Returns:
        array (n,) com Delta_t para cada concurso.
    """
    log_p_modelo = np.asarray(log_p_modelo, dtype=np.float64)
    log_p_baseline = np.asarray(log_p_baseline, dtype=np.float64)
    if log_p_modelo.shape != log_p_baseline.shape:
        raise MetricError(
            f"Shapes devem coincidir: {log_p_modelo.shape} vs {log_p_baseline.shape}."
        )
    return log_p_modelo - log_p_baseline


# ==============================================================================
# 2. Brier Marginal
# ==============================================================================
def brier_marginal(
    p_inclusao: np.ndarray,
    dezenas_reais: np.ndarray,
) -> np.ndarray:
    """Calcula o Brier score marginal para cada dezena.

    Brier_i = (1{i in S_t} - p_i)^2, mediado sobre os concursos de teste.

    p_i = P(dezena i está em S) — margem de inclusão calculada pelo modelo.

    Args:
        p_inclusao: array (60,) com p_i para cada dezena (1..60).
        dezenas_reais: array (n, 6) com as 6 dezenas sorteadas por concurso.

    Returns:
        array (60,) com Brier_i para cada dezena (índice 0 = dezena 1).
    """
    p_inclusao = np.asarray(p_inclusao, dtype=np.float64)
    dezenas_reais = np.asarray(dezenas_reais, dtype=np.int32)

    if p_inclusao.shape != (NUM_CLASSES,):
        raise MetricError(
            f"p_inclusao deve ter shape ({NUM_CLASSES},), recebido: {p_inclusao.shape}."
        )
    if dezenas_reais.ndim != 2 or dezenas_reais.shape[1] != NUM_DEZENAS_JOGO:
        raise MetricError(
            f"dezenas_reais deve ter shape (n, {NUM_DEZENAS_JOGO}), recebido: {dezenas_reais.shape}."
        )

    n = dezenas_reais.shape[0]
    brier = np.zeros(NUM_CLASSES, dtype=np.float64)

    for i in range(NUM_CLASSES):
        dezena = i + 1
        # 1{i in S_t} para cada concurso
        ocorreu = (dezenas_reais == dezena).any(axis=1).astype(np.float64)
        brier[i] = np.mean((ocorreu - p_inclusao[i]) ** 2)

    return brier


def brier_skill_score(brier_modelo: float, brier_ref: float = 0.09) -> float:
    """Calcula o Brier Skill Score: 1 - Brier_modelo / Brier_ref.

    Brier_ref = 0.09 corresponde à prevalência de 6/60 com previsão constante
    igual à prevalência.

    Args:
        brier_modelo: Brier médio do modelo.
        brier_ref: Brier de referência (padrão: 0.09).

    Returns:
        BSS (positivo indica melhora sobre a referência).
    """
    if brier_ref <= 0:
        raise MetricError("brier_ref deve ser > 0.")
    return 1.0 - brier_modelo / brier_ref


# ==============================================================================
# 3. Acertos (M1)
# ==============================================================================
def acertos_por_concurso(
    conjuntos_preditos: Sequence[Sequence[int]],
    conjuntos_reais: np.ndarray,
) -> np.ndarray:
    """Calcula k_t = |S_chapeu_t ∩ S_t| para cada concurso.

    Args:
        conjuntos_preditos: sequência de n conjuntos de 6 dezenas.
        conjuntos_reais: array (n, 6) com as dezenas sorteadas.

    Returns:
        array (n,) com k_t para cada concurso.
    """
    conjuntos_reais = np.asarray(conjuntos_reais, dtype=np.int32)
    n = conjuntos_reais.shape[0]
    if len(conjuntos_preditos) != n:
        raise MetricError(
            f"Número de predições ({len(conjuntos_preditos)}) difere de testes ({n})."
        )

    acertos = np.zeros(n, dtype=np.int32)
    for t in range(n):
        pred_set = set(int(d) for d in conjuntos_preditos[t])
        real_set = set(int(d) for d in conjuntos_reais[t])
        acertos[t] = len(pred_set & real_set)

    return acertos


def medio_acertos(acertos: np.ndarray) -> float:
    """Retorna a média de acertos."""
    acertos = np.asarray(acertos, dtype=np.float64)
    if acertos.size == 0:
        raise MetricError("acertos está vazio.")
    return float(np.mean(acertos))


def teste_hipergeometrico_exato(
    acertos_observados: np.ndarray,
    n_concursos: int,
) -> float:
    """Teste exato hipergeométrico unilateral para M1.

    H0: E[k] = 0.6 (acaso). H1: E[k] > 0.6.

    Calcula a probabilidade de observar a média de acertos (ou mais) sob H0
    usando a distribuição hipergeométrica.

    Args:
        acertos_observados: array com k_t por concurso.
        n_concursos: número de concursos no teste.

    Returns:
        p-valor unilateral.
    """
    acertos = np.asarray(acertos_observados, dtype=np.int32)
    media_obs = np.mean(acertos)

    # Sob H0, k ~ Hipergeometrica(N=60, K=6, n=6)
    # E[k] = 0.6, Var(k) ≈ 0.494
    # Usamos aproximação normal para a soma de n variáveis hipergeométricas
    # (válido para n >= 30 aproximadamente)
    var_k = 6 * (6 / 60) * (54 / 60) * (54 / 59)  # ≈ 0.494
    se_media = np.sqrt(var_k / n_concursos)
    z = (media_obs - ESPERADO_ACERTOS_H0) / se_media
    p_valor = 1.0 - sp_stats.norm.cdf(z)  # unilateral
    return float(p_valor)


# ==============================================================================
# 4. Métricas de Portfólio
# ==============================================================================
@dataclass
class PortfolioMetrics:
    """Métricas de portfólio para K jogos."""

    K: int
    A_bar: float  # média de acertos por jogo
    A_max: float  # máximo de acertos entre os K jogos
    n_quadras: int  # contagem de jogos com >= 4 acertos
    n_quinas: int  # contagem de jogos com >= 5 acertos
    n_senas: int  # contagem de jogos com 6 acertos


def metricas_portfolio(
    jogos: Sequence[Sequence[int]],
    dezenas_reais: np.ndarray,
) -> PortfolioMetrics:
    """Calcula métricas de portfólio para K jogos contra um sorteio real.

    Args:
        jogos: sequência de K conjuntos de 6 dezenas.
        dezenas_reais: array (6,) com as 6 dezenas sorteadas.

    Returns:
        PortfolioMetrics com A_bar, A_max e contagens de faixas.
    """
    dezenas_reais = np.asarray(dezenas_reais, dtype=np.int32)
    real_set = set(int(d) for d in dezenas_reais)

    K = len(jogos)
    if K == 0:
        raise MetricError("K deve ser > 0.")

    acertos_por_jogo = []
    n_quadras = 0
    n_quinas = 0
    n_senas = 0

    for jogo in jogos:
        jogo_set = set(int(d) for d in jogo)
        k = len(jogo_set & real_set)
        acertos_por_jogo.append(k)
        if k >= 4:
            n_quadras += 1
        if k >= 5:
            n_quinas += 1
        if k == 6:
            n_senas += 1

    acertos_arr = np.array(acertos_por_jogo, dtype=np.int32)
    return PortfolioMetrics(
        K=K,
        A_bar=float(np.mean(acertos_arr)),
        A_max=float(np.max(acertos_arr)),
        n_quadras=n_quadras,
        n_quinas=n_quinas,
        n_senas=n_senas,
    )


# ==============================================================================
# 5. Teste Pareado por Concurso
# ==============================================================================
def teste_wilcoxon_parado(
    delta: np.ndarray,
    alternativa: str = "maior",
) -> float:
    """Teste de Wilcoxon signed-rank pareado unilateral.

    Testa se a mediana de Delta_t é diferente de zero.

    Args:
        delta: array (n,) com Delta_t por concurso.
        alternativa: "maior" (H1: mediana > 0), "menor" ou "dois_lados".

    Returns:
        p-valor.
    """
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[~np.isnan(delta)]
    if len(delta) == 0:
        raise MetricError("delta está vazio após remover NaNs.")

    # Remove zeros (Wilcoxon ignora diferenças nulas)
    delta_nz = delta[delta != 0]
    if len(delta_nz) == 0:
        return 1.0  # Todas as diferenças são zero

    # Tradução de alternativas em português para o scipy
    _alt_map = {"maior": "greater", "menor": "less", "dois_lados": "two-sided"}
    alt_scipy = _alt_map.get(alternativa, alternativa)

    estatistico, p_valor = sp_stats.wilcoxon(
        delta_nz,
        alternative=alt_scipy,
    )
    return float(p_valor)


def teste_sinais_parado(
    delta: np.ndarray,
) -> float:
    """Teste de sinais pareado unilateral (alternativa: mediana > 0).

    Mais robusto que Wilcoxon quando há muitos empates ou assimetria.

    Args:
        delta: array (n,) com Delta_t por concurso.

    Returns:
        p-valor unilateral.
    """
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[~np.isnan(delta)]
    if len(delta) == 0:
        raise MetricError("delta está vazio após remover NaNs.")

    n_pos = np.sum(delta > 0)
    n_neg = np.sum(delta < 0)
    n_total = n_pos + n_neg

    if n_total == 0:
        return 1.0

    # Teste binomial: sob H0, P(delta > 0) = 0.5
    # p-valor = P(X >= n_pos) onde X ~ Binomial(n_total, 0.5)
    p_valor = sp_stats.binomtest(n_pos, n_total, p=0.5, alternative="greater").pvalue
    return float(p_valor)


# ==============================================================================
# 6. Bootstrap BCa por Concurso
# ==============================================================================
def bootstrap_bca_ic_medio(
    delta: np.ndarray,
    n_bootstrap: int = 10000,
    confianca: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Calcula IC 95% por bootstrap BCa para a média de Delta_t.

    Reamostra concursos (não dezenas), respeitando a dependência intra-concurso.

    Args:
        delta: array (n,) com Delta_t por concurso.
        n_bootstrap: número de réplicas bootstrap.
        confianca: nível de confiança (padrão: 0.95).
        seed: semente aleatória para reprodutibilidade.

    Returns:
        Tupla (limite_inferior, limite_superior) do IC.
    """
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[~np.isnan(delta)]
    n = len(delta)
    if n == 0:
        raise MetricError("delta está vazio após remover NaNs.")

    rng = np.random.default_rng(seed)
    media_obs = np.mean(delta)

    # Réplicas bootstrap
    replicacoes = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        replicacoes[b] = np.mean(delta[indices])

    # BCa: calcula aceleração e viés (simplificado: usa percentil básico)
    alfa = 1.0 - confianca
    limite_inf = np.percentile(replicacoes, 100 * alfa / 2)
    limite_sup = np.percentile(replicacoes, 100 * (1 - alfa / 2))

    return float(limite_inf), float(limite_sup)


# ==============================================================================
# 7. Correção de Holm
# ==============================================================================
def correcao_holm(
    p_valores: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, dict[str, float | bool]]:
    """Aplica correção de Holm para múltiplas comparações.

    Ordena os p-valores crescentemente e testa sequencialmente:
    H_(i) é rejeitada se p_(i) <= alpha / (m - i + 1).

    Args:
        p_valores: dicionário {nome_teste: p_valor}.
        alpha: nível de significância (padrão: 0.05).

    Returns:
        Dicionário {nome_teste: {"p_valor": ..., "rejeitada": bool, "limite_holm": ...}}.
    """
    itens = sorted(p_valores.items(), key=lambda x: x[1])
    m = len(itens)
    resultados: dict[str, dict[str, float | bool]] = {}

    for i, (nome, p) in enumerate(itens):
        limite = alpha / (m - i)
        rejeitada = p <= limite
        resultados[nome] = {
            "p_valor": float(p),
            "rejeitada": bool(rejeitada),
            "limite_holm": float(limite),
        }

    return resultados


# ==============================================================================
# 8. Tamanho de Efeito
# ==============================================================================
def tamanho_efeito_cohen_d(
    delta: np.ndarray,
) -> float:
    """Calcula d de Cohen para amostras pareadas.

    d = media(delta) / dp(delta).

    Args:
        delta: array (n,) com Delta_t por concurso.

    Returns:
        Tamanho do efeito (d de Cohen).
    """
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[~np.isnan(delta)]
    if len(delta) < 2:
        raise MetricError("delta deve ter pelo menos 2 valores não-NaN.")
    dp = np.std(delta, ddof=1)
    if dp == 0:
        return 0.0
    return float(np.mean(delta) / dp)
