"""Módulo de testes estatísticos, sanidade e veredito padronizado.

Implementa:

- Teste de permutação para NLL e acertos (não paramétrico, distribuição nula empírica).
- Teste de Wilcoxon signed-rank pareado (via metrics.py).
- Teste de sinais pareado (via metrics.py).
- Bootstrap BCa por concurso (via metrics.py).
- Correção de Holm para múltiplas comparações (via metrics.py).
- Verificações de sanidade (RV-009):
  1. Rótulos permutados: modelo não deve vencer baseline.
  2. Entradas aleatórias: modelo não deve vencer baseline.
  3. Sentinela de vazamento: concursos futuros rejeitados.
- Veredito padronizado (RV-011): Evidência / Inconclusivo / Sem evidência / Erro-vazamento.

Cobre: math_spec.md §8 (estatística e veredito),
ADR-002/ADR-003 (hipóteses e comparabilidade),
RV-004 a RV-011 (testes, poder, sanidade, veredito).
P0-09: implementação de estatística e sanidade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from lstm_virada.metrics import (
    bootstrap_bca_ic_medio,
    correcao_holm,
    tamanho_efeito_cohen_d,
)

NUM_CONCURSOS_SANIDADE: Final[int] = 100
ALFA_TESTE: Final[float] = 0.05
LIMIAR_RELEVANCIA_NATS: Final[float] = 0.05


class StatsError(ValueError):
    """Exceção levantada quando testes estatísticos encontram dados inválidos."""


@dataclass(frozen=True)
class ResultadoTeste:
    """Resultado de um teste estatístico."""

    nome: str
    estatistico: float
    p_valor: float
    rejeitada: bool  # após correção de Holm, se aplicável
    tamanho_efeito: float
    ic_95: tuple[float, float] | None
    alternativa: str


@dataclass(frozen=True)
class Veredito:
    """Veredito padronizado do experimento."""

    categoria: str  # "Evidencia", "Inconclusivo", "SemEvidencia", "ErroVazamento"
    tipo_sinal: (
        str  # "global_frequencia", "ajuste_PL", "temporal_neural", "acerto_portfolio"
    )
    detalhes: dict


# ==============================================================================
# 1. Teste de Permutação
# ==============================================================================
def teste_permutacao_nll(
    log_p_modelo: np.ndarray,
    log_p_baseline: np.ndarray,
    n_permutacoes: int = 10000,
    seed: int = 42,
) -> float:
    """Teste de permutação para comparação de NLL entre modelo e baseline.

    Sob H0 (modelo não é melhor que baseline), os rótulos "modelo" e "baseline"
    podem ser trocados dentro de cada concurso sem alterar a distribuição conjunta.
    Calculamos a diferença de NLL média observada e comparamos com a distribuição
    obtida ao trocar aleatoriamente os rótulos.

    Args:
        log_p_modelo: array (n,) com log p_modelo(S_t).
        log_p_baseline: array (n,) com log p_baseline(S_t).
        n_permutacoes: número de permutações para a distribuição nula.
        seed: semente aleatória.

    Returns:
        p-valor unilateral (H1: modelo tem NLL menor que baseline).
    """
    log_p_modelo = np.asarray(log_p_modelo, dtype=np.float64)
    log_p_baseline = np.asarray(log_p_baseline, dtype=np.float64)
    n = len(log_p_modelo)
    if n != len(log_p_baseline):
        raise StatsError("log_p_modelo e log_p_baseline devem ter o mesmo tamanho.")

    # Diferença observada: NLL_baseline - NLL_modelo (positivo = modelo melhor)
    nll_modelo = -log_p_modelo
    nll_baseline = -log_p_baseline
    delta_observado = np.mean(nll_baseline - nll_modelo)

    rng = np.random.default_rng(seed)
    contador = 0

    for _ in range(n_permutacoes):
        # Troca aleatória dos rótulos dentro de cada concurso
        mascara_troca = rng.integers(0, 2, size=n).astype(bool)
        nll_m_perm = np.where(mascara_troca, nll_baseline, nll_modelo)
        nll_b_perm = np.where(mascara_troca, nll_modelo, nll_baseline)
        delta_perm = np.mean(nll_b_perm - nll_m_perm)
        if delta_perm >= delta_observado:
            contador += 1

    p_valor = (contador + 1) / (n_permutacoes + 1)
    return float(p_valor)


def teste_permutacao_acertos(
    acertos_modelo: np.ndarray,
    acertos_baseline: np.ndarray,
    n_permutacoes: int = 10000,
    seed: int = 42,
) -> float:
    """Teste de permutação para comparação de acertos entre modelo e baseline.

    Args:
        acertos_modelo: array (n,) com k_t do modelo.
        acertos_baseline: array (n,) com k_t do baseline.
        n_permutacoes: número de permutações.
        seed: semente aleatória.

    Returns:
        p-valor unilateral (H1: modelo tem mais acertos que baseline).
    """
    acertos_modelo = np.asarray(acertos_modelo, dtype=np.float64)
    acertos_baseline = np.asarray(acertos_baseline, dtype=np.float64)
    n = len(acertos_modelo)
    if n != len(acertos_baseline):
        raise StatsError("acertos_modelo e acertos_baseline devem ter o mesmo tamanho.")

    delta_observado = np.mean(acertos_modelo - acertos_baseline)

    rng = np.random.default_rng(seed)
    contador = 0

    for _ in range(n_permutacoes):
        mascara_troca = rng.integers(0, 2, size=n).astype(bool)
        m_perm = np.where(mascara_troca, acertos_baseline, acertos_modelo)
        b_perm = np.where(mascara_troca, acertos_modelo, acertos_baseline)
        delta_perm = np.mean(m_perm - b_perm)
        if delta_perm >= delta_observado:
            contador += 1

    p_valor = (contador + 1) / (n_permutacoes + 1)
    return float(p_valor)


# ==============================================================================
# 2. Sanidade (RV-009)
# ==============================================================================
def sanitidade_rotulos_permutados(
    log_p_modelo_permutado: np.ndarray,
    log_p_baseline: np.ndarray,
    alpha: float = ALFA_TESTE,
) -> tuple[bool, float]:
    """Sanidade 1: com rótulos permutados, o modelo não deve vencer o baseline.

    Se o modelo "vence" (p < alpha) com rótulos aleatórios, há vazamento ou
    overfit, e o resultado não é confiável.

    Args:
        log_p_modelo_permutado: log p do modelo com alvos permutados.
        log_p_baseline: log p do baseline nos mesmos concursos.
        alpha: nível de significância.

    Returns:
        (aprovou_sanidade, p_valor).
    """
    p_valor = teste_permutacao_nll(log_p_modelo_permutado, log_p_baseline)
    aprovou = p_valor >= alpha  # não deve rejeitar H0 com rótulos aleatórios
    return aprovou, p_valor


def sanitidade_entradas_aleatorias(
    log_p_modelo_entrada_aleatoria: np.ndarray,
    log_p_baseline: np.ndarray,
    alpha: float = ALFA_TESTE,
) -> tuple[bool, float]:
    """Sanidade 2: com entradas aleatórias, o modelo não deve vencer o baseline.

    Args:
        log_p_modelo_entrada_aleatoria: log p do modelo com janelas aleatórias.
        log_p_baseline: log p do baseline nos mesmos concursos.
        alpha: nível de significância.

    Returns:
        (aprovou_sanidade, p_valor).
    """
    p_valor = teste_permutacao_nll(log_p_modelo_entrada_aleatoria, log_p_baseline)
    aprovou = p_valor >= alpha
    return aprovou, p_valor


def sanitidade_sentinela_futura(
    as_of_concurso: int,
    ultimo_concurso_disponivel: int,
) -> tuple[bool, str]:
    """Sanidade 3: concurso de corte além do último disponível deve ser rejeitado.

    Args:
        as_of_concurso: concurso de corte proposto.
        ultimo_concurso_disponível: último concurso no dataset.

    Returns:
        (aprovou_sanidade, mensagem).
    """
    if as_of_concurso > ultimo_concurso_disponivel:
        return False, (
            f"Sentinela futura detectada: as_of={as_of_concurso} > "
            f"último disponível={ultimo_concurso_disponivel}."
        )
    return True, "Sentinela futura rejeitada corretamente."


# ==============================================================================
# 3. Veredito Padronizado (RV-011)
# ==============================================================================
def veredito_padronizado(
    p_valor_primario: float,
    delta_medio_primario: float,
    ic_95_primario: tuple[float, float] | None,
    sanidades_aprovadas: bool,
    efeito_relevante: bool,
) -> Veredito:
    """Determina o veredito padronizado do experimento.

    Regras (math_spec.md §8.5):
    - Evidência: H1 rejeitada (p < 0.05), delta >= 0.05 nats, IC exclui 0,
      sanidades aprovadas.
    - Inconclusivo: H1 não rejeitada, mas IC inclui efeitos >= 0.05;
      ou rejeitada mas robustez falha.
    - Sem evidência: H1 não rejeitada e IC exclui efeitos >= 0.05.
    - Erro-vazamento: qualquer sanidade falha.

    Args:
        p_valor_primario: p-valor do teste primário (H1).
        delta_medio_primario: Delta_t médio observado (nats/concurso).
        ic_95_primario: IC 95% para delta_medio.
        sanidades_aprovadas: se todas as verificações de sanidade passaram.
        efeito_relevante: se o efeito é >= 0.05 nats/concurso.

    Returns:
        Veredito com categoria e tipo de sinal.
    """
    if not sanidades_aprovadas:
        return Veredito(
            categoria="ErroVazamento",
            tipo_sinal="nenhum",
            detalhes={
                "motivo": "Sanidade falhou: possível vazamento ou overfit.",
            },
        )

    p_rejeitada = p_valor_primario < ALFA_TESTE
    ic_exclui_zero = (
        ic_95_primario is not None and ic_95_primario[0] > 0 and ic_95_primario[1] > 0
    )
    ic_inclui_relevante = (
        ic_95_primario is not None and ic_95_primario[1] >= LIMIAR_RELEVANCIA_NATS
    )

    if p_rejeitada and efeito_relevante and ic_exclui_zero:
        return Veredito(
            categoria="Evidencia",
            tipo_sinal="global_frequencia",
            detalhes={
                "p_valor": p_valor_primario,
                "delta_medio": delta_medio_primario,
                "ic_95": ic_95_primario,
            },
        )

    if (not p_rejeitada and ic_inclui_relevante) or (
        p_rejeitada and not efeito_relevante
    ):
        return Veredito(
            categoria="Inconclusivo",
            tipo_sinal="indeterminado",
            detalhes={
                "p_valor": p_valor_primario,
                "delta_medio": delta_medio_primario,
                "ic_95": ic_95_primario,
            },
        )

    return Veredito(
        categoria="SemEvidencia",
        tipo_sinal="nenhum",
        detalhes={
            "p_valor": p_valor_primario,
            "delta_medio": delta_medio_primario,
            "ic_95": ic_95_primario,
        },
    )


# ==============================================================================
# 4. Correção de Holm (wrapper)
# ==============================================================================
def aplicar_holm(
    p_valores: dict[str, float],
    alpha: float = ALFA_TESTE,
) -> dict[str, dict]:
    """Wrapper para correção de Holm (delegado a metrics.correcao_holm).

    Args:
        p_valores: dicionário {nome_teste: p_valor}.
        alpha: nível de significância.

    Returns:
        Resultado da correção de Holm.
    """
    return correcao_holm(p_valores, alpha=alpha)


# ==============================================================================
# 5. Tamanho do Efeito
# ==============================================================================
def calcular_tamanho_efeito(
    delta: np.ndarray,
) -> float:
    """Wrapper para tamanho de efeito (delegado a metrics.tamanho_efeito_cohen_d)."""
    return tamanho_efeito_cohen_d(delta)


# ==============================================================================
# 6. Bootstrap IC (wrapper)
# ==============================================================================
def calcular_ic_bootstrap(
    delta: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Wrapper para bootstrap BCa (delegado a metrics.bootstrap_bca_ic_medio)."""
    return bootstrap_bca_ic_medio(delta, n_bootstrap=n_bootstrap, seed=seed)
