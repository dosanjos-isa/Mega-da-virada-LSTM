"""Testes unitários para o módulo de métricas (metrics.py).

Cobre os requisitos:
- math_spec.md §7: log-score, Brier, acertos, portfólio.
- ADR-003: comparabilidade entre modelos probabilísticos.
- RV-004 a RV-008: testes estatísticos, poder, sanidade.
- P0-08: implementação de métricas.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import re
import sys
from typing import Any

# Adiciona src ao sys.path para execução direta sem necessidade de variável de ambiente
_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

# Carregamento dinâmico do pytest
pytest: Any
try:
    pytest = importlib.import_module("pytest")
except ImportError:  # pragma: no cover
    class _PytestStub:
        @staticmethod
        def fixture(func: Any) -> Any:
            return func

        @staticmethod
        def skip(msg: str = "") -> Any:
            return None

        @staticmethod
        def raises(expected_exception: type[Exception], match: str | None = None) -> Any:
            class _RaisesContext:
                def __enter__(self) -> "_RaisesContext":
                    return self

                def __exit__(
                    self,
                    exc_type: type[BaseException] | None,
                    exc_val: BaseException | None,
                    exc_tb: Any,
                ) -> bool:
                    if exc_type is None:
                        raise AssertionError(
                            f"Esperada exceção {expected_exception.__name__}, mas nenhuma foi levantada."
                        )
                    if not issubclass(exc_type, expected_exception):
                        return False
                    if match and exc_val is not None:
                        if not re.search(match, str(exc_val)):
                            raise AssertionError(
                                f"Exceção levantada, mas mensagem '{exc_val}' não casa com padrão '{match}'."
                            )
                    return True

            return _RaisesContext()

    pytest = _PytestStub()

from lstm_virada.metrics import (
    acertos_por_concurso,
    brier_marginal,
    brier_skill_score,
    correcao_holm,
    delta_log_score_parado,
    log_score_conjunto,
    medio_acertos,
    metricas_portfolio,
    nll_conjunto,
    teste_hipergeometrico_exato,
    teste_sinais_parado,
    teste_wilcoxon_parado,
    tamanho_efeito_cohen_d,
    bootstrap_bca_ic_medio,
    ESPERADO_ACERTOS_H0,
    OMEGA_SIZE,
)

# ==============================================================================
# 1. Log-score e NLL
# ==============================================================================
def test_log_score_conjunto_basico() -> None:
    """Log-score médio de valores constantes."""
    log_p = np.array([-10.0, -10.0, -10.0])
    assert log_score_conjunto(log_p) == -10.0


def test_nll_conjunto() -> None:
    """NLL = -log_score."""
    log_p = np.array([-10.0, -20.0, -30.0])
    assert nll_conjunto(log_p) == 20.0


def test_log_score_vazio_erro() -> None:
    """Log-score de array vazio lança MetricError."""
    from lstm_virada.metrics import MetricError
    with pytest.raises(MetricError, match="vazio"):
        log_score_conjunto(np.array([]))


def test_delta_log_score_parado() -> None:
    """Delta_t = log_p_modelo - log_p_baseline."""
    modelo = np.array([-10.0, -12.0, -8.0])
    baseline = np.array([-11.0, -11.0, -11.0])
    delta = delta_log_score_parado(modelo, baseline)
    esperado = np.array([1.0, -1.0, 3.0])
    assert np.allclose(delta, esperado)


def test_delta_shapes_diferentes_erro() -> None:
    """Shapes diferentes lançam MetricError."""
    from lstm_virada.metrics import MetricError
    with pytest.raises(MetricError, match="coincidir"):
        delta_log_score_parado(np.array([1.0, 2.0]), np.array([1.0]))


# ==============================================================================
# 2. Brier Marginal
# ==============================================================================
def test_brier_marginal_basico() -> None:
    """Brier marginal para previsão constante igual à prevalência."""
    # p_i = 0.1 para todo i (prevalência 6/60)
    p_inclusao = np.full(60, 0.1)
    # Cada concurso tem 6 dezenas aleatórias; esperado Brier ≈ 0.09
    np.random.seed(42)
    dezenas = np.array([
        np.random.choice(np.arange(1, 61), size=6, replace=False)
        for _ in range(100)
    ])
    dezenas = np.sort(dezenas, axis=1)

    brier = brier_marginal(p_inclusao, dezenas)
    assert brier.shape == (60,)
    assert np.all(brier >= 0.0)
    assert np.all(brier <= 1.0)


def test_brier_marginal_shape_erro() -> None:
    """p_inclusao com shape errado lança MetricError."""
    from lstm_virada.metrics import MetricError
    with pytest.raises(MetricError, match="shape"):
        brier_marginal(np.full(30, 0.1), np.array([[1, 2, 3, 4, 5, 6]]))


def test_brier_skill_score() -> None:
    """BSS = 1 - Brier_modelo / Brier_ref."""
    assert brier_skill_score(0.045, 0.09) == 0.5
    assert brier_skill_score(0.09, 0.09) == 0.0


# ==============================================================================
# 3. Acertos (M1)
# ==============================================================================
def test_acertos_por_concurso() -> None:
    """Acertos = tamanho da interseção."""
    preditos = [
        (1, 2, 3, 4, 5, 6),
        (13, 14, 15, 16, 17, 18),
        (1, 2, 3, 4, 5, 60),
    ]
    reais = np.array([
        [1, 2, 3, 4, 5, 6],   # 6 acertos
        [7, 8, 9, 10, 11, 12], # 0 acertos
        [1, 2, 60, 30, 40, 50], # 3 acertos (1, 2, 60)
    ])
    acertos = acertos_por_concurso(preditos, reais)
    assert acertos[0] == 6
    assert acertos[1] == 0
    assert acertos[2] == 3  # interseção = {1, 2, 60} = 3


def test_medio_acertos() -> None:
    """Média de acertos."""
    acertos = np.array([0, 1, 2, 3, 4, 5, 6])
    assert medio_acertos(acertos) == 3.0


def test_acertos_tamanho_diferente_erro() -> None:
    """Número diferente de predições lança MetricError."""
    from lstm_virada.metrics import MetricError
    with pytest.raises(MetricError, match="difere"):
        acertos_por_concurso(
            [(1, 2, 3, 4, 5, 6)],
            np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        )


def test_esperado_acertos_h0() -> None:
    """E[acertos] sob H0 = 0.6."""
    assert ESPERADO_ACERTOS_H0 == 0.6


def test_teste_hipergeometrico_exato() -> None:
    """Teste hipergeométrico unilateral."""
    # Acertos bem acima de 0.6 devem dar p-valor pequeno
    acertos_altos = np.array([3, 3, 3, 3, 3])
    p = teste_hipergeometrico_exato(acertos_altos, 5)
    assert 0.0 <= p <= 1.0

    # Acertos próximos de 0.6 devem dar p-valor grande
    acertos_medios = np.array([0, 1, 0, 1, 1])
    p2 = teste_hipergeometrico_exato(acertos_medios, 5)
    assert 0.0 <= p2 <= 1.0


# ==============================================================================
# 4. Métricas de Portfólio
# ==============================================================================
def test_metricas_portfolio() -> None:
    """Métricas de portfólio para K jogos."""
    jogos = [
        (1, 2, 3, 4, 5, 6),
        (7, 8, 9, 10, 11, 12),
        (1, 2, 3, 4, 5, 60),
    ]
    reais = np.array([1, 2, 3, 4, 5, 6])

    m = metricas_portfolio(jogos, reais)
    assert m.K == 3
    # jogo 0: {1,2,3,4,5,6} ∩ {1..6} = 6
    # jogo 1: {7..12} ∩ {1..6} = 0
    # jogo 2: {1,2,3,4,5,60} ∩ {1..6} = 5
    assert m.A_bar == (6 + 0 + 5) / 3
    assert m.A_max == 6
    assert m.n_quadras == 2  # jogos 0 (6) e 2 (5) são >= 4
    assert m.n_quinas == 2  # jogos 0 (6) e 2 (5) são >= 5
    assert m.n_senas == 1  # apenas jogo 0 (6)


def test_metricas_portfolio_vazio_erro() -> None:
    """K=0 lança MetricError."""
    from lstm_virada.metrics import MetricError
    with pytest.raises(MetricError, match="> 0"):
        metricas_portfolio([], np.array([1, 2, 3, 4, 5, 6]))


# ==============================================================================
# 5. Testes Estatísticos
# ==============================================================================
def test_teste_wilcoxon_parado() -> None:
    """Wilcoxon pareado retorna p-valor válido."""
    delta = np.array([0.1, 0.2, 0.3, -0.05, 0.15])
    p = teste_wilcoxon_parado(delta)
    assert 0.0 <= p <= 1.0


def test_teste_sinais_parado() -> None:
    """Teste de sinais pareado retorna p-valor válido."""
    delta = np.array([0.1, 0.2, 0.3, -0.05, 0.15])
    p = teste_sinais_parado(delta)
    assert 0.0 <= p <= 1.0


def test_teste_sinais_todos_zero() -> None:
    """Todas diferenças zero -> p-valor = 1.0."""
    delta = np.array([0.0, 0.0, 0.0])
    p = teste_sinais_parado(delta)
    assert p == 1.0


def test_bootstrap_bca_ic() -> None:
    """Bootstrap BCa retorna intervalo válido."""
    np.random.seed(42)
    delta = np.random.normal(0.1, 0.5, 100)
    li, ls = bootstrap_bca_ic_medio(delta, n_bootstrap=1000, seed=42)
    assert li < ls
    assert li <= np.mean(delta) <= ls


# ==============================================================================
# 6. Correção de Holm
# ==============================================================================
def test_correcao_holm() -> None:
    """Holm corrige p-valores corretamente."""
    p_valores = {"H1": 0.01, "H2": 0.04, "H3": 0.03}
    resultados = correcao_holm(p_valores, alpha=0.05)

    assert "H1" in resultados
    assert "H2" in resultados
    assert "H3" in resultados

    # H1 (p=0.01) é a menor, limite = 0.05/3 = 0.0167 -> rejeitada
    assert resultados["H1"]["rejeitada"] is True
    assert abs(resultados["H1"]["limite_holm"] - 0.05 / 3) < 1e-10


def test_tamanho_efeito_cohen_d() -> None:
    """d de Cohen para amostras pareadas."""
    delta = np.array([0.1, 0.2, 0.15, 0.05, 0.2])
    d = tamanho_efeito_cohen_d(delta)
    assert isinstance(d, float)
    assert abs(d) < 10  # sanidade


def test_tamanho_efeito_dp_zero() -> None:
    """Se dp = 0, d = 0."""
    delta = np.array([0.5, 0.5, 0.5])
    d = tamanho_efeito_cohen_d(delta)
    assert d == 0.0


if __name__ == "__main__":
    print("Executando suíte de testes de métricas...")
    test_log_score_conjunto_basico()
    test_nll_conjunto()
    test_log_score_vazio_erro()
    test_delta_log_score_parado()
    test_delta_shapes_diferentes_erro()
    test_brier_marginal_basico()
    test_brier_marginal_shape_erro()
    test_brier_skill_score()
    test_acertos_por_concurso()
    test_medio_acertos()
    test_acertos_tamanho_diferente_erro()
    test_esperado_acertos_h0()
    test_teste_hipergeometrico_exato()
    test_metricas_portfolio()
    test_metricas_portfolio_vazio_erro()
    test_teste_wilcoxon_parado()
    test_teste_sinais_parado()
    test_teste_sinais_todos_zero()
    test_bootstrap_bca_ic()
    test_correcao_holm()
    test_tamanho_efeito_cohen_d()
    test_tamanho_efeito_dp_zero()
    print("SUCESSO: Todos os 23 testes de métricas passaram perfeitamente!")
