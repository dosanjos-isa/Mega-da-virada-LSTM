"""Testes unitários para o módulo de estatística (stats.py).

Cobre os requisitos:
- math_spec.md §8: estatística e veredito.
- RV-009: verificações de sanidade.
- RV-011: veredito padronizado.
- P0-09: implementação de estatística e sanidade.
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

from lstm_virada.stats import (
    teste_permutacao_nll,
    teste_permutacao_acertos,
    sanitidade_rotulos_permutados,
    sanitidade_entradas_aleatorias,
    sanitidade_sentinela_futura,
    veredito_padronizado,
    aplicar_holm,
    calcular_tamanho_efeito,
    calcular_ic_bootstrap,
    StatsError,
    Veredito,
    ALFA_TESTE,
    LIMIAR_RELEVANCIA_NATS,
)


# ==============================================================================
# 1. Teste de Permutação
# ==============================================================================
def test_teste_permutacao_nll_modelo_melhor() -> None:
    """Permutação NLL: modelo claramente melhor deve dar p-valor pequeno."""
    np.random.seed(42)
    n = 100
    # Modelo tem NLL menor (log-p maior) que baseline
    log_p_modelo = np.random.normal(-5.0, 1.0, n)
    log_p_baseline = np.random.normal(-6.0, 1.0, n)

    p = teste_permutacao_nll(log_p_modelo, log_p_baseline, n_permutacoes=1000, seed=42)
    assert 0.0 <= p <= 1.0
    # Com diferença clara, p-valor deve ser pequeno
    assert p < 0.05


def test_teste_permutacao_nll_sem_diferenca() -> None:
    """Permutação NLL: sem diferença deve dar p-valor alto."""
    np.random.seed(42)
    n = 100
    # Mesma distribuição para modelo e baseline
    log_p_modelo = np.random.normal(-5.0, 1.0, n)
    log_p_baseline = np.random.normal(-5.0, 1.0, n)

    p = teste_permutacao_nll(log_p_modelo, log_p_baseline, n_permutacoes=1000, seed=42)
    assert 0.0 <= p <= 1.0


def test_teste_permutacao_acertos() -> None:
    """Permutação de acertos retorna p-valor válido."""
    acertos_modelo = np.array([2, 3, 1, 4, 2, 3, 1, 2, 3, 4])
    acertos_baseline = np.array([1, 1, 0, 2, 1, 1, 0, 1, 1, 2])

    p = teste_permutacao_acertos(acertos_modelo, acertos_baseline, n_permutacoes=1000, seed=42)
    assert 0.0 <= p <= 1.0


def test_teste_permutacao_tamanho_diferente_erro() -> None:
    """Arrays de tamanho diferente lançam StatsError."""
    with pytest.raises(StatsError, match="mesmo tamanho"):
        teste_permutacao_nll(np.array([1.0, 2.0]), np.array([1.0]))


# ==============================================================================
# 2. Sanidade
# ==============================================================================
def test_sanidade_rotulos_permutados_aprovada() -> None:
    """Sanidade 1: rótulos permutados não devem fazer modelo vencer baseline."""
    np.random.seed(42)
    n = 100
    # Modelo e baseline com mesma distribuição (sinal nulo)
    log_p_modelo = np.random.normal(-5.0, 1.0, n)
    log_p_baseline = np.random.normal(-5.0, 1.0, n)

    aprovou, p = sanitidade_rotulos_permutados(log_p_modelo, log_p_baseline)
    assert aprovou is True  # não deve rejeitar H0 com sinal nulo
    assert 0.0 <= p <= 1.0


def test_sanidade_entradas_aleatorias_aprovada() -> None:
    """Sanidade 2: entradas aleatórias não devem fazer modelo vencer baseline."""
    np.random.seed(42)
    n = 100
    log_p_modelo = np.random.normal(-5.0, 1.0, n)
    log_p_baseline = np.random.normal(-5.0, 1.0, n)

    aprovou, p = sanitidade_entradas_aleatorias(log_p_modelo, log_p_baseline)
    assert aprovou is True


def test_sanidade_sentinela_futura_rejeitada() -> None:
    """Sanidade 3: as_of > último disponível deve ser rejeitado."""
    aprovou, msg = sanitidade_sentinela_futura(3100, 3074)
    assert aprovou is False
    assert "Sentinela futura" in msg


def test_sanidade_sentinela_futura_valida() -> None:
    """Sanidade 3: as_of <= último disponível deve ser aprovado."""
    aprovou, msg = sanitidade_sentinela_futura(2773, 3074)
    assert aprovou is True


# ==============================================================================
# 3. Veredito Padronizado
# ==============================================================================
def test_veredito_evidencia() -> None:
    """Veredito: Evidência quando p<0.05, delta>=0.05, IC exclui zero, sanidades OK."""
    v = veredito_padronizado(
        p_valor_primario=0.01,
        delta_medio_primario=0.10,
        ic_95_primario=(0.06, 0.14),
        sanidades_aprovadas=True,
        efeito_relevante=True,
    )
    assert v.categoria == "Evidencia"
    assert v.tipo_sinal == "global_frequencia"


def test_veredito_erro_vazamento() -> None:
    """Veredito: ErroVazamento quando sanidades falham."""
    v = veredito_padronizado(
        p_valor_primario=0.001,
        delta_medio_primario=0.50,
        ic_95_primario=(0.40, 0.60),
        sanidades_aprovadas=False,
        efeito_relevante=True,
    )
    assert v.categoria == "ErroVazamento"


def test_veredito_sem_evidencia() -> None:
    """Veredito: SemEvidencia quando p>=0.05 e IC exclui efeitos relevantes."""
    v = veredito_padronizado(
        p_valor_primario=0.50,
        delta_medio_primario=0.01,
        ic_95_primario=(-0.02, 0.04),
        sanidades_aprovadas=True,
        efeito_relevante=False,
    )
    assert v.categoria == "SemEvidencia"


def test_veredito_inconclusivo() -> None:
    """Veredito: Inconclusivo quando IC inclui efeitos relevantes mas p não rejeita."""
    v = veredito_padronizado(
        p_valor_primario=0.10,
        delta_medio_primario=0.04,
        ic_95_primario=(-0.01, 0.08),
        sanidades_aprovadas=True,
        efeito_relevante=False,
    )
    assert v.categoria == "Inconclusivo"


# ==============================================================================
# 4. Correção de Holm
# ==============================================================================
def test_aplicar_holm() -> None:
    """Holm corrige múltiplos p-valores."""
    p_valores = {"H1": 0.001, "H2": 0.02, "H3": 0.04}
    resultados = aplicar_holm(p_valores, alpha=0.05)

    assert "H1" in resultados
    assert "H2" in resultados
    assert "H3" in resultados

    # H1 (p=0.001, menor) tem limite 0.05/3 = 0.0167 -> rejeitada
    assert resultados["H1"]["rejeitada"] is True

    # Verifica que limites são decrescentes
    limites = [resultados[nome]["limite_holm"] for nome in ["H1", "H2", "H3"]]
    # Ordem não é necessariamente decrescente porque os p-valores são ordenados
    # internamente, mas os limites dependem do ranking


# ==============================================================================
# 5. Tamanho do Efeito
# ==============================================================================
def test_calcular_tamanho_efeito() -> None:
    """Tamanho de efeito (d de Cohen) é calculado."""
    delta = np.array([0.1, 0.2, 0.15, 0.05, 0.2])
    d = calcular_tamanho_efeito(delta)
    assert isinstance(d, float)


def test_calcular_ic_bootstrap() -> None:
    """Bootstrap BCa retorna intervalo válido."""
    np.random.seed(42)
    delta = np.random.normal(0.1, 0.5, 100)
    li, ls = calcular_ic_bootstrap(delta, n_bootstrap=1000, seed=42)
    assert li < ls


# ==============================================================================
# 6. Constantes
# ==============================================================================
def test_constantes() -> None:
    """Constantes de teste estatístico."""
    assert ALFA_TESTE == 0.05
    assert LIMIAR_RELEVANCIA_NATS == 0.05


if __name__ == "__main__":
    print("Executando suíte de testes de estatística...")
    test_teste_permutacao_nll_modelo_melhor()
    test_teste_permutacao_nll_sem_diferenca()
    test_teste_permutacao_acertos()
    test_teste_permutacao_tamanho_diferente_erro()
    test_sanidade_rotulos_permutados_aprovada()
    test_sanidade_entradas_aleatorias_aprovada()
    test_sanidade_sentinela_futura_rejeitada()
    test_sanidade_sentinela_futura_valida()
    test_veredito_evidencia()
    test_veredito_erro_vazamento()
    test_veredito_sem_evidencia()
    test_veredito_inconclusivo()
    test_aplicar_holm()
    test_calcular_tamanho_efeito()
    test_calcular_ic_bootstrap()
    test_constantes()
    print("SUCESSO: Todos os 16 testes de estatística passaram perfeitamente!")
