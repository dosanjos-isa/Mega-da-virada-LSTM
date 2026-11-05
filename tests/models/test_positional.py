"""Testes unitários para o modelo M3 — Produto Posicional normalizado.

Baseado em spec/math_spec.md §4.4:
- P_M3(S) = prod_j P_j(d_j) / Z
- Z calculado por DP em O(k * N)

Cobre: ADR-002 (M3 como ablacao), math_spec.md §4.4, Apêndice B.
P1-03: implementacao do M3.
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

from lstm_virada.models.positional import PositionalModel, PositionalError


# Universo reduzido: N=5, k=2
# supp_1 = {1,2,3,4}, supp_2 = {2,3,4,5}
# P_1 = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
# P_2 = {2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4}
# Z = soma sobre d1<d2 de P_1(d1)*P_2(d2) = 0.65

P1_VALUES = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
P2_VALUES = {2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4}

# Z calculado manualmente:
# 0.1*0.1 + 0.1*0.2 + 0.1*0.3 + 0.1*0.4  = 0.10
# 0.2*0.2 + 0.2*0.3 + 0.2*0.4            = 0.18
# 0.3*0.3 + 0.3*0.4                      = 0.21
# 0.4*0.4                                = 0.16
# Total = 0.65
Z_ESPERADO = 0.65

# Probabilidades esperadas para cada conjunto
M3_VETORES_TESTE = [
    ((1, 2), 0.1 * 0.1 / Z_ESPERADO),   # ≈ 0,015385
    ((1, 3), 0.1 * 0.2 / Z_ESPERADO),   # ≈ 0,030769
    ((1, 4), 0.1 * 0.3 / Z_ESPERADO),   # ≈ 0,046154
    ((1, 5), 0.1 * 0.4 / Z_ESPERADO),   # ≈ 0,061538
    ((2, 3), 0.2 * 0.2 / Z_ESPERADO),   # ≈ 0,061538
    ((2, 4), 0.2 * 0.3 / Z_ESPERADO),   # ≈ 0,092308
    ((2, 5), 0.2 * 0.4 / Z_ESPERADO),   # ≈ 0,123077
    ((3, 4), 0.3 * 0.3 / Z_ESPERADO),   # ≈ 0,138462
    ((3, 5), 0.3 * 0.4 / Z_ESPERADO),   # ≈ 0,184615
    ((4, 5), 0.4 * 0.4 / Z_ESPERADO),   # ≈ 0,246154
]


def _criar_modelo_posicional(p1: dict[int, float], p2: dict[int, float]) -> PositionalModel:
    """Cria um M3 com distribuciones positionais dadas."""
    modelo = PositionalModel(num_classes=5, num_dezenas_jogo=2)
    modelo.ajustar_distribuicoes([p1, p2])
    return modelo


# ==============================================================================
# 1. Normalização e vetores de teste
# ==============================================================================
def test_m3_normalizacao_universo_reduzido() -> None:
    """M3: soma das probabilidades de todos os conjuntos deve ser 1."""
    modelo = _criar_modelo_posicional(P1_VALUES, P2_VALUES)

    dezenas = [1, 2, 3, 4, 5]
    soma = 0.0
    for i in range(len(dezenas)):
        for j in range(i + 1, len(dezenas)):
            S = (dezenas[i], dezenas[j])
            soma += modelo.prob_conjunto(S)

    assert np.isclose(soma, 1.0, atol=1e-10), f"Soma = {soma}, esperado 1.0"


def test_m3_vetores_teste() -> None:
    """M3: cada conjunto confere com o valor esperado."""
    modelo = _criar_modelo_posicional(P1_VALUES, P2_VALUES)

    for S, esperado in M3_VETORES_TESTE:
        obtido = modelo.prob_conjunto(S)
        assert np.isclose(obtido, esperado, atol=1e-10), (
            f"Conjunto {S}: esperado {esperado}, obtido {obtido}"
        )


def test_m3_z_calculado_por_dp() -> None:
    """M3: Z deve ser 0.65 para as distribuciones de teste."""
    modelo = _criar_modelo_posicional(P1_VALUES, P2_VALUES)
    z = modelo.z()
    assert np.isclose(z, Z_ESPERADO, atol=1e-10), f"Z = {z}, esperado {Z_ESPERADO}"


def test_m3_modo_universo_reduzido() -> None:
    """M3: o modo eh o conjunto de maior probabilidade."""
    modelo = _criar_modelo_posicional(P1_VALUES, P2_VALUES)

    modo = modelo.modo()
    # O conjunto de maior probabilidade eh (4,5) com p = 0.16/0.65 ≈ 0.246
    assert modo == (4, 5), f"Modo esperado (4, 5), obtido {modo}"


def test_m3_caso_uniforme() -> None:
    """M3: com distribuicoes uniformes nos suportes, P(S) = 1/C(N,k)."""
    # Para N=5, k=2:
    # supp_1 = {1,2,3,4} -> P_1(c) = 1/4 para cada
    # supp_2 = {2,3,4,5} -> P_2(c) = 1/4 para cada
    p1 = {c: 1 / 4 for c in range(1, 5)}
    p2 = {c: 1 / 4 for c in range(2, 6)}
    modelo = _criar_modelo_posicional(p1, p2)

    dezenas = [1, 2, 3, 4, 5]
    for i in range(len(dezenas)):
        for j in range(i + 1, len(dezenas)):
            S = (dezenas[i], dezenas[j])
            p = modelo.prob_conjunto(S)
            assert np.isclose(p, 1 / 10, atol=1e-10), (
                f"Caso uniforme: {S} deveria ter p=0.1, obtido {p}"
            )


# ==============================================================================
# 2. Log-verossimilhança
# ==============================================================================
def test_m3_log_prob_conjunto() -> None:
    """M3: log_prob_conjunto deve ser log de prob_conjunto."""
    modelo = _criar_modelo_posicional(P1_VALUES, P2_VALUES)

    S = (4, 5)
    p = modelo.prob_conjunto(S)
    lp = modelo.log_prob_conjunto(S)
    assert np.isclose(lp, np.log(p), atol=1e-10)


def test_m3_log_verossimilhança_conjunto_dados() -> None:
    """M3: log-verossimilhança de um conjunto de dados."""
    modelo = _criar_modelo_posicional(P1_VALUES, P2_VALUES)

    dados = [(1, 2), (3, 4), (4, 5)]
    ll = modelo.log_verossimilhança(dados)
    esperado = sum(modelo.log_prob_conjunto(S) for S in dados)
    assert np.isclose(ll, esperado, atol=1e-10)


# ==============================================================================
# 3. Validações de entrada
# ==============================================================================
def test_m3_distribuicoes_devem_somar_um() -> None:
    """M3: cada distribuicao posicional deve somar 1."""
    modelo = PositionalModel(num_classes=5, num_dezenas_jogo=2)

    # P_1 nao soma 1
    p1_errado = {1: 0.5, 2: 0.3, 3: 0.1, 4: 0.05}  # soma 0.95
    p2_ok = {2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4}

    with pytest.raises(PositionalError, match="soma"):
        modelo.ajustar_distribuicoes([p1_errado, p2_ok])


def test_m3_dezenas_fora_do_intervalo() -> None:
    """M3: dezenas devem estar em [1, N]."""
    modelo = _criar_modelo_posicional(P1_VALUES, P2_VALUES)

    with pytest.raises(PositionalError, match=r"\["):
        modelo.prob_conjunto((1, 6))


def test_m3_dezenas_nao_distintas() -> None:
    """M3: dezenas devem ser distintas."""
    modelo = _criar_modelo_posicional(P1_VALUES, P2_VALUES)

    with pytest.raises(PositionalError, match="distint"):
        modelo.prob_conjunto((1, 1))


def test_m3_numero_errado_de_distribuicoes() -> None:
    """M3: deve haver exatamente k distribuicoes positionais."""
    modelo = PositionalModel(num_classes=5, num_dezenas_jogo=2)

    p1 = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    with pytest.raises(PositionalError, match="2"):
        modelo.ajustar_distribuicoes([p1])  # apenas 1 distribuicao, esperado 2


if __name__ == "__main__":
    print("Executando suíte de testes do M3 (Produto Posicional)...")
    test_m3_normalizacao_universo_reduzido()
    test_m3_vetores_teste()
    test_m3_z_calculado_por_dp()
    test_m3_modo_universo_reduzido()
    test_m3_caso_uniforme()
    test_m3_log_prob_conjunto()
    test_m3_log_verossimilhança_conjunto_dados()
    test_m3_distribuicoes_devem_somar_um()
    test_m3_dezenas_fora_do_intervalo()
    test_m3_dezenas_nao_distintas()
    test_m3_numero_errado_de_distribuicoes()
    print("SUCESSO: Todos os 11 testes do M3 passaram perfeitamente!")
