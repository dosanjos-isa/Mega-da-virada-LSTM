"""Testes unitários para o modelo M2 — Fisher (produto condicional).

Baseado nos vetores de teste do Apêndice B.1 da spec/math_spec.md:
- Universo reduzido: N=5, k=2, w=(1,2,3,4,5), W=15
- e_2(w) = 85

Cobre: ADR-002 (M2 como sensibilidade), math_spec.md §4.3, Apêndice B.1.
P1-02: implementação do M2.
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

from lstm_virada.models.fisher import FisherModel, FisherError


# Vetores de teste do Apêndice B.1 — universo reduzido N=5, k=2, w=(1,2,3,4,5)
# e_2(w) = 85
M2_VETORES_TESTE = [
    ((1, 2), 2 / 85),      # ≈ 0,023529
    ((1, 3), 3 / 85),      # ≈ 0,035294
    ((1, 4), 4 / 85),      # ≈ 0,047059
    ((1, 5), 5 / 85),      # ≈ 0,058824
    ((2, 3), 6 / 85),      # ≈ 0,070588
    ((2, 4), 8 / 85),      # ≈ 0,094118
    ((2, 5), 10 / 85),     # ≈ 0,117647
    ((3, 4), 12 / 85),     # ≈ 0,141176
    ((3, 5), 15 / 85),     # ≈ 0,176471
    ((4, 5), 20 / 85),     # ≈ 0,235294
]


def _criar_modelo_pesos(pesos: dict[int, float]) -> FisherModel:
    """Cria um M2 com pesos dados (dezena -> peso)."""
    modelo = FisherModel(num_classes=5, num_dezenas_jogo=2)
    modelo.ajustar_pesos(pesos)
    return modelo


# ==============================================================================
# 1. Normalização e vetores de teste (Apêndice B.1)
# ==============================================================================
def test_m2_normalizacao_universo_reduzido() -> None:
    """M2: soma das probabilidades de todos os conjuntos deve ser 1."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    dezenas = [1, 2, 3, 4, 5]
    soma = 0.0
    for i in range(len(dezenas)):
        for j in range(i + 1, len(dezenas)):
            S = (dezenas[i], dezenas[j])
            soma += modelo.prob_conjunto(S)

    assert np.isclose(soma, 1.0, atol=1e-10), f"Soma = {soma}, esperado 1.0"


def test_m2_vetores_teste_apendice_b1() -> None:
    """M2: cada conjunto confere com o vetor de teste do Apêndice B.1."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    for S, esperado in M2_VETORES_TESTE:
        obtido = modelo.prob_conjunto(S)
        assert np.isclose(obtido, esperado, atol=1e-10), (
            f"Conjunto {S}: esperado {esperado}, obtido {obtido}"
        )


def test_m2_modo_universo_reduzido() -> None:
    """M2: o modo eh o conjunto das 6 (aqui 2) maiores pesos."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    modo = modelo.modo()
    assert modo == (4, 5), f"Modo esperado (4, 5), obtido {modo}"


def test_m2_caso_uniforme() -> None:
    """M2: com pesos iguais, deve coincidir com o uniforme M0."""
    pesos = {i: 1.0 for i in range(1, 6)}
    modelo = _criar_modelo_pesos(pesos)

    dezenas = [1, 2, 3, 4, 5]
    n_conjuntos = 0
    for i in range(len(dezenas)):
        for j in range(i + 1, len(dezenas)):
            S = (dezenas[i], dezenas[j])
            p = modelo.prob_conjunto(S)
            assert np.isclose(p, 1 / 10, atol=1e-10), (
                f"Caso uniforme: {S} deveria ter p=0.1, obtido {p}"
            )
            n_conjuntos += 1
    assert n_conjuntos == 10


# ==============================================================================
# 2. Log-verossimilhança
# ==============================================================================
def test_m2_log_prob_conjunto() -> None:
    """M2: log_prob_conjunto deve ser log de prob_conjunto."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    S = (4, 5)
    p = modelo.prob_conjunto(S)
    lp = modelo.log_prob_conjunto(S)
    assert np.isclose(lp, np.log(p), atol=1e-10)


def test_m2_log_verossimilhança_conjunto_dados() -> None:
    """M2: log-verossimilhança de um conjunto de dados."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    dados = [(1, 2), (3, 4), (4, 5)]
    ll = modelo.log_verossimilhança(dados)
    esperado = sum(modelo.log_prob_conjunto(S) for S in dados)
    assert np.isclose(ll, esperado, atol=1e-10)


# ==============================================================================
# 3. e_6 (polinômio simétrico elementar)
# ==============================================================================
def test_m2_e_k_universo_reduzido() -> None:
    """M2: e_k deve ser calculado corretamente pela DP.

    Para w=(1,2,3,4,5), e_2(w) = 85 antes da normalizacao.
    Depois da normalizacao por media geometrica, e_2 e escalado
    por media_geometricas^(-2).
    """
    import math

    pesos_orig = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos_orig)

    # Media geometrica dos pesos originais
    mg = math.exp(np.mean([math.log(v) for v in pesos_orig.values()]))
    # e_2 normalizado = e_2_original / mg^2
    e2_esperado = 85.0 / (mg ** 2)

    e2 = modelo.e_k()
    assert np.isclose(e2, e2_esperado, atol=1e-8), (
        f"e_2 = {e2}, esperado {e2_esperado}"
    )


# ==============================================================================
# 4. Validações de entrada
# ==============================================================================
def test_m2_pesos_devem_ser_positivos() -> None:
    """M2: pesos devem ser estritamente positivos."""
    modelo = _criar_modelo_pesos({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    with pytest.raises(FisherError, match="posit"):
        modelo.ajustar_pesos({1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})

    with pytest.raises(FisherError, match="posit"):
        modelo.ajustar_pesos({1: -1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})


def test_m2_dezenas_fora_do_intervalo() -> None:
    """M2: dezenas devem estar em [1, N]."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    with pytest.raises(FisherError, match=r"\["):
        modelo.prob_conjunto((1, 6))


def test_m2_dezenas_nao_distintas() -> None:
    """M2: dezenas devem ser distintas."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    with pytest.raises(FisherError, match="distint"):
        modelo.prob_conjunto((1, 1))


if __name__ == "__main__":
    print("Executando suíte de testes do M2 (Fisher)...")
    test_m2_normalizacao_universo_reduzido()
    test_m2_vetores_teste_apendice_b1()
    test_m2_modo_universo_reduzido()
    test_m2_caso_uniforme()
    test_m2_log_prob_conjunto()
    test_m2_log_verossimilhança_conjunto_dados()
    test_m2_e_k_universo_reduzido()
    test_m2_pesos_devem_ser_positivos()
    test_m2_dezenas_fora_do_intervalo()
    test_m2_dezenas_nao_distintas()
    print("SUCESSO: Todos os 10 testes do M2 passaram perfeitamente!")
