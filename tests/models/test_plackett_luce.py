"""Testes unitários para o modelo M1 — Plackett–Luce marginalizado.

Baseado nos vetores de teste do Apêndice B da spec/math_spec.md:
- Universo reduzido: N=5, k=2, w=(1,2,3,4,5), W=15
- Caso uniforme: w=(1,1,1,1,1)

Cobre: ADR-002 (modelo primário M1), math_spec.md §4.2, Apêndice B.
P1-01: implementação do M1.
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

from lstm_virada.models.plackett_luce import PlackettLuce, PlackettLuceError


# Vetores de teste do Apêndice B.2 — universo reduzido N=5, k=2, w=(1,2,3,4,5)
# soma deve ser 1
M1_VETORES_TESTE = [
    ((1, 2), 9 / 455),       # ≈ 0,019780
    ((1, 3), 13 / 420),      # ≈ 0,030952
    ((1, 4), 10 / 231),      # ≈ 0,043290
    ((1, 5), 2 / 35),        # ≈ 0,057143
    ((2, 3), 5 / 78),        # ≈ 0,064103
    ((2, 4), 64 / 715),      # ≈ 0,089510
    ((2, 5), 23 / 195),      # ≈ 0,117949
    ((3, 4), 23 / 165),      # ≈ 0,139394
    ((3, 5), 11 / 60),       # ≈ 0,183333
    ((4, 5), 14 / 55),       # ≈ 0,254545
]


def _criar_modelo_pesos(pesos: dict[int, float]) -> PlackettLuce:
    """Cria um M1 com pesos dados (dezena -> peso)."""
    modelo = PlackettLuce(num_classes=5, num_dezenas_jogo=2)
    modelo.ajustar_pesos(pesos)
    return modelo


# ==============================================================================
# 1. Normalização e vetores de teste (Apêndice B)
# ==============================================================================
def test_m1_normalizacao_universo_reduzido() -> None:
    """M1: soma das probabilidades de todos os conjuntos deve ser 1 (universo reduzido)."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    dezenas = [1, 2, 3, 4, 5]
    soma = 0.0
    for i in range(len(dezenas)):
        for j in range(i + 1, len(dezenas)):
            S = (dezenas[i], dezenas[j])
            soma += modelo.prob_conjunto(S)

    assert np.isclose(soma, 1.0, atol=1e-10), f"Soma das probabilidades = {soma}, esperado 1.0"


def test_m1_vetores_teste_apendice_b() -> None:
    """M1: cada conjunto deve conferir com o vetor de teste do Apêndice B."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    for S, esperado in M1_VETORES_TESTE:
        obtido = modelo.prob_conjunto(S)
        assert np.isclose(obtido, esperado, atol=1e-10), (
            f"Conjunto {S}: esperado {esperado}, obtido {obtido}"
        )


def test_m1_modo_universo_reduzido() -> None:
    """M1: o modo do universo reduzido N=5, k=2 deve ser {4, 5}."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    modo = modelo.modo()
    assert modo == (4, 5), f"Modo esperado (4, 5), obtido {modo}"


def test_m1_caso_uniforme() -> None:
    """M1: com pesos iguais, deve coincidir com o uniforme M0."""
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
def test_m1_log_prob_conjunto() -> None:
    """M1: log_prob_conjunto deve ser log de prob_conjunto."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    S = (4, 5)
    p = modelo.prob_conjunto(S)
    lp = modelo.log_prob_conjunto(S)
    assert np.isclose(lp, np.log(p), atol=1e-10)


def test_m1_log_verossimilhança_conjunto_dados() -> None:
    """M1: log-verossimilhança de um conjunto de dados é soma dos log-probs."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    dados = [(1, 2), (3, 4), (4, 5)]
    ll = modelo.log_verossimilhança(dados)
    esperado = sum(modelo.log_prob_conjunto(S) for S in dados)
    assert np.isclose(ll, esperado, atol=1e-10)


# ==============================================================================
# 3. Validações de entrada
# ==============================================================================
def test_m1_pesos_devem_ser_positivos() -> None:
    """M1: pesos devem ser estritamente positivos."""
    modelo = _criar_modelo_pesos({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    with pytest.raises(PlackettLuceError, match="positivo"):
        modelo.ajustar_pesos({1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})

    with pytest.raises(PlackettLuceError, match="positivo"):
        modelo.ajustar_pesos({1: -1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})


def test_m1_dezenas_fora_do_intervalo() -> None:
    """M1: dezenas devem estar em [1, N]."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    with pytest.raises(PlackettLuceError, match=r"\["):
        modelo.prob_conjunto((1, 6))  # 6 fora de [1, 5]


def test_m1_dezenas_nao_distintas() -> None:
    """M1: dezenas devem ser distintas."""
    pesos = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    modelo = _criar_modelo_pesos(pesos)

    with pytest.raises(PlackettLuceError, match="distint"):
        modelo.prob_conjunto((1, 1))


if __name__ == "__main__":
    print("Executando suíte de testes do M1 (Plackett–Luce)...")
    test_m1_normalizacao_universo_reduzido()
    test_m1_vetores_teste_apendice_b()
    test_m1_modo_universo_reduzido()
    test_m1_caso_uniforme()
    test_m1_log_prob_conjunto()
    test_m1_log_verossimilhança_conjunto_dados()
    test_m1_pesos_devem_ser_positivos()
    test_m1_dezenas_fora_do_intervalo()
    test_m1_dezenas_nao_distintas()
    print("SUCESSO: Todos os 9 testes do M1 passaram perfeitamente!")
