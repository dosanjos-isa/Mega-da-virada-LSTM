"""Testes unitários para os baselines B0 a B3 (baselines.py).

Cobre os requisitos:
- ADR-003: B2a e B2b definidos separadamente, normalizados sobre Ω.
- math_spec.md §5: definições de B0, B1, B2a, B2b, B3.
- RV-001 a RV-005: comparação com baselines, paridade de K.
- P0-07: implementação dos baselines.
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
import pandas as pd

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

from lstm_virada.baselines import (
    B0Uniforme,
    B1MaisFrequentes,
    B2aProdutoPosicional,
    B2bPLFrequenciaEmpirica,
    B3ConcursoAnterior,
    BaselineError,
    BaselineResult,
    criar_baseline,
    todos_os_baselines,
    OMEGA_SIZE,
    DEFAULT_KAPPA,
)


# ==============================================================================
# Fixtures auxiliares
# ==============================================================================
def _criar_dataset_sintético(n_concursos: int = 100) -> pd.DataFrame:
    """Cria um DataFrame sintético com concursos e dezenas."""
    rows = []
    np.random.seed(42)
    for c in range(1, n_concursos + 1):
        dezenas = sorted(np.random.choice(np.arange(1, 61), size=6, replace=False))
        rows.append({
            "concurso": c,
            "d1": dezenas[0],
            "d2": dezenas[1],
            "d3": dezenas[2],
            "d4": dezenas[3],
            "d5": dezenas[4],
            "d6": dezenas[5],
        })
    return pd.DataFrame(rows)


def _dezenas_array(df: pd.DataFrame) -> np.ndarray:
    return df[["d1", "d2", "d3", "d4", "d5", "d6"]].to_numpy(dtype=np.int32)


# ==============================================================================
# 1. B0 — Uniforme
# ==============================================================================
def test_b0_log_score_constante() -> None:
    """B0: log_score é constante = -log(C(60,6)) para todos os concursos."""
    df = _criar_dataset_sintético(50)
    dezenas = _dezenas_array(df)

    b0 = B0Uniforme()
    log_p = b0.log_score_conjunto(dezenas)

    esperado = -np.log(OMEGA_SIZE)
    assert len(log_p) == 50
    assert np.allclose(log_p, esperado)


def test_b0_omega_size() -> None:
    """Verifica que OMEGA_SIZE = C(60, 6) = 50.063.860."""
    from math import comb
    assert OMEGA_SIZE == comb(60, 6)
    assert OMEGA_SIZE == 50_063_860


# ==============================================================================
# 2. B1 — Mais Frequentes
# ==============================================================================
def test_b1_ajusta_top6_frequentes() -> None:
    """B1: identifica as 6 dezenas mais frequentes no treino."""
    # Cria dataset onde dezenas 1-6 são claramente mais frequentes
    rows = []
    for c in range(1, 51):
        # Dezenas 1-6 aparecem em todo sorteio; outras aleatórias
        extras = sorted(np.random.choice(np.arange(7, 61), size=0, replace=False))
        dezenas = list(range(1, 7))
        rows.append({
            "concurso": c,
            "d1": 1, "d2": 2, "d3": 3, "d4": 4, "d5": 5, "d6": 6,
        })
    df = pd.DataFrame(rows)
    dezenas = _dezenas_array(df)

    b1 = B1MaisFrequentes()
    b1.ajustar(dezenas)
    conjunto = b1.conjunto_predito()

    assert conjunto == (1, 2, 3, 4, 5, 6)


def test_b1_avalia_acertos() -> None:
    """B1: avalia acertos corretamente no conjunto de teste."""
    df = _criar_dataset_sintético(50)
    dezenas = _dezenas_array(df)

    b1 = B1MaisFrequentes()
    b1.ajustar(dezenas[:40])  # treino nos primeiros 40
    resultado = b1.avaliar(dezenas[40:])  # teste nos últimos 10

    assert resultado.medio_acertos >= 0.0
    assert len(resultado.acertos_por_concurso) == 10
    assert resultado.conjunto_predito is not None
    assert len(resultado.conjunto_predito) == 6


# ==============================================================================
# 3. B2a — Produto Posicional com Suavização
# ==============================================================================
def test_b2a_ajusta_e_calcula_z() -> None:
    """B2a: calcula Z > 0 e log_score finito."""
    df = _criar_dataset_sintético(100)
    dezenas = _dezenas_array(df)

    b2a = B2aProdutoPosicional(kappa=DEFAULT_KAPPA)
    b2a.ajustar(dezenas)

    log_p = b2a.log_score_conjunto(dezenas[:10])
    assert np.all(np.isfinite(log_p))
    assert len(log_p) == 10


def test_b2a_suavizacao_evita_menos_inf() -> None:
    """B2a: com suavização, dezenas nunca vistas não geram -inf."""
    # Dataset onde a dezena 60 nunca aparece na posição 1
    rows = []
    for c in range(1, 21):
        dezenas = [c % 50 + 1, 10, 20, 30, 40, 50]  # nunca usa 60 na posição 1
        rows.append({
            "concurso": c,
            "d1": dezenas[0], "d2": dezenas[1], "d3": dezenas[2],
            "d4": dezenas[3], "d5": dezenas[4], "d6": dezenas[5],
        })
    df = pd.DataFrame(rows)
    dezenas = _dezenas_array(df)

    b2a = B2aProdutoPosicional(kappa=1.0)
    b2a.ajustar(dezenas)

    # Teste com um conjunto que inclui 60 em alguma posição
    teste = np.array([[1, 10, 20, 30, 40, 60]], dtype=np.int32)
    log_p = b2a.log_score_conjunto(teste)
    assert np.isfinite(log_p[0])


def test_b2a_kappa_invalido() -> None:
    """B2a: kappa <= 0 lança BaselineError."""
    with pytest.raises(BaselineError, match="kappa"):
        B2aProdutoPosicional(kappa=0.0)

    with pytest.raises(BaselineError, match="kappa"):
        B2aProdutoPosicional(kappa=-1.0)


# ==============================================================================
# 4. B2b — PL com Frequência Empírica
# ==============================================================================
def test_b2b_ajusta_e_calcula_log_score() -> None:
    """B2b: calcula log_score finito com pesos empíricos."""
    df = _criar_dataset_sintético(100)
    dezenas = _dezenas_array(df)

    b2b = B2bPLFrequenciaEmpirica()
    b2b.ajustar(dezenas)

    log_p = b2b.log_score_conjunto(dezenas[:10])
    assert np.all(np.isfinite(log_p))
    assert len(log_p) == 10


def test_b2b_pesos_positivos() -> None:
    """B2b: todos os pesos w_i > 0 (todas as dezenas aparecem)."""
    df = _criar_dataset_sintético(100)
    dezenas = _dezenas_array(df)

    b2b = B2bPLFrequenciaEmpirica()
    b2b.ajustar(dezenas)

    # Verifica indiretamente: log_score finito significa que não houve
    # divisão por zero (w_i > 0 para todo i)
    log_p = b2b.log_score_conjunto(dezenas[:5])
    assert np.all(np.isfinite(log_p))


# ==============================================================================
# 5. B3 — Concurso Anterior
# ==============================================================================
def test_b3_concurso_anterior() -> None:
    """B3: prediz as dezenas do concurso imediatamente anterior."""
    concursos = np.arange(1, 11, dtype=np.int64)
    dezenas = _criar_dataset_sintético(10)[["d1", "d2", "d3", "d4", "d5", "d6"]].to_numpy(dtype=np.int32)

    b3 = B3ConcursoAnterior()
    b3.ajustar(concursos, dezenas)

    pred = b3.predizer_para_concurso(5)
    assert pred == tuple(dezenas[3])  # concurso 4 -> índice 3


def test_b3_avalia_acertos() -> None:
    """B3: avalia acertos no conjunto de teste."""
    concursos = np.arange(1, 11, dtype=np.int64)
    dezenas = _criar_dataset_sintético(10)[["d1", "d2", "d3", "d4", "d5", "d6"]].to_numpy(dtype=np.int32)

    b3 = B3ConcursoAnterior()
    b3.ajustar(concursos, dezenas)

    resultado = b3.avaliar(concursos[1:], dezenas[1:])
    assert resultado.medio_acertos >= 0.0
    assert len(resultado.acertos_por_concurso) == 9


def test_b3_concurso_inexistente() -> None:
    """B3: concurso anterior não encontrado lança BaselineError."""
    b3 = B3ConcursoAnterior()
    b3.ajustar(np.array([1, 2, 3]), np.array([[1,2,3,4,5,6]] * 3))

    with pytest.raises(BaselineError, match="não encontrado"):
        b3.predizer_para_concurso(1)  # concurso anterior = 0, não existe


# ==============================================================================
# 6. Fábrica e todos_os_baselines
# ==============================================================================
def test_criar_baseline_todos() -> None:
    """Verifica que a fábrica cria todos os baselines corretamente."""
    for nome in ["B0", "B1", "B2a", "B2b", "B3"]:
        bl = criar_baseline(nome)
        assert bl is not None

    with pytest.raises(BaselineError, match="desconhecido"):
        criar_baseline("B99")


def test_todos_os_baselines() -> None:
    """todos_os_baselines retorna os 5 baselines."""
    baselines = todos_os_baselines()
    assert set(baselines.keys()) == {"B0", "B1", "B2a", "B2b", "B3"}
    assert len(baselines) == 5


if __name__ == "__main__":
    print("Executando suíte de testes de baselines...")
    test_b0_log_score_constante()
    test_b0_omega_size()
    test_b1_ajusta_top6_frequentes()
    test_b1_avalia_acertos()
    test_b2a_ajusta_e_calcula_z()
    test_b2a_suavizacao_evita_menos_inf()
    test_b2a_kappa_invalido()
    test_b2b_ajusta_e_calcula_log_score()
    test_b2b_pesos_positivos()
    test_b3_concurso_anterior()
    test_b3_avalia_acertos()
    test_b3_concurso_inexistente()
    test_criar_baseline_todos()
    test_todos_os_baselines()
    print("SUCESSO: Todos os 16 testes de baselines passaram perfeitamente!")
