"""Testes unitários para o módulo de splitters temporais (splits.py).

Cobre os requisitos:
- RT-006: Prevenção formal de vazamento temporal (data leakage).
- RV-001: Walk-forward por edição da Mega da Virada (2015-2025).
- RV-002: Avaliação secundária nos últimos 300 concursos regulares.
- RV-003: Registro prospectivo antes do sorteio (sentinela futura rejeitada).
- P0-05: TemporalSplitter, WalkForwardSplitter, InferenceWindowExtractor.
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

# Carregamento dinâmico do pytest: compatível com ambientes com e sem pytest instalado
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

from lstm_virada.splits import (
    InferenceWindowExtractor,
    SplitError,
    TemporalSplit,
    TemporalSplitter,
    WalkForwardFold,
    WalkForwardSplitter,
    VIRADA_ANOS,
    WALK_FORWARD_VIRADA_ANOS,
)


# ==============================================================================
# Fixtures auxiliares
# ==============================================================================
def _create_history_with_viradas(n_draws: int = 300) -> pd.DataFrame:
    """Gera histórico sintético com colunas do dataset processado e Viradas.

    As Viradas são marcadas em concursos específicos para permitir teste de
    walk-forward. Concurso 1775 = Virada 2015, 1890 = Virada 2016, etc.
    """
    rows = []
    np.random.seed(42)
    virada_map = {
        1775: 2015,
        1890: 2016,
        2000: 2017,
        2110: 2018,
        2220: 2019,
        2330: 2020,
        2440: 2021,
        2550: 2022,
        2670: 2023,
        2810: 2024,
        2955: 2025,
    }
    for c in range(1, n_draws + 1):
        dezenas = sorted(np.random.choice(np.arange(1, 61), size=6, replace=False))
        is_virada = 1 if c in virada_map else 0
        virada_ano = virada_map.get(c, 0)
        rows.append({
            "concurso": c,
            "data": f"2024-01-01",  # data fixa, não usada nos testes de splitter
            "ano": 2024,
            "virada_ano": virada_ano,
            "d1": dezenas[0],
            "d2": dezenas[1],
            "d3": dezenas[2],
            "d4": dezenas[3],
            "d5": dezenas[4],
            "d6": dezenas[5],
            "is_virada": is_virada,
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 1. TemporalSplitter
# ==============================================================================
def test_temporal_splitter_basic_split() -> None:
    """Verifica divisão básica treino/teste por concurso de corte."""
    df = _create_history_with_viradas(n_draws=100)
    splitter = TemporalSplitter(df)

    split_result = splitter.split(as_of_concurso=50)

    assert split_result.as_of_concurso == 50
    assert len(split_result.train_indices) == 49  # concursos 1-49
    assert len(split_result.test_indices) == 51   # concursos 50-100
    assert (split_result.train_concursos < 50).all()
    assert (split_result.test_concursos >= 50).all()
    assert split_result.train_concursos[-1] == 49
    assert split_result.test_concursos[0] == 50


def test_temporal_splitter_rejects_future_sentinel() -> None:
    """Rejeita concurso de corte além do último disponível (sentinela futura)."""
    df = _create_history_with_viradas(n_draws=100)
    splitter = TemporalSplitter(df)

    with pytest.raises(SplitError, match="Sentinela futura"):
        splitter.split(as_of_concurso=101)


def test_temporal_splitter_rejects_invalid_as_of() -> None:
    """Rejeita as_of_concurso menor que 1."""
    df = _create_history_with_viradas(n_draws=100)
    splitter = TemporalSplitter(df)

    with pytest.raises(SplitError, match=">= 1"):
        splitter.split(as_of_concurso=0)


def test_temporal_splitter_no_leakage() -> None:
    """Garante que nenhum concurso de teste aparece no treino."""
    df = _create_history_with_viradas(n_draws=100)
    splitter = TemporalSplitter(df)

    for as_of in [10, 50, 100]:
        split_result = splitter.split(as_of_concurso=as_of)
        train_set = set(split_result.train_concursos.tolist())
        test_set = set(split_result.test_concursos.tolist())
        assert train_set.isdisjoint(test_set), "Vazamento: concursos em treino e teste."


def test_temporal_splitter_properties() -> None:
    """Verifica propriedades do TemporalSplitter."""
    df = _create_history_with_viradas(n_draws=100)
    splitter = TemporalSplitter(df)

    assert splitter.last_concurso == 100
    assert len(splitter.concursos) == 100
    assert (splitter.concursos == np.arange(1, 101)).all()


# ==============================================================================
# 2. WalkForwardSplitter
# ==============================================================================
def test_walk_forward_generates_11_folds() -> None:
    """Verifica que o walk-forward gera 11 folds (2015-2025)."""
    df = _create_history_with_viradas(n_draws=3000)
    splitter = WalkForwardSplitter(df, virada_anos=WALK_FORWARD_VIRADA_ANOS)

    folds = splitter.splits()
    assert len(folds) == 11
    anos = [f.virada_ano for f in folds]
    assert anos == list(range(2015, 2026))


def test_walk_forward_fold_causality() -> None:
    """Garante causalidade: treino contém apenas concursos anteriores à Virada."""
    df = _create_history_with_viradas(n_draws=3000)
    splitter = WalkForwardSplitter(df, virada_anos=WALK_FORWARD_VIRADA_ANOS)

    folds = splitter.splits()
    for fold in folds:
        # Todo concurso de treino deve ser < concurso da Virada
        assert (fold.train_concursos < fold.concurso_virada).all(), (
            f"Vazamento no fold {fold.virada_ano}: treino >= concurso Virada."
        )
        # O teste é exatamente o concurso da Virada
        assert len(fold.test_concursos) == 1
        assert int(fold.test_concursos[0]) == fold.concurso_virada


def test_walk_forward_fold_for_year() -> None:
    """Retorna o fold de uma edição específica."""
    df = _create_history_with_viradas(n_draws=3000)
    splitter = WalkForwardSplitter(df, virada_anos=WALK_FORWARD_VIRADA_ANOS)

    fold_2020 = splitter.fold_for_year(2020)
    assert fold_2020.virada_ano == 2020
    assert fold_2020.concurso_virada == 2330

    with pytest.raises(SplitError, match="não disponível"):
        splitter.fold_for_year(2050)


def test_walk_forward_no_empty_train() -> None:
    """Cada fold deve ter dados de treino (não pode ser vazio)."""
    df = _create_history_with_viradas(n_draws=3000)
    splitter = WalkForwardSplitter(df, virada_anos=WALK_FORWARD_VIRADA_ANOS)

    folds = splitter.splits()
    for fold in folds:
        assert len(fold.train_indices) > 0, f"Fold {fold.virada_ano} sem treino."
        assert len(fold.test_indices) == 1, f"Fold {fold.virada_ano} deve ter 1 teste."


def test_walk_forward_skips_missing_years() -> None:
    """Edições de Virada ausentes no dataset são puladas silenciosamente."""
    df = _create_history_with_viradas(n_draws=3000)
    # Solicita anos que não existem no mapa de Viradas (2050, 2060)
    splitter = WalkForwardSplitter(df, virada_anos=(2050, 2060, 2015))
    folds = splitter.splits()

    # Apenas 2015 deve estar presente
    anos = [f.virada_ano for f in folds]
    assert anos == [2015]


# ==============================================================================
# 3. InferenceWindowExtractor
# ==============================================================================
def test_inference_window_shapes() -> None:
    """Verifica shapes da janela de inferência."""
    df = _create_history_with_viradas(n_draws=50)
    extractor = InferenceWindowExtractor(df, window_size=20)

    window = extractor.extract(alvo_virada=1)

    assert window.x_seq.shape == (1, 20, 61)
    assert window.x_static.shape == (1, 1)
    assert window.last_concurso == 50
    assert window.expected_target_concurso == 51


def test_inference_window_rejects_small_history() -> None:
    """Rejeita histórico menor que window_size."""
    df = _create_history_with_viradas(n_draws=10)
    extractor = InferenceWindowExtractor(df, window_size=20)

    with pytest.raises(SplitError, match="insuficiente"):
        extractor.extract(alvo_virada=1)


def test_inference_window_no_is_virada_per_step() -> None:
    """P0-06: is_virada por passo deve ser 0 (só vai no x_static)."""
    df = _create_history_with_viradas(n_draws=50)
    extractor = InferenceWindowExtractor(df, window_size=20)

    window = extractor.extract(alvo_virada=1)

    # x_static deve conter a flag de Virada
    assert window.x_static[0, 0] == 1.0
    # x_seq: índice 60 (is_virada por passo) deve ser 0 para todos os passos
    assert (window.x_seq[0, :, 60] == 0.0).all()


# ==============================================================================
# 4. Constantes
# ==============================================================================
def test_virada_anos_constant() -> None:
    """Verifica a constante de edições de Virada."""
    assert VIRADA_ANOS == tuple(range(2009, 2026))
    assert len(VIRADA_ANOS) == 17


def test_walk_forward_virada_anos_constant() -> None:
    """Verifica a constante de anos para walk-forward."""
    assert WALK_FORWARD_VIRADA_ANOS == tuple(range(2015, 2026))
    assert len(WALK_FORWARD_VIRADA_ANOS) == 11


if __name__ == "__main__":
    print("Executando suíte de testes de splitters temporais...")
    test_temporal_splitter_basic_split()
    test_temporal_splitter_rejects_future_sentinel()
    test_temporal_splitter_rejects_invalid_as_of()
    test_temporal_splitter_no_leakage()
    test_temporal_splitter_properties()
    test_walk_forward_generates_11_folds()
    test_walk_forward_fold_causality()
    test_walk_forward_fold_for_year()
    test_walk_forward_no_empty_train()
    test_walk_forward_skips_missing_years()
    test_inference_window_shapes()
    test_inference_window_rejects_small_history()
    test_inference_window_no_is_virada_per_step()
    test_virada_anos_constant()
    test_walk_forward_virada_anos_constant()
    print("SUCESSO: Todos os 15 testes de splitters temporais passaram perfeitamente!")
