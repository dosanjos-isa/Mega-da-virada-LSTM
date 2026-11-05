"""Testes unitários para o treino walk-forward (P1-06).

Baseado em spec/math_spec.md §6 (estimação e treino) e
spec/pre_registration.md §7 (treino e hiperparâmetros).

- max_epochs = 100 (fixo)
- Early stopping: paciência 10 sobre validação sem peso
- 10 seeds para M3/M4/M5, 3 reinicializações para M1/M2
- Walk-forward por edição da Virada (2015-2025)

Cobre: RV-001 (walk-forward), RV-007 (10 sementes), RV-008 (poder),
P1-06: treino walk-forward com 100 épocas.
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

# Tenta importar TensorFlow
try:
    import tensorflow as tf
    TF_DISPONIVEL = True
except ImportError:
    TF_DISPONIVEL = False

if TF_DISPONIVEL:
    from lstm_virada.train_walkforward import (
        treinar_walkforward_modelo,
        TreinoWalkForwardError,
    )
else:
    treinar_walkforward_modelo = None  # type: ignore
    TreinoWalkForwardError = Exception  # type: ignore


# ==============================================================================
# Fixtures auxiliares
# ==============================================================================
def _criar_dataset_pequeno(n_concursos: int = 100) -> dict:
    """Cria um dataset sintético pequeno para teste de treino walk-forward."""
    np.random.seed(42)
    concursos = np.arange(1, n_concursos + 1, dtype=np.int64)
    dezenas = np.array([
        np.random.choice(np.arange(1, 61), size=6, replace=False)
        for _ in range(n_concursos)
    ], dtype=np.int32)
    dezenas = np.sort(dezenas, axis=1)
    is_virada = np.zeros(n_concursos, dtype=np.int32)
    virada_ano = np.zeros(n_concursos, dtype=np.int32)
    # Marcar alguns concursos como Virada com edição correspondente
    virada_mapeamento = {50: 2015, 100: 2016, 150: 2017}
    for vc, ano in virada_mapeamento.items():
        if vc <= n_concursos:
            is_virada[vc - 1] = 1
            virada_ano[vc - 1] = ano

    return {
        "concursos": concursos,
        "dezenas": dezenas,
        "is_virada": is_virada,
        "virada_ano": virada_ano,
    }


# ==============================================================================
# 1. Testes que não exigem TensorFlow (estrutura e validações)
# ==============================================================================
def test_walkforward_importacao() -> None:
    """P1-06: se TF não estiver disponível, o teste é pulado."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")
    assert treinar_walkforward_modelo is not None


def test_walkforward_modelo_invalido() -> None:
    """P1-06: modelo inválido deve lançar erro."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")
    with pytest.raises(TreinoWalkForwardError, match="não suportado"):
        treinar_walkforward_modelo(
            modelo_nome="MODELO_INEXISTENTE",
            dados=_criar_dataset_pequeno(100),
            epocas=5,
        )


# ==============================================================================
# 2. Testes de treino walk-forward por modelo (exigem TF)
# ==============================================================================
def test_walkforward_m1() -> None:
    """P1-06: treino walk-forward M1 (Plackett-Luce estático)."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")

    dados = _criar_dataset_pequeno(200)
    resultado = treinar_walkforward_modelo(
        modelo_nome="M1",
        dados=dados,
        epocas=5,
        seeds=[1, 2, 3],
    )

    assert resultado is not None
    assert hasattr(resultado, "fold_results")
    assert len(resultado.fold_results) > 0
    # Todas as perdas devem ser finitas
    for fold in resultado.fold_results:
        assert np.isfinite(fold.loss), f"Perda não finita: {fold.loss}"


def test_walkforward_m2() -> None:
    """P1-06: treino walk-forward M2 (Fisher)."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")

    dados = _criar_dataset_pequeno(200)
    resultado = treinar_walkforward_modelo(
        modelo_nome="M2",
        dados=dados,
        epocas=5,
        seeds=[1, 2, 3],
    )

    assert resultado is not None
    assert hasattr(resultado, "fold_results")
    assert len(resultado.fold_results) > 0
    for fold in resultado.fold_results:
        assert np.isfinite(fold.loss)


def test_walkforward_m3_set() -> None:
    """P1-06: treino walk-forward M3 (Produto Posicional, variante SET)."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")

    dados = _criar_dataset_pequeno(200)
    resultado = treinar_walkforward_modelo(
        modelo_nome="M3",
        dados=dados,
        epocas=5,
        seeds=[1, 2, 3],
    )

    assert resultado is not None
    assert hasattr(resultado, "fold_results")
    assert len(resultado.fold_results) > 0
    for fold in resultado.fold_results:
        assert np.isfinite(fold.loss)


def test_walkforward_m5_set() -> None:
    """P1-06: treino walk-forward M5-SET (LSTM 6 cabeças, likelihood de conjunto)."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")

    dados = _criar_dataset_pequeno(200)
    resultado = treinar_walkforward_modelo(
        modelo_nome="M5-SET",
        dados=dados,
        epocas=5,
        seeds=[1, 2, 3],
    )

    assert resultado is not None
    assert hasattr(resultado, "fold_results")
    assert len(resultado.fold_results) > 0
    for fold in resultado.fold_results:
        assert np.isfinite(fold.loss)


def test_walkforward_causalidade_respeitada() -> None:
    """P1-06: treino walk-forward deve respeitar causalidade (treino < alvo)."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")

    dados = _criar_dataset_pequeno(200)
    resultado = treinar_walkforward_modelo(
        modelo_nome="M1",
        dados=dados,
        epocas=5,
        seeds=[1, 2, 3],
    )

    # Verificar que cada fold tem treino apenas com concursos anteriores ao alvo
    for fold in resultado.fold_results:
        concurso_alvo = fold.concurso_alvo
        concursos_treino = fold.concursos_treino
        assert all(c < concurso_alvo for c in concursos_treino), (
            f"Vazamento de causalidade: treino tem concurso >= alvo "
            f"({concurso_alvo})"
        )


def test_walkforward_max_epochs() -> None:
    """P1-06: max_epochs deve ser respeitado (100 por padrão, ou menos com early stopping)."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")

    dados = _criar_dataset_pequeno(200)
    resultado = treinar_walkforward_modelo(
        modelo_nome="M3",
        dados=dados,
        epocas=100,  # valor padrão do projeto
        seeds=[1],  # apenas 1 seed para teste rápido
    )

    for fold in resultado.fold_results:
        # O número de épocas executadas deve ser <= max_epochs
        assert fold.epocas_executadas <= 100, (
            f"Epocas excederam o limite: {fold.epocas_executadas}"
        )


def test_walkforward_10_seeds() -> None:
    """P1-06: modelos neurais devem usar 10 seeds."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")

    dados = _criar_dataset_pequeno(200)
    resultado = treinar_walkforward_modelo(
        modelo_nome="M3",
        dados=dados,
        epocas=5,
        seeds=list(range(10)),
    )

    # Verificar que houve 10 sementes executadas
    n_seeds = len(resultado.seeds_usadas)
    assert n_seeds == 10, f"Esperadas 10 seeds, obtidas {n_seeds}"


if __name__ == "__main__":
    print("Executando suíte de testes de treino walk-forward...")
    test_walkforward_importacao()
    test_walkforward_modelo_invalido()
    test_walkforward_m1()
    test_walkforward_m2()
    test_walkforward_m3_set()
    test_walkforward_m5_set()
    test_walkforward_causalidade_respeitada()
    test_walkforward_max_epochs()
    test_walkforward_10_seeds()
    print("SUCESSO: Todos os 9 testes de treino walk-forward passaram perfeitamente!")
