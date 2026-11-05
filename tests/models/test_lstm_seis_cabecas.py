"""Testes unitários para o modelo M5 — LSTM de 6 cabeças.

Baseado em spec/math_spec.md §4.6:
- M5-SET: treinada com -sum_j log P_j(d_j) + log Z (likelihood de conjunto)
- M5-CCE: treinada com soma de CCEs (pseudo-likelihood)
- Arquitetura: LSTM(128) -> Dropout(0.2) -> LSTM(64) -> concat -> Dense(128)
  -> 6 x Dense(60, softmax)

Cobre: ADR-002 (M5 como ablacao), math_spec.md §4.6.
P1-05: implementacao do M5.
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
    from lstm_virada.models.lstm_seis_cabecas import (
        LSTMSeisCabecas,
        LSTMSeisCabecasError,
        VARIANTE_SET,
        VARIANTE_CCE,
    )
else:
    LSTMSeisCabecas = None  # type: ignore
    LSTMSeisCabecasError = Exception  # type: ignore
    VARIANTE_SET = "set"
    VARIANTE_CCE = "cce"


# ==============================================================================
# Testes que não exigem TensorFlow (estrutura e variantes)
# ==============================================================================
def test_m5_importacao() -> None:
    """M5: se TF não estiver disponível, o teste é pulado."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")
    assert LSTMSeisCabecas is not None


def test_m5_variantes() -> None:
    """M5: variantes SET e CCE devem existir."""
    assert VARIANTE_SET == "set"
    assert VARIANTE_CCE == "cce"


def test_m5_instancia_set() -> None:
    """M5: M5-SET pode ser instanciada."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")
    modelo = LSTMSeisCabecas(variante=VARIANTE_SET, num_classes=5, num_dezenas_jogo=2)
    assert modelo is not None
    assert modelo.variante == VARIANTE_SET
    assert not modelo.esta_ajustado


def test_m5_instancia_cce() -> None:
    """M5: M5-CCE pode ser instanciada."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")
    modelo = LSTMSeisCabecas(variante=VARIANTE_CCE, num_classes=5, num_dezenas_jogo=2)
    assert modelo is not None
    assert modelo.variante == VARIANTE_CCE


def test_m5_variante_invalida() -> None:
    """M5: variante inválida deve lançar erro."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")
    with pytest.raises(LSTMSeisCabecasError, match="Variante"):
        LSTMSeisCabecas(variante="invalida", num_classes=5, num_dezenas_jogo=2)


# ==============================================================================
# Testes que exigem treino (executados apenas se TF disponível)
# ==============================================================================
def test_m5_treino_set_pequeno() -> None:
    """M5-SET: treino curto em dados sintéticos reduzidos deve funcionar."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")

    np.random.seed(42)

    dados = []
    for _ in range(20):
        S = tuple(sorted(np.random.choice(np.arange(1, 6), size=2, replace=False)))
        dados.append(S)

    modelo = LSTMSeisCabecas(
        variante=VARIANTE_SET, num_classes=5, num_dezenas_jogo=2
    )
    historico = modelo.treinar(
        dados=dados,
        epocas=5,
        taxa_aprendizado=1e-2,
        verbose=0,
    )

    assert modelo.esta_ajustado
    assert "loss" in historico
    assert len(historico["loss"]) > 0
    assert np.all(np.isfinite(historico["loss"]))


def test_m5_treino_cce_pequeno() -> None:
    """M5-CCE: treino curto em dados sintéticos reduzidos deve funcionar."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow não instalado")

    np.random.seed(42)

    dados = []
    for _ in range(20):
        S = tuple(sorted(np.random.choice(np.arange(1, 6), size=2, replace=False)))
        dados.append(S)

    modelo = LSTMSeisCabecas(
        variante=VARIANTE_CCE, num_classes=5, num_dezenas_jogo=2
    )
    historico = modelo.treinar(
        dados=dados,
        epocas=5,
        taxa_aprendizado=1e-2,
        verbose=0,
    )

    assert modelo.esta_ajustado
    assert "loss" in historico
    assert len(historico["loss"]) > 0
    assert np.all(np.isfinite(historico["loss"]))


if __name__ == "__main__":
    print("Executando suíte de testes do M5 (LSTM de 6 cabeças)...")
    test_m5_importacao()
    test_m5_variantes()
    test_m5_instancia_set()
    test_m5_instancia_cce()
    test_m5_variante_invalida()
    test_m5_treino_set_pequeno()
    test_m5_treino_cce_pequeno()
    print("SUCESSO: Todos os 7 testes do M5 passaram perfeitamente!")
