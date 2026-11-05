"""Testes unitários para o modelo M4 — Plackett–Luce Neural Sequencial.

Baseado em spec/math_spec.md §4.5:
- Rede condicionada no histórico e no prefixo de dezenas já escolhidas
- Produz pesos w_{j,i} para as dezenas restantes a cada passo j
- Verossimilhança sequencial: P_ord(pi|historico) = prod_j w_{j,pi_j} / (soma_{i nao em prefixo} w_{j,i})
- Probabilidade do conjunto: soma sobre as 720 permutacoes

Cobre: ADR-002 (M4 como ablacao opcional), math_spec.md §4.5.
P1-04: implementacao do M4.
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

# Tenta importar TensorFlow; se nao estiver disponível, os testes que exigem
# Keras serao pulados.
try:
    import tensorflow as tf
    TF_DISPONIVEL = True
except ImportError:
    TF_DISPONIVEL = False

if TF_DISPONIVEL:
    from lstm_virada.models.seq_pl import SeqPLModel, SeqPLError
else:
    SeqPLModel = None  # type: ignore
    SeqPLError = Exception  # type: ignore


# ==============================================================================
# Testes que nao exigem TensorFlow (estrutura e likelihood sequencial)
# ==============================================================================
def test_m4_importacao() -> None:
    """M4: se TF nao estiver disponível, o teste é pulado."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow nao instalado")
    assert SeqPLModel is not None


def test_m4_instancia_modelo() -> None:
    """M4: o modelo pode ser instanciado com parametros padrao."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow nao instalado")
    modelo = SeqPLModel(num_classes=60, num_dezenas_jogo=6)
    assert modelo is not None
    assert not modelo.esta_ajustado


def test_m4_likelihood_sequencial_pesos_fixos() -> None:
    """M4: a likelihood sequencial para pesos fixos confere com o calculo manual."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow nao instalado")

    # Universo reduzido: N=5, k=2
    pesos_sequencia = [
        {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0},  # passo 0
        {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0, 5: 1.0},  # passo 1
    ]

    modelo = SeqPLModel(num_classes=5, num_dezenas_jogo=2)
    modelo.definir_pesos_sequencia(pesos_sequencia)

    # Para S = (1, 2):
    # Permutacoes: (1,2) e (2,1)
    # Cada passo tem seu próprio denominador (soma dos pesos restantes):
    # P_ord(1,2) = w_{0,1}/15 * w_{1,2}/(4+3+2+1) = 1/15 * 4/10 = 4/150
    # P_ord(2,1) = w_{0,2}/15 * w_{1,1}/(5+3+2+1) = 2/15 * 5/11 = 10/165
    # P(S) = 4/150 + 10/165

    p_esperado = 4 / 150 + 10 / 165
    p_obtido = modelo.prob_conjunto((1, 2))
    assert np.isclose(p_obtido, p_esperado, atol=1e-10), (
        f"S=(1,2): esperado {p_esperado}, obtido {p_obtido}"
    )

    # Normalizacao: soma sobre todos os pares deve ser 1
    soma = 0.0
    for i in range(1, 6):
        for j in range(i + 1, 6):
            soma += modelo.prob_conjunto((i, j))
    assert np.isclose(soma, 1.0, atol=1e-8), f"Soma = {soma}, esperado 1.0"


def test_m4_log_prob_conjunto() -> None:
    """M4: log_prob_conjunto deve ser log de prob_conjunto."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow nao instalado")

    pesos_sequencia = [
        {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0},
        {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0, 5: 1.0},
    ]
    modelo = SeqPLModel(num_classes=5, num_dezenas_jogo=2)
    modelo.definir_pesos_sequencia(pesos_sequencia)

    S = (1, 2)
    p = modelo.prob_conjunto(S)
    lp = modelo.log_prob_conjunto(S)
    assert np.isclose(lp, np.log(p), atol=1e-10)


# ==============================================================================
# Testes que exigem treino (executados apenas se TF disponível)
# ==============================================================================
def test_m4_treino_pequeno() -> None:
    """M4: treino curto em dados sintéticos reduzidos deve funcionar."""
    if not TF_DISPONIVEL:
        pytest.skip("TensorFlow nao instalado")

    np.random.seed(42)

    # Gera dados sintéticos: 20 amostras de 2 dezenas de N=5
    dados = []
    for _ in range(20):
        S = tuple(sorted(np.random.choice(np.arange(1, 6), size=2, replace=False)))
        dados.append(S)

    modelo = SeqPLModel(num_classes=5, num_dezenas_jogo=2)
    historico = modelo.treinar(
        dados=dados,
        epocas=5,
        taxa_aprendizado=1e-2,
        verbose=0,
    )

    assert modelo.esta_ajustado
    assert "loss" in historico
    assert len(historico["loss"]) > 0
    # A perda deve ser finita (não NaN)
    assert np.all(np.isfinite(historico["loss"]))


if __name__ == "__main__":
    print("Executando suíte de testes do M4 (PL Neural Sequencial)...")
    test_m4_importacao()
    test_m4_instancia_modelo()
    test_m4_likelihood_sequencial_pesos_fixos()
    test_m4_log_prob_conjunto()
    test_m4_treino_pequeno()
    print("SUCESSO: Todos os 5 testes do M4 passaram perfeitamente!")
