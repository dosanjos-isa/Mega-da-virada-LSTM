"""Teste end-to-end do pipeline completo.

Cobre o fluxo: Dados → Preparação → Treinamento → Avaliação →
Artefato → Inferência → Geração → Manifesto.

Usa universo reduzido (N=5, k=2) para execução rápida e determinística.
Pula se TensorFlow não estiver disponível (treino MLE de M1/M2 requer TF).

Cobre: Objetivo #1-#10, Critério de aceite #14 (teste de integração).
"""

from __future__ import annotations

import importlib
from pathlib import Path
import re
import sys
from typing import Any

# Adiciona src ao sys.path
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
    import tensorflow as tf  # noqa: F401
    TF_DISPONIVEL = True
except ImportError:
    TF_DISPONIVEL = False

if TF_DISPONIVEL:
    from lstm_virada.models.plackett_luce import PlackettLuce
    from lstm_virada.generation import Generator, save_generation, load_generation
else:
    PlackettLuce = None  # type: ignore
    Generator = None  # type: ignore
    save_generation = None  # type: ignore
    load_generation = None  # type: ignore


def _criar_dataset_pequeno() -> list[tuple[int, ...]]:
    """Cria um dataset sintético pequeno (N=60, k=6) para teste rápido."""
    np.random.seed(42)
    dados = []
    for _ in range(30):
        S = tuple(sorted(np.random.choice(np.arange(1, 61), size=6, replace=False)))
        dados.append(S)
    return dados


def _validar_jogo(jogo: list[int], N: int = 60, k: int = 6) -> None:
    """Valida que um jogo tem k dezenas distintas em [1, N], ordenados."""
    assert len(jogo) == k, f"Jogo deve ter {k} dezenas, tem {len(jogo)}: {jogo}"
    assert len(set(jogo)) == k, f"Jogo tem dezenas duplicadas: {jogo}"
    assert all(1 <= d <= N for d in jogo), f"Dezenas fora de [1, {N}]: {jogo}"
    assert jogo == sorted(jogo), f"Jogo não está ordenado: {jogo}"


# ==============================================================================
# Teste end-to-end
# ==============================================================================
@pytest.mark.skipif(not TF_DISPONIVEL, reason="TensorFlow não instalado")
def test_e2e_pipeline_completo(tmp_path: Path) -> None:
    """Executa o pipeline completo: treino → salvamento → carregamento → geração."""
    if PlackettLuce is None or Generator is None:
        pytest.skip("Modelos ou gerador não disponíveis sem TensorFlow.")

    dados = _criar_dataset_pequeno()

    # --- 1. Treinamento (MLE de M1) ---
    modelo = PlackettLuce(num_classes=60, num_dezenas_jogo=6)
    historico = modelo.treinar_mle(
        dados=dados,
        epocas=10,
        taxa_aprendizado=1e-2,
        paciencia=5,
        fracao_validacao=0.10,
        seed=42,
        verbose=0,
    )
    assert modelo.esta_ajustado
    assert "loss" in historico
    assert len(historico["loss"]) > 0

    # --- 2. Avaliação fora da amostra (mesmo dataset, para verificação rápida) ---
    nll = 0.0
    for S in dados:
        nll += -modelo.log_prob_conjunto(tuple(int(d) for d in S))
    nll_medio = nll / len(dados)
    assert np.isfinite(nll_medio), f"NLL não finito: {nll_medio}"

    # --- 3. Persistência (salvar artefato) ---
    modelo_path = tmp_path / "modelo_m1.json"
    modelo.save(str(modelo_path))
    assert modelo_path.exists()

    # --- 4. Inferência (carregar artefato) ---
    modelo_carregado = PlackettLuce.load(str(modelo_path))
    assert modelo_carregado.esta_ajustado
    # Verificar que os pesos são os mesmos (todas as 60 dezenas)
    for d in range(1, 61):
        assert modelo.pesos[d - 1] == modelo_carregado.pesos[d - 1]

    # --- 5. Geração de jogos ---
    gen = Generator(model_name="M1", model=modelo_carregado, seed=42)
    resultado = gen.generate(K=5, max_shared_numbers=1)
    assert len(resultado.games) == 5
    for jogo in resultado.games:
        _validar_jogo(jogo)

    # --- 6. Manifesto de geração ---
    save_generation(resultado, tmp_path)
    arquivos_gerados = list(tmp_path.glob("*.json"))
    assert len(arquivos_gerados) >= 1, f"Nenhum arquivo de geração em {tmp_path}"
    caminho_gen = arquivos_gerados[0]

    carregado = load_generation(caminho_gen)
    assert len(carregado.games) == 5
    assert carregado.metadata.seed == 42
    assert carregado.metadata.model_name == "M1"
    assert carregado.metadata.saturated is False

    for jogo_original, jogo_carregado in zip(resultado.games, carregado.games):
        assert jogo_original == jogo_carregado

    print("E2E: pipeline completo executado com sucesso!")


def test_e2e_geracao_uniforme_sem_tf(tmp_path: Path) -> None:
    """Teste de geração uniforme que não requer TensorFlow."""
    from lstm_virada.generation import Generator

    gen = Generator(model_name="M0", seed=42)
    resultado = gen.generate(K=3, max_shared_numbers=2)
    assert len(resultado.games) == 3
    for jogo in resultado.games:
        _validar_jogo(jogo, N=60, k=6)

    # Salvar e carregar geração
    from lstm_virada.generation import save_generation, load_generation

    save_generation(resultado, tmp_path)
    arquivos_gerados = list(tmp_path.glob("*.json"))
    assert len(arquivos_gerados) >= 1, f"Nenhum arquivo de geração em {tmp_path}"
    caminho_gen = arquivos_gerados[0]

    carregado = load_generation(caminho_gen)
    assert len(carregado.games) == 3
    assert carregado.metadata.model_name == "M0"
    assert carregado.metadata.seed == 42
    print("E2E: geração uniforme (sem TF) executada com sucesso!")
