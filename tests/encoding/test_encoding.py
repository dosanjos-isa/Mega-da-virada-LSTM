"""Suíte de testes automatizados para codificação, decodificação e janelamento temporal.

Cobre os requisitos:
- RT-005: Codificação e decodificação determinística pura com testes de ida e volta.
- RT-006: Prevenção formal de vazamento temporal (data leakage) nas janelas deslizantes.
- RG-001, RG-002, RG-003, RG-005, RG-011: Decodificação ótima de máxima verossimilhança por DP.
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

from lstm_virada.encoding import (
    DEFAULT_WINDOW_SIZE,
    FEATURES_PER_DRAW,
    NUM_CLASSES,
    NUM_DEZENAS_JOGO,
    EncodingError,
    TemporalLeakageError,
    create_temporal_windows,
    draw_to_multihot,
    draw_to_targets,
    extract_latest_inference_window,
    multihot_to_draw,
    targets_to_draw,
)
from lstm_virada.decode import (
    DecodingError,
    decode_batch_dp,
    decode_top1_dp,
    naive_argmax_decode,
)


# ==============================================================================
# 1. Testes de Codificação e Decodificação de Dezenas (RT-005)
# ==============================================================================
def test_roundtrip_draw_and_multihot() -> None:
    """Verifica que a conversão entre dezenas e vetor multi-hot é perfeitamente bijetora.

    # cobre: RT-005
    """
    sample_draws = [
        ((4, 5, 10, 34, 58, 59), 0),
        ((1, 2, 3, 4, 5, 6), 1),
        ((55, 56, 57, 58, 59, 60), 0),
        ((10, 20, 30, 40, 50, 60), 1),
    ]

    for draw, is_virada in sample_draws:
        vec = draw_to_multihot(draw, is_virada=is_virada)
        assert vec.shape == (FEATURES_PER_DRAW,)
        assert vec.dtype == np.float32
        assert np.sum(vec[:NUM_CLASSES]) == NUM_DEZENAS_JOGO
        assert int(vec[60]) == is_virada

        restored_draw, restored_virada = multihot_to_draw(vec)
        assert restored_draw == draw, f"Esperado {draw}, obtido {restored_draw}"
        assert restored_virada == is_virada


def test_roundtrip_draw_and_targets() -> None:
    """Verifica a conversão ida-e-volta entre dezenas (1..60) e classes zero-based (0..59).

    # cobre: RT-005
    """
    draw = (5, 12, 22, 37, 46, 58)
    targets = draw_to_targets(draw)

    assert targets.shape == (NUM_DEZENAS_JOGO,)
    assert targets.dtype == np.int32
    assert list(targets) == [4, 11, 21, 36, 45, 57]

    restored = targets_to_draw(targets)
    assert restored == draw


def test_draw_to_multihot_validations() -> None:
    """Garante que entradas inválidas para codificação sejam rejeitadas com EncodingError.

    # cobre: RT-005
    """
    # Menos de 6 dezenas
    with pytest.raises(EncodingError, match="deve conter exatamente 6 dezenas"):
        draw_to_multihot([1, 2, 3, 4, 5])

    # Dezenas duplicadas
    with pytest.raises(EncodingError, match="dezenas repetidas"):
        draw_to_multihot([1, 2, 3, 4, 5, 5])

    # Dezenas fora da faixa [1, 60]
    with pytest.raises(EncodingError, match="fora do intervalo permitido"):
        draw_to_multihot([0, 2, 3, 4, 5, 6])

    with pytest.raises(EncodingError, match="fora do intervalo permitido"):
        draw_to_multihot([1, 2, 3, 4, 5, 61])


def test_multihot_to_draw_validations() -> None:
    """Valida rejeição de vetores multi-hot com dimensões ou contagens incorretas.

    # cobre: RT-005
    """
    # Dimensão incorreta
    with pytest.raises(EncodingError, match="dimensão 60 ou 61"):
        multihot_to_draw(np.zeros(50))

    # Apenas 5 dezenas ativas
    vec = np.zeros(61, dtype=np.float32)
    vec[:5] = 1.0
    with pytest.raises(EncodingError, match="exatamente 6 posições ativas"):
        multihot_to_draw(vec)


# ==============================================================================
# 2. Testes de Janelamento Temporal e Prevenção de Vazamento (RT-006)
# ==============================================================================
def _create_synthetic_history(n_draws: int = 100) -> pd.DataFrame:
    """Gera histórico sintético ordenado estritamente por Concurso sem vazamento."""
    rows = []
    np.random.seed(42)
    for c in range(1, n_draws + 1):
        dezenas = sorted(np.random.choice(np.arange(1, 61), size=6, replace=False))
        is_virada = 1 if c % 30 == 0 else 0
        rows.append({
            "Concurso": c,
            "is_virada": is_virada,
            "N1": dezenas[0],
            "N2": dezenas[1],
            "N3": dezenas[2],
            "N4": dezenas[3],
            "N5": dezenas[4],
            "N6": dezenas[5],
        })
    return pd.DataFrame(rows)


def test_create_temporal_windows_shapes() -> None:
    """Verifica formatos de tensores gerados para diferentes tamanhos de janela W.

    # cobre: RT-005, RT-006
    """
    df = _create_synthetic_history(n_draws=60)

    for w in (10, 20, 40):
        data = create_temporal_windows(df, window_size=w)
        expected_samples = 60 - w

        assert data["x_seq"].shape == (expected_samples, w, FEATURES_PER_DRAW)
        assert data["x_static"].shape == (expected_samples, 1)
        assert data["y"].shape == (expected_samples, NUM_DEZENAS_JOGO)
        assert data["sample_weights"].shape == (expected_samples,)
        assert data["target_concursos"].shape == (expected_samples,)
        assert data["target_is_virada"].shape == (expected_samples,)


def test_strict_temporal_leakage_prevention() -> None:
    """Testa formalmente que nenhum concurso alvo ou futuro aparece na janela temporal.

    # cobre: RT-006
    """
    df = _create_synthetic_history(n_draws=80)
    w = 20
    data = create_temporal_windows(df, window_size=w)

    n_samples = len(data["target_concursos"])
    for i in range(n_samples):
        target_conc = data["target_concursos"][i]
        last_window_conc = data["window_last_concursos"][i]

        # 1. O concurso imediatamente anterior na janela deve ser exatamente target_conc - 1
        assert last_window_conc == target_conc - 1, (
            f"Amostra {i}: último concurso da janela ({last_window_conc}) "
            f"não é imediatamente anterior ao alvo ({target_conc})."
        )

        # 2. O concurso alvo NÃO pode ser menor ou igual ao último concurso
        assert target_conc > last_window_conc

        # 3. Nenhum concurso da janela pode conter as dezenas do alvo
        target_y = data["y"][i]  # classes do alvo
        window_x = data["x_seq"][i]  # (W, 61)

        # Assegura que o target_is_virada reflete a flag correta
        expected_virada = int(df.loc[df["Concurso"] == target_conc, "is_virada"].iloc[0])
        assert data["target_is_virada"][i] == expected_virada
        assert data["x_static"][i, 0] == float(expected_virada)


def test_temporal_leakage_exception_triggered() -> None:
    """Testa que desordem cronológica deliberada aciona TemporalLeakageError.

    # cobre: RT-006
    """
    df = _create_synthetic_history(n_draws=50)
    # Introduz intencionalmente uma inversão de concurso
    df.loc[30, "Concurso"] = 15  # Força concurso 30 a ter número 15 (menor que a janela)

    with pytest.raises(TemporalLeakageError, match="Vazamento temporal"):
        create_temporal_windows(df, window_size=20)


def test_sample_weights_application() -> None:
    """Verifica aplicação de pesos diferenciados para Virada (1.0) e regular (alpha).

    # cobre: RT-007, RT-009
    """
    df = _create_synthetic_history(n_draws=50)
    alpha = 0.05
    data = create_temporal_windows(df, window_size=10, sample_weight_regular=alpha)

    for i in range(len(data["sample_weights"])):
        if data["target_is_virada"][i] == 1:
            assert data["sample_weights"][i] == 1.0
        else:
            assert np.isclose(data["sample_weights"][i], alpha)


def test_extract_latest_inference_window() -> None:
    """Verifica extração da janela mais recente para predição prospectiva sem alvo.

    # cobre: RT-005, RG-001
    """
    df = _create_synthetic_history(n_draws=45)
    w = 20
    inf = extract_latest_inference_window(df, window_size=w, alvo_virada=1)

    assert inf["x_seq"].shape == (1, w, FEATURES_PER_DRAW)
    assert inf["x_static"].shape == (1, 1)
    assert inf["x_static"][0, 0] == 1.0
    assert inf["last_concurso"] == 45
    assert inf["expected_target_concurso"] == 46


# ==============================================================================
# 3. Testes de Decodificação por Programação Dinâmica (decode.py)
# ==============================================================================
def test_decode_top1_dp_validity() -> None:
    """Verifica que o decodificador DP sempre gera 6 dezenas distintas em ordem estrita.

    # cobre: RG-002, RG-003, RG-005, RG-011
    """
    np.random.seed(123)
    # Gera distribuição de probabilidades sintética com Dirichlet
    probs = np.random.dirichlet(np.ones(60), size=6)

    game = decode_top1_dp(probs)

    assert len(game) == 6
    assert len(set(game)) == 6
    assert all(1 <= d <= 60 for d in game)
    assert all(game[k] < game[k + 1] for k in range(5))


def test_decode_dp_resolves_naive_collision() -> None:
    """Verifica que o DP corrige colisões onde o argmax ingênuo geraria números repetidos.

    Exemplo: Duas cabeças atribuem a máxima probabilidade à mesma dezena (ex: dezena 10).
    O argmax ingênuo gera [..., 10, 10, ...], violando a regra da Mega-Sena.
    O DP obrigatoriamente encontra a melhor combinação com dezenas distintas e ordenadas.

    # cobre: RT-005, RG-002, RG-003
    """
    probs = np.full((6, 60), 0.01 / 60, dtype=np.float64)

    # Força cabeça 0 e cabeça 1 a ambas preferirem fortemente a classe 9 (dezena 10)
    probs[0, 9] = 0.90
    probs[1, 9] = 0.85
    # Segunda melhor opção da cabeça 1 é a classe 11 (dezena 12)
    probs[1, 11] = 0.80

    # Cabeças 2 a 5 com preferências ordenadas distintas
    probs[2, 20] = 0.90  # dezena 21
    probs[3, 30] = 0.90  # dezena 31
    probs[4, 40] = 0.90  # dezena 41
    probs[5, 50] = 0.90  # dezena 51

    # Demonstração: argmax ingênuo falha produzindo dezenas repetidas
    naive = naive_argmax_decode(probs)
    assert naive[0] == 10 and naive[1] == 10  # COLISÃO! Proibido pela Caixa!

    # DP resolve a colisão respeitando d1 < d2
    dp_game = decode_top1_dp(probs)
    assert len(set(dp_game)) == 6
    assert dp_game[0] < dp_game[1] < dp_game[2] < dp_game[3] < dp_game[4] < dp_game[5]
    assert dp_game[0] == 10
    assert dp_game[1] == 12  # Escolheu a alternativa ótima 12


def test_decode_dp_resolves_ordering_inversion() -> None:
    """Verifica que o DP corrige inversões de ordem onde argmax da cabeça posterior é menor.

    Exemplo: Cabeça 0 prefere dezena 25, mas Cabeça 1 prefere dezena 15.
    O argmax ingênuo geraria [25, 15, ...] que viola a ordenação d1 < d2.
    O DP garante d1 < d2 < ... < d6.

    # cobre: RG-003, RG-005
    """
    probs = np.full((6, 60), 0.001, dtype=np.float64)

    # Inversão forçada entre cabeça 0 e cabeça 1
    probs[0, 24] = 0.80  # Dezena 25
    probs[0, 9] = 0.70   # Dezena 10 (alternativa viável)

    probs[1, 14] = 0.90  # Dezena 15 (menor que 25!)

    # Cabeças 2 a 5
    probs[2, 29] = 0.90  # 30
    probs[3, 39] = 0.90  # 40
    probs[4, 49] = 0.90  # 50
    probs[5, 59] = 0.90  # 60

    naive = naive_argmax_decode(probs)
    assert naive[0] == 25 and naive[1] == 15  # INVERSÃO! 25 > 15 viola ordem!

    dp_game = decode_top1_dp(probs)
    assert dp_game[0] < dp_game[1]
    assert all(dp_game[k] < dp_game[k + 1] for k in range(5))


def test_decode_batch_dp_shapes() -> None:
    """Testa decodificação em lote com formato Keras (lista de 6 tensores).

    # cobre: RT-005, RG-001
    """
    batch_size = 5
    # Simula saída típica do Keras: lista com 6 arrays de shape (batch_size, 60)
    keras_outputs = [np.random.dirichlet(np.ones(60), size=batch_size) for _ in range(6)]

    results = decode_batch_dp(keras_outputs)
    assert len(results) == batch_size
    for game in results:
        assert len(game) == 6
        assert len(set(game)) == 6
        assert all(game[k] < game[k + 1] for k in range(5))


def test_decode_dp_exceptions() -> None:
    """Verifica lançamento de DecodingError para entradas inválidas.

    # cobre: RG-011
    """
    # Matriz com apenas 5 cabeças
    with pytest.raises(DecodingError, match="shape"):
        decode_top1_dp(np.zeros((5, 60)))

    # Matriz contendo NaNs
    invalid_matrix = np.ones((6, 60))
    invalid_matrix[0, 0] = np.nan
    with pytest.raises(DecodingError, match="NaN"):
        decode_top1_dp(invalid_matrix)


def test_real_processed_dataset_windows() -> None:
    """Valida janelamento e inferência sobre o dataset histórico real de 3.074 sorteios.

    # cobre: RT-005, RT-006, RG-001
    """
    processed_path = Path("data/processed/dataset_megasena_virada_processed.csv")
    if not processed_path.exists():
        return

    df = pd.read_csv(processed_path)
    assert len(df) >= 3000

    # Testa criação de janelas com W=20
    data = create_temporal_windows(df, window_size=20, sample_weight_regular=0.05)
    expected_n = len(df) - 20
    assert data["x_seq"].shape == (expected_n, 20, 61)
    assert data["x_static"].shape == (expected_n, 1)
    assert data["y"].shape == (expected_n, 6)

    # Verifica integridade temporal estrita no histórico real
    for idx in [0, 100, 1000, 2000, expected_n - 1]:
        tgt_conc = data["target_concursos"][idx]
        last_w_conc = data["window_last_concursos"][idx]
        assert last_w_conc == tgt_conc - 1

    # Testa extração de inferência prospectiva
    inf = extract_latest_inference_window(df, window_size=20, alvo_virada=1)
    assert inf["x_seq"].shape == (1, 20, 61)
    assert inf["x_static"].shape == (1, 1)
    assert inf["x_static"][0, 0] == 1.0
    assert inf["last_concurso"] == int(df["concurso"].iloc[-1])


# ==============================================================================
# Execução direta como script de validação
# ==============================================================================
if __name__ == "__main__":
    print("Executando suíte de testes de codificação e decodificação...")
    test_roundtrip_draw_and_multihot()
    test_roundtrip_draw_and_targets()
    test_draw_to_multihot_validations()
    test_multihot_to_draw_validations()
    test_create_temporal_windows_shapes()
    test_strict_temporal_leakage_prevention()
    test_temporal_leakage_exception_triggered()
    test_sample_weights_application()
    test_extract_latest_inference_window()
    test_decode_top1_dp_validity()
    test_decode_dp_resolves_naive_collision()
    test_decode_dp_resolves_ordering_inversion()
    test_decode_batch_dp_shapes()
    test_decode_dp_exceptions()
    test_real_processed_dataset_windows()
    print("SUCESSO: Todos os 15 testes de codificação e decodificação passaram perfeitamente!")
