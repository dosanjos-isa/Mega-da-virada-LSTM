"""Módulo de codificação, decodificação e construção de janelas temporais.

Responsável por:
- Vetorização multi-hot de 60 dezenas + indicador binário is_virada (61 dimensões por concurso).
- Transformação pura e determinística entre dezenas sorteadas e classes de treinamento.
- Construção de janelas deslizantes (W) preservando estrita causalidade temporal.
- Prevenção formal de vazamento de dados (data leakage): nenhum concurso-alvo ou posterior
  é admitido no histórico da janela temporal.

Cobre: RT-005, RT-006, RF-003
Tarefa: T2 — Codificação e decodificação
"""

from __future__ import annotations

from typing import Final, Sequence
import numpy as np
import pandas as pd

from lstm_virada.data import DEZENA_COLS

# ==============================================================================
# Constantes do Domínio
# ==============================================================================
NUM_CLASSES: Final[int] = 60
NUM_DEZENAS_JOGO: Final[int] = 6
FEATURES_PER_DRAW: Final[int] = 61  # 60 classes multi-hot + 1 flag is_virada
DEFAULT_WINDOW_SIZE: Final[int] = 20
ALLOWED_WINDOW_SIZES: Final[tuple[int, ...]] = (10, 20, 40)


class EncodingError(ValueError):
    """Exceção levantada quando ocorre inconsistência na codificação ou decodificação."""
    pass


class TemporalLeakageError(RuntimeError):
    """Exceção levantada quando é detectado vazamento de dados temporais na janela."""
    pass


def _resolve_dataframe_columns(df: pd.DataFrame) -> tuple[str, list[str], str]:
    """Resolve dinamicamente os nomes das colunas de concurso, dezenas e is_virada.

    Suporta tanto o dataset processado oficial (concurso, d1..d6, is_virada)
    quanto formatos alternativos (Concurso, N1..N6, is_virada).
    """
    cols = set(df.columns)

    # Identifica coluna de concurso
    conc_col: str | None = None
    for cand in ("concurso", "Concurso", "CONCURSO"):
        if cand in cols:
            conc_col = cand
            break
    if conc_col is None:
        raise EncodingError("DataFrame não contém a coluna de identificação do concurso ('concurso').")

    # Identifica colunas de dezenas
    dez_cols: list[str] | None = None
    if all(f"d{i}" in cols for i in range(1, 7)):
        dez_cols = [f"d{i}" for i in range(1, 7)]
    elif all(f"N{i}" in cols for i in range(1, 7)):
        dez_cols = [f"N{i}" for i in range(1, 7)]
    elif all(f"n{i}" in cols for i in range(1, 7)):
        dez_cols = [f"n{i}" for i in range(1, 7)]
    else:
        raise EncodingError("DataFrame não contém as 6 colunas de dezenas (esperado 'd1..d6' ou 'N1..N6').")

    # Identifica coluna is_virada
    virada_col: str | None = None
    for cand in ("is_virada", "IS_VIRADA"):
        if cand in cols:
            virada_col = cand
            break
    if virada_col is None:
        raise EncodingError("DataFrame não contém a coluna 'is_virada'.")

    return conc_col, dez_cols, virada_col


# ==============================================================================
# Funções Puras de Conversão (RT-005)
# ==============================================================================
def draw_to_multihot(draw: Sequence[int], is_virada: int | bool = 0) -> np.ndarray:
    """Converte um sorteio de 6 dezenas no vetor multi-hot estendido de 61 posições.

    # cobre: RT-005

    Args:
        draw: Sequência de 6 dezenas inteiras distintas no intervalo [1, 60].
        is_virada: Indicador se o sorteio em questão é da Mega da Virada (0 ou 1).

    Returns:
        np.ndarray de 61 posições float32:
        - Índices 0 a 59: 1.0 se a dezena (índice + 1) foi sorteada, senão 0.0.
        - Índice 60: 1.0 se is_virada=1, senão 0.0.

    Raises:
        EncodingError: Se o sorteio não possuir exatamente 6 dezenas distintas em [1, 60].
    """
    if len(draw) != NUM_DEZENAS_JOGO:
        raise EncodingError(
            f"Sorteio deve conter exatamente {NUM_DEZENAS_JOGO} dezenas, recebido: {len(draw)}."
        )

    draw_set = set(draw)
    if len(draw_set) != NUM_DEZENAS_JOGO:
        raise EncodingError(
            f"Sorteio contém dezenas repetidas: {list(draw)}."
        )

    for num in draw:
        if not (1 <= num <= NUM_CLASSES):
            raise EncodingError(
                f"Dezena {num} fora do intervalo permitido [1, {NUM_CLASSES}]."
            )

    vector = np.zeros(FEATURES_PER_DRAW, dtype=np.float32)
    for num in draw:
        vector[num - 1] = 1.0

    vector[60] = 1.0 if bool(is_virada) else 0.0
    return vector


def multihot_to_draw(vector: np.ndarray) -> tuple[tuple[int, ...], int]:
    """Converte um vetor multi-hot de volta para a tupla de 6 dezenas e flag is_virada.

    # cobre: RT-005

    Args:
        vector: Array 1D de 60 ou 61 posições.

    Returns:
        Tupla com:
        - 6 dezenas ordenadas de forma estritamente crescente no intervalo [1, 60].
        - Inteiro is_virada (0 ou 1).

    Raises:
        EncodingError: Se o vetor não tiver tamanho 60 ou 61, ou se a quantidade
                       de dezenas ativas não for exatamente 6.
    """
    vec = np.asarray(vector).ravel()
    if vec.shape[0] not in (NUM_CLASSES, FEATURES_PER_DRAW):
        raise EncodingError(
            f"Vetor de entrada deve ter dimensão {NUM_CLASSES} ou {FEATURES_PER_DRAW}, recebido: {vec.shape[0]}."
        )

    multihot_part = vec[:NUM_CLASSES]
    active_indices = np.where(multihot_part > 0.5)[0]

    if len(active_indices) != NUM_DEZENAS_JOGO:
        raise EncodingError(
            f"Vetor multi-hot deve possuir exatamente {NUM_DEZENAS_JOGO} posições ativas, encontrado: {len(active_indices)}."
        )

    dezenas = tuple(int(idx + 1) for idx in sorted(active_indices))
    is_virada = int(vec[60] > 0.5) if vec.shape[0] == FEATURES_PER_DRAW else 0

    return dezenas, is_virada


def draw_to_targets(draw: Sequence[int]) -> np.ndarray:
    """Converte dezenas ordenadas (1 a 60) em classes-alvo zero-based [0, 59].

    # cobre: RT-005

    Args:
        draw: Sequência de 6 dezenas ordenadas no intervalo [1, 60].

    Returns:
        np.ndarray de 6 inteiros int32 com valores em [0, 59].
    """
    if len(draw) != NUM_DEZENAS_JOGO:
        raise EncodingError(f"Esperadas {NUM_DEZENAS_JOGO} dezenas, recebido: {len(draw)}.")

    sorted_draw = tuple(sorted(draw))
    if sorted_draw != tuple(draw):
        raise EncodingError(f"Dezenas devem ser fornecidas em ordem crescente: {draw}.")

    if len(set(draw)) != NUM_DEZENAS_JOGO:
        raise EncodingError(f"Dezenas não podem se repetir: {draw}.")

    targets = np.empty(NUM_DEZENAS_JOGO, dtype=np.int32)
    for i, num in enumerate(draw):
        if not (1 <= num <= NUM_CLASSES):
            raise EncodingError(f"Dezena {num} fora do intervalo [1, {NUM_CLASSES}].")
        targets[i] = num - 1

    return targets


def targets_to_draw(targets: Sequence[int]) -> tuple[int, ...]:
    """Converte classes-alvo zero-based [0, 59] de volta para dezenas (1 a 60).

    # cobre: RT-005

    Args:
        targets: Sequência de 6 inteiros em [0, 59].

    Returns:
        Tupla de 6 dezenas em [1, 60].
    """
    if len(targets) != NUM_DEZENAS_JOGO:
        raise EncodingError(f"Esperados {NUM_DEZENAS_JOGO} alvos, recebido: {len(targets)}.")

    if not all(0 <= t < NUM_CLASSES for t in targets):
        raise EncodingError(f"Alvos contêm valores fora da faixa [0, {NUM_CLASSES - 1}]: {targets}.")

    return tuple(t + 1 for t in targets)


# ==============================================================================
# Janelamento Temporal e Prevenção de Vazamento (RT-006)
# ==============================================================================
def create_temporal_windows(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    target_virada_only: bool = False,
    sample_weight_regular: float = 1.0,
) -> dict[str, np.ndarray]:
    """Constrói amostras de janelas deslizantes (W) com validação estrita anti-vazamento.

    Para prever o concurso no índice temporal t, a entrada sequencial é composta
    exclusivamente pelos concursos nos índices [t - window_size, t - 1].
    O concurso t (alvo) NUNCA é incluído na janela sequencial de entrada.

    # cobre: RT-005, RT-006

    Args:
        df: DataFrame ordenado por Concurso com colunas ['Concurso', 'is_virada', 'N1'..'N6'].
        window_size: Tamanho da janela W (default: 20).
        target_virada_only: Se True, filtra apenas amostras cujo alvo é Mega da Virada.
        sample_weight_regular: Peso alpha para concursos regulares (Virada tem peso 1.0 fixo).

    Returns:
        Dicionário com arrays NumPy:
        - "x_seq": (N, W, 61) float32
        - "x_static": (N, 1) float32 (alvo_virada)
        - "y": (N, 6) int32 (alvos das 6 cabeças)
        - "sample_weights": (N,) float32
        - "target_concursos": (N,) int64
        - "target_is_virada": (N,) int32
        - "window_last_concursos": (N,) int64 (último concurso de cada janela, para auditoria)

    Raises:
        EncodingError: Se os dados forem insuficientes ou o formato for inválido.
        TemporalLeakageError: Se houver qualquer sobreposição entre janela e alvo.
    """
    if window_size < 1:
        raise EncodingError(f"Tamanho de janela deve ser positivo, recebido: {window_size}.")

    conc_col, dez_cols, virada_col = _resolve_dataframe_columns(df)

    total_rows = len(df)
    if total_rows <= window_size:
        raise EncodingError(
            f"Base com {total_rows} concursos é insuficiente para janela de tamanho {window_size}."
        )

    # Pré-calcula matriz multi-hot de todos os concursos para eficiência
    # Shape: (total_rows, 61)
    concurso_arr = df[conc_col].to_numpy(dtype=np.int64)
    is_virada_arr = df[virada_col].to_numpy(dtype=np.int32)
    dezenas_matrix = df[dez_cols].to_numpy(dtype=np.int32)

    all_multihot = np.zeros((total_rows, FEATURES_PER_DRAW), dtype=np.float32)
    for r in range(total_rows):
        for dez in dezenas_matrix[r]:
            all_multihot[r, dez - 1] = 1.0
        all_multihot[r, 60] = float(is_virada_arr[r])

    # Pré-calcula alvos para todas as linhas (índices zero-based 0 a 59)
    all_targets = dezenas_matrix - 1

    x_seq_list: list[np.ndarray] = []
    x_static_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    weights_list: list[float] = []
    target_concursos_list: list[int] = []
    target_is_virada_list: list[int] = []
    window_last_concursos_list: list[int] = []

    # Iteração temporal a partir de window_size
    for t in range(window_size, total_rows):
        is_target_virada = int(is_virada_arr[t])

        if target_virada_only and not is_target_virada:
            continue

        target_conc = int(concurso_arr[t])
        window_concursos = concurso_arr[t - window_size : t]
        last_window_conc = int(window_concursos[-1])

        # Verificação rigorosa contra vazamento temporal (RT-006)
        if last_window_conc >= target_conc:
            raise TemporalLeakageError(
                f"Vazamento temporal detectado no índice {t}: último concurso da janela "
                f"({last_window_conc}) >= concurso alvo ({target_conc})."
            )

        if target_conc in window_concursos:
            raise TemporalLeakageError(
                f"Vazamento temporal: concurso alvo {target_conc} está presente na janela de entrada!"
            )

        # Fatia sequencial de tamanho W
        window_features = all_multihot[t - window_size : t]  # shape: (W, 61)

        weight = 1.0 if is_target_virada else float(sample_weight_regular)

        x_seq_list.append(window_features)
        x_static_list.append(np.array([float(is_target_virada)], dtype=np.float32))
        y_list.append(all_targets[t])
        weights_list.append(weight)
        target_concursos_list.append(target_conc)
        target_is_virada_list.append(is_target_virada)
        window_last_concursos_list.append(last_window_conc)

    return {
        "x_seq": np.asarray(x_seq_list, dtype=np.float32),
        "x_static": np.asarray(x_static_list, dtype=np.float32),
        "y": np.asarray(y_list, dtype=np.int32),
        "sample_weights": np.asarray(weights_list, dtype=np.float32),
        "target_concursos": np.asarray(target_concursos_list, dtype=np.int64),
        "target_is_virada": np.asarray(target_is_virada_list, dtype=np.int32),
        "window_last_concursos": np.asarray(window_last_concursos_list, dtype=np.int64),
    }


def extract_latest_inference_window(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    alvo_virada: int = 1,
) -> dict[str, np.ndarray | int]:
    """Extrai a janela temporal mais recente do histórico para inferência operacional.

    # cobre: RT-005, RG-001

    Args:
        df: DataFrame ordenado por Concurso com os dados históricos consolidados.
        window_size: Quantidade de sorteios mais recentes a incluir na janela (default: 20).
        alvo_virada: Flag do contexto estático para o sorteio a prever (padrão: 1 para Mega da Virada).

    Returns:
        Dicionário com:
        - "x_seq": Tensor 3D de shape (1, window_size, 61) pronto para model.predict().
        - "x_static": Tensor 2D de shape (1, 1) com valor [[float(alvo_virada)]].
        - "last_concurso": Inteiro com o número do concurso mais recente contido na janela.
        - "expected_target_concurso": Inteiro com o próximo número de concurso esperado.

    Raises:
        EncodingError: Se o DataFrame tiver menos sorteios que window_size.
    """
    total_rows = len(df)
    if total_rows < window_size:
        raise EncodingError(
            f"Histórico com {total_rows} concursos é insuficiente para janela de inferência {window_size}."
        )

    conc_col, dez_cols, virada_col = _resolve_dataframe_columns(df)

    # Coleta exatamente os últimos window_size sorteios
    recent_df = df.iloc[-window_size:]
    concursos = recent_df[conc_col].to_numpy(dtype=np.int64)
    last_conc = int(concursos[-1])

    dezenas = recent_df[dez_cols].to_numpy(dtype=np.int32)
    virada_flags = recent_df[virada_col].to_numpy(dtype=np.int32)

    features = np.zeros((window_size, FEATURES_PER_DRAW), dtype=np.float32)
    for r in range(window_size):
        for dez in dezenas[r]:
            features[r, dez - 1] = 1.0
        features[r, 60] = float(virada_flags[r])

    return {
        "x_seq": np.expand_dims(features, axis=0),  # shape: (1, W, 61)
        "x_static": np.array([[float(alvo_virada)]], dtype=np.float32),  # shape: (1, 1)
        "last_concurso": last_conc,
        "expected_target_concurso": last_conc + 1,
    }
