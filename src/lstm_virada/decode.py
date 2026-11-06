"""Módulo de decodificação das seis dezenas via Programação Dinâmica.

Implementa a decodificação ótima de máxima verossimilhança (Top-1 / Best Sequence)
em O(6 × 60), maximizando a soma das log-probabilidades sob a restrição estrita
de ordenação:
    max_{1 <= d1 < d2 < ... < d6 <= 60} sum_{i=1}^6 log P(i, di)

É terminantemente proibido utilizar o argmax isolado de cada cabeça, uma vez que
cabeças independentes podem predizer a mesma dezena ou violar a ordem estrita.

Cobre: RT-005, RG-002, RG-003, RG-005, RG-011
Tarefa: T2 — Codificação e decodificação
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

NUM_HEADS: Final[int] = 6
NUM_CLASSES: Final[int] = 60
EPSILON: Final[float] = 1e-12


class DecodingError(ValueError):
    """Exceção levantada quando a matriz de probabilidades fornecida é inválida."""

    pass


def _standardize_probs_matrix(probs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """Padroniza e valida a matriz de probabilidades para o formato (6, 60).

    Aceita tanto uma matriz 2D (6, 60) quanto uma lista/tupla de 6 saídas
    individuais de cabeças do Keras (cada uma com shape (60,) ou (1, 60)).

    Args:
        probs: Array NumPy ou sequência de arrays com as probabilidades.

    Returns:
        np.ndarray 2D de shape (6, 60) float64.

    Raises:
        DecodingError: Se o formato ou dimensões não corresponderem a (6, 60).
    """
    if isinstance(probs, (list, tuple)):
        if len(probs) != NUM_HEADS:
            raise DecodingError(
                f"Lista de saídas deve conter exatamente {NUM_HEADS} cabeças, recebido: {len(probs)}."
            )
        extracted = []
        for i, head_arr in enumerate(probs):
            arr = np.asarray(head_arr, dtype=np.float64).squeeze()
            if arr.shape != (NUM_CLASSES,):
                raise DecodingError(
                    f"Cabeça {i} possui formato {arr.shape}, esperado ({NUM_CLASSES},)."
                )
            extracted.append(arr)
        matrix = np.stack(extracted, axis=0)
    else:
        matrix = np.asarray(probs, dtype=np.float64)
        if matrix.ndim == 3 and matrix.shape[0] == 1:
            matrix = matrix.squeeze(axis=0)
        if matrix.shape != (NUM_HEADS, NUM_CLASSES):
            raise DecodingError(
                f"Matriz de probabilidades deve ter shape ({NUM_HEADS}, {NUM_CLASSES}), recebido: {matrix.shape}."
            )

    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        raise DecodingError("Matriz de probabilidades contém valores NaN ou Infinitos.")

    if np.any(matrix < 0.0):
        raise DecodingError("Matriz de probabilidades contém valores negativos.")

    if np.any(matrix > 1.0):
        raise DecodingError("Matriz de probabilidades contém valores maiores que 1.0.")

    return matrix


def decode_top1_dp(
    probs: np.ndarray | Sequence[np.ndarray],
    eps: float = EPSILON,
) -> tuple[int, int, int, int, int, int]:
    """Decodifica a combinação ótima global de 6 dezenas por Programação Dinâmica.

    Garante por construção matemática:
    1. Exatamente 6 dezenas distintas.
    2. Valores no intervalo fechado [1, 60].
    3. Ordenação estritamente crescente: d1 < d2 < d3 < d4 < d5 < d6.
    4. Complexidade exata O(6 * 60) = 360 operações de transição.

    # cobre: RT-005, RG-002, RG-003, RG-005, RG-011

    Args:
        probs: Matriz de probabilidades (6, 60) ou lista de 6 arrays de tamanho (60,).
        eps: Valor infinitesimal para evitar log(0). Padrão: 1e-12.

    Returns:
        Tupla com 6 inteiros correspondentes às dezenas sorteadas [1, 60].

    Raises:
        DecodingError: Se as dimensões forem inválidas.
    """
    matrix = _standardize_probs_matrix(probs)

    # Converte probabilidades em log-verossimilhanças numéricas seguras
    log_p = np.log(np.clip(matrix, eps, 1.0))

    # Tabela DP: dp[i, c] = max log-prob acumulada para as cabeças 0..i escolhendo a classe c na cabeça i
    # Backpointer: bp[i, c] = melhor classe da cabeça i-1
    neg_inf = -1e30
    dp = np.full((NUM_HEADS, NUM_CLASSES), neg_inf, dtype=np.float64)
    bp = np.full((NUM_HEADS, NUM_CLASSES), -1, dtype=np.int32)

    # Caso Base: Cabeça 0 (primeira dezena)
    # A primeira dezena pode variar de 0 até 54 (dezenas 1 a 55),
    # deixando espaço para pelo menos 5 dezenas subsequentes (55, 56, 57, 58, 59).
    max_c0 = NUM_CLASSES - NUM_HEADS  # 54
    for c in range(0, max_c0 + 1):
        dp[0, c] = log_p[0, c]

    # Transição: Cabeças 1 a 5
    for i in range(1, NUM_HEADS):
        min_c = i
        max_c = NUM_CLASSES - (NUM_HEADS - i)  # para i=5: 59; i=4: 58; etc.

        # Manutenção de prefix max O(1) sobre a cabeça anterior (i - 1)
        best_prev_val = neg_inf
        best_prev_idx = -1

        # Preenche de forma incremental mantendo o melhor predecessor c' < c
        for c in range(min_c, max_c + 1):
            cand_prev = c - 1
            if dp[i - 1, cand_prev] > best_prev_val:
                best_prev_val = dp[i - 1, cand_prev]
                best_prev_idx = cand_prev

            if best_prev_idx != -1 and best_prev_val > neg_inf / 2:
                dp[i, c] = best_prev_val + log_p[i, c]
                bp[i, c] = best_prev_idx

    # Encontrar a melhor classe final na cabeça 5
    c5_candidates = range(NUM_HEADS - 1, NUM_CLASSES)
    best_c5 = -1
    best_total_score = neg_inf

    for c in c5_candidates:
        if dp[5, c] > best_total_score:
            best_total_score = dp[5, c]
            best_c5 = c

    if best_c5 == -1:
        raise DecodingError(
            "Falha ao encontrar caminho viável na programação dinâmica."
        )

    # Reconstrução do caminho ótimo por backtracking
    selected_classes = [0] * NUM_HEADS
    curr = best_c5
    for i in range(NUM_HEADS - 1, -1, -1):
        selected_classes[i] = curr
        if i > 0:
            curr = bp[i, curr]

    # Conversão das classes zero-based [0, 59] para dezenas da Mega-Sena [1, 60]
    dezenas = tuple(c + 1 for c in selected_classes)

    # Validação matemática formal de integridade (RG-002, RG-003, RG-011)
    # P0-10: usa exceções DecodingError em vez de assert.
    if len(dezenas) != 6:
        raise DecodingError(f"Jogo deve conter 6 dezenas, obtido {len(dezenas)}.")
    if len(set(dezenas)) != 6:
        raise DecodingError(f"Jogo contém dezenas duplicadas: {dezenas}.")
    if not all(1 <= d <= 60 for d in dezenas):
        raise DecodingError(f"Dezenas fora do intervalo [1, 60]: {dezenas}.")
    if not all(dezenas[k] < dezenas[k + 1] for k in range(5)):
        raise DecodingError(f"Dezenas fora de ordem estrita: {dezenas}.")

    return (
        int(dezenas[0]),
        int(dezenas[1]),
        int(dezenas[2]),
        int(dezenas[3]),
        int(dezenas[4]),
        int(dezenas[5]),
    )


def decode_batch_dp(
    probs_batch: np.ndarray | Sequence[np.ndarray],
    eps: float = EPSILON,
) -> list[tuple[int, int, int, int, int, int]]:
    """Decodifica um lote de previsões em uma lista de tuplas de 6 dezenas ótimas.

    # cobre: RT-005, RG-001

    Args:
        probs_batch: Se lista de 6 arrays Keras, cada um tem shape (batch_size, 60).
                     Se array NumPy 3D, shape (batch_size, 6, 60).
        eps: Epsilon numérico para log.

    Returns:
        Lista de tuplas de 6 dezenas ordenadas.
    """
    if isinstance(probs_batch, (list, tuple)):
        # Lista com 6 arrays de shape (batch_size, 60)
        batch_size = probs_batch[0].shape[0]
        results = []
        for b in range(batch_size):
            sample_heads = [probs_batch[i][b] for i in range(NUM_HEADS)]
            results.append(decode_top1_dp(sample_heads, eps=eps))
        return results

    arr = np.asarray(probs_batch)
    if arr.ndim == 2 and arr.shape == (NUM_HEADS, NUM_CLASSES):
        return [decode_top1_dp(arr, eps=eps)]
    elif arr.ndim == 3 and arr.shape[1:] == (NUM_HEADS, NUM_CLASSES):
        return [decode_top1_dp(arr[b], eps=eps) for b in range(arr.shape[0])]
    else:
        raise DecodingError(
            f"Formato de lote não reconhecido: {arr.shape}. Esperado (batch, {NUM_HEADS}, {NUM_CLASSES}) "
            f"ou lista de {NUM_HEADS} saídas (batch, {NUM_CLASSES})."
        )


def naive_argmax_decode(probs: np.ndarray | Sequence[np.ndarray]) -> list[int]:
    """Executa o argmax independente por cabeça (PROIBIDO EM PRODUÇÃO).

    Esta função é fornecida estritamente para propósitos de teste, comparação
    científica e demonstração de falhas (demonstrando que o argmax independente
    gera duplicatas e violações de ordenação que o decodificador DP corrige).

    # cobre: RT-005 (demonstração de contraexemplo)

    Args:
        probs: Matriz de probabilidades (6, 60).

    Returns:
        Lista de 6 inteiros resultantes de argmax isolado + 1.
    """
    matrix = _standardize_probs_matrix(probs)
    return [int(np.argmax(matrix[i]) + 1) for i in range(NUM_HEADS)]
