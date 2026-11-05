"""Modelo M2 — Fisher (produto condicional).

Implementa o modelo de sensibilidade definido em spec/math_spec.md §4.3:

    P_M2(S) = prod_{i in S} w_i / e_6(w)

    onde e_6(w) = soma_{|T|=6} prod_{i in T} w_i (polinomio simetrico elementar).

A normalizacao e calculada por DP em O(N * k):
    e[0] = 1
    para i = 1..N:
        para j = min(i, k) ate 1:
            e[j] += w[i] * e[j-1]

Identificabilidade: w e c*w produzem a mesma distribuicao. Fixa-se
soma_i log w_i = 0 (media geometrica 1).

O modo e exato: o conjunto das k maiores pesos.

Cobre: ADR-002 (M2 como sensibilidade), math_spec.md §4.3, Apêndice B.1.
P1-02: implementacao do M2.
"""

from __future__ import annotations

from itertools import combinations
from typing import Final

import numpy as np

NUM_CLASSES: Final[int] = 60
NUM_DEZENAS_JOGO: Final[int] = 6


class FisherError(ValueError):
    """Excecao levantada quando o modelo Fisher encontra dados invalidos."""


class FisherModel:
    """Modelo Fisher (produto condicional) estático (não usa histórico).

    Atribui um peso w_i > 0 a cada dezena i. A probabilidade de um conjunto
    S = {d1, ..., dk} eh:

        P(S) = prod_{i in S} w_i / e_k(w)

    onde e_k(w) eh o polinomio simetrico elementar de grau k.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_dezenas_jogo: int = NUM_DEZENAS_JOGO,
    ) -> None:
        self._N = int(num_classes)
        self._k = int(num_dezenas_jogo)
        self._w: np.ndarray | None = None
        self._log_w: np.ndarray | None = None
        self._e_k: float | None = None  # valor em cache de e_k(w)

    # ------------------------------------------------------------------
    # Ajuste de pesos
    # ------------------------------------------------------------------
    def ajustar_pesos(self, pesos: dict[int, float]) -> None:
        """Define os pesos das dezenas.

        Args:
            pesos: dicionario {dezena: peso} com dezenas em [1, N].

        Raises:
            FisherError: Se algum peso nao for estritamente positivo
                ou se faltarem dezenas.
        """
        if not pesos:
            raise FisherError("Pesos nao podem ser vazios.")

        esperadas = set(range(1, self._N + 1))
        fornecidas = set(pesos.keys())
        if fornecidas != esperadas:
            faltando = esperadas - fornecidas
            extras = fornecidas - esperadas
            msg = ""
            if faltando:
                msg += f"Dezenas faltando: {sorted(faltando)}. "
            if extras:
                msg += f"Dezenas extras: {sorted(extras)}. "
            raise FisherError(msg.strip())

        w = np.array([pesos[i] for i in range(1, self._N + 1)], dtype=np.float64)

        if np.any(w <= 0):
            raise FisherError(
                f"Todos os pesos devem ser estritamente positivos. "
                f"Encontrados pesos <= 0 em: "
                f"{[i for i, p in pesos.items() if p <= 0]}."
            )

        # Identificabilidade: media geometrica 1
        w = w / np.exp(np.mean(np.log(w)))

        self._w = w
        self._log_w = np.log(w)
        self._e_k = self._calcular_e_k(w)

    # ------------------------------------------------------------------
    # Polinomio simetrico elementar e_k(w) por DP
    # ------------------------------------------------------------------
    def _calcular_e_k(self, w: np.ndarray) -> float:
        """Calcula e_k(w) por programacao dinamica em O(N * k).

        Recorrencia:
            e[0] = 1
            e[j] = 0 para j = 1..k
            para i = 1..N:
                para j = min(i, k) ate 1:
                    e[j] += w[i] * e[j-1]
        """
        e = np.zeros(self._k + 1, dtype=np.float64)
        e[0] = 1.0

        for i in range(self._N):
            wi = w[i]
            max_j = min(i + 1, self._k)
            for j in range(max_j, 0, -1):
                e[j] += wi * e[j - 1]

        return float(e[self._k])

    def e_k(self) -> float:
        """Retorna o valor de e_k(w) para os pesos atuais."""
        if self._w is None:
            raise FisherError(
                "Modelo nao ajustado. Chame ajustar_pesos() primeiro."
            )
        if self._e_k is None:
            self._e_k = self._calcular_e_k(self._w)
        return self._e_k

    # ------------------------------------------------------------------
    # Probabilidade de um conjunto
    # ------------------------------------------------------------------
    def _validar_conjunto(self, S: tuple[int, ...]) -> tuple[int, ...]:
        """Valida e normaliza um conjunto de dezenas."""
        if len(S) != self._k:
            raise FisherError(
                f"Conjunto deve conter exatamente {self._k} dezenas, "
                f"recebido: {len(S)}."
            )

        S_ordenado = tuple(sorted(int(d) for d in S))

        if len(set(S_ordenado)) != self._k:
            raise FisherError(
                f"Dezenas devem ser distintas: {S_ordenado}."
            )

        if not all(1 <= d <= self._N for d in S_ordenado):
            raise FisherError(
                f"Todas as dezenas devem estar em [1, {self._N}]: {S_ordenado}."
            )

        return S_ordenado

    def prob_conjunto(self, S: tuple[int, ...]) -> float:
        """Retorna P_M2(S) = prod_{i in S} w_i / e_k(w)."""
        if self._w is None:
            raise FisherError(
                "Modelo nao ajustado. Chame ajustar_pesos() primeiro."
            )

        S_ordenado = self._validar_conjunto(S)
        S_idx = tuple(d - 1 for d in S_ordenado)

        # Numerador: prod_{i in S} w_i
        log_prod_w = float(np.sum(self._log_w[list(S_idx)]))

        e_k = self.e_k()
        return float(np.exp(log_prod_w - np.log(e_k)))

    def log_prob_conjunto(self, S: tuple[int, ...]) -> float:
        """Retorna log P_M2(S) para um conjunto S."""
        p = self.prob_conjunto(S)
        if p <= 0.0:
            return -np.inf
        return float(np.log(p))

    def log_verossimilhança(self, dados: list[tuple[int, ...]]) -> float:
        """Retorna a log-verossimilhança de um conjunto de dados."""
        if self._w is None:
            raise FisherError(
                "Modelo nao ajustado. Chame ajustar_pesos() primeiro."
            )

        return float(sum(self.log_prob_conjunto(S) for S in dados))

    # ------------------------------------------------------------------
    # Modo do conjunto (exato para Fisher)
    # ------------------------------------------------------------------
    def modo(self) -> tuple[int, ...]:
        """Retorna o conjunto das k maiores pesos (modo exato do Fisher).

        Como e_k(w) e constante, o modo e simplesmente o conjunto das
        k dezenas de maior peso.
        """
        if self._w is None:
            raise FisherError(
                "Modelo nao ajustado. Chame ajustar_pesos() primeiro."
            )

        # Indices das k maiores pesos (ordem crescente de dezenas)
        top_k_idx = np.argsort(self._w)[-self._k :]
        top_k_idx = np.sort(top_k_idx)
        return tuple(int(d + 1) for d in top_k_idx)

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------
    @property
    def pesos(self) -> np.ndarray:
        """Retorna copia dos pesos w_1..w_N."""
        if self._w is None:
            raise FisherError(
                "Modelo nao ajustado. Chame ajustar_pesos() primeiro."
            )
        return self._w.copy()

    @property
    def esta_ajustado(self) -> bool:
        """Retorna True se os pesos ja foram ajustados."""
        return self._w is not None
