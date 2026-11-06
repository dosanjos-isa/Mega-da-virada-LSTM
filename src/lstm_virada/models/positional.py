"""Modelo M3 — Produto Posicional normalizado (order statistics).

Implementa o modelo definido em spec/math_spec.md §4.4:

    P_M3(S) = prod_{j=1..k} P_j(d_j) / Z

    onde Z = soma_{d1<...<dk} prod_{j=1..k} P_j(d_j), calculado por DP
    em O(k * N).

P_j e uma distribuicao de probabilidade sobre a j-ésima menor dezena,
com suporte supp_j = {j, ..., N-k+j}.

O modo e exato: DP encontra argmax de prod_j P_j(d_j), que eh o mesmo
que o modo de P_M3 (Z eh constante).

Cobre: ADR-002 (M3 como ablacao), math_spec.md §4.4.
P1-03: implementacao do M3.
"""

from __future__ import annotations

from typing import Final

import numpy as np

NUM_CLASSES: Final[int] = 60
NUM_DEZENAS_JOGO: Final[int] = 6


class PositionalError(ValueError):
    """Excecao levantada quando o modelo Produto Posicional encontra dados invalidos."""


class PositionalModel:
    """Modelo Produto Posicional estático (não usa histórico).

    Recebe k distribuicoes positionais P_1..P_k, onde P_j(c) eh a
    probabilidade de a j-ésima menor dezena ser c.

    A probabilidade de um conjunto S = {d1 < ... < dk} eh:

        P(S) = prod_{j=1..k} P_j(d_j) / Z

    onde Z eh a normalizacao sobre todos os conjuntos crescentes.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_dezenas_jogo: int = NUM_DEZENAS_JOGO,
    ) -> None:
        self._N = int(num_classes)
        self._k = int(num_dezenas_jogo)
        self._P: np.ndarray | None = None  # shape (k, N), P[j, c-1] = P_j(c)
        self._Z: float | None = None

    # ------------------------------------------------------------------
    # Ajuste de distribuicoes positionais
    # ------------------------------------------------------------------
    def ajustar_distribuicoes(
        self,
        distribuicoes: list[dict[int, float]],
    ) -> None:
        """Define as distribuicoes positionais P_1..P_k.

        Args:
            distribuicoes: lista de k dicionarios {dezena: probabilidade}.
                O j-esimo dicionario define P_j para c in supp_j =
                {j, ..., N-k+j}.

        Raises:
            PositionalError: Se o numero de distribuicoes nao for k,
                se alguma nao somar 1, ou se faltarem dezenas no suporte.
        """
        if len(distribuicoes) != self._k:
            raise PositionalError(
                f"Esperadas {self._k} distribuicoes positionais, "
                f"recebidas: {len(distribuicoes)}."
            )

        P = np.zeros((self._k, self._N), dtype=np.float64)

        for j in range(self._k):
            dist = distribuicoes[j]

            # Suporte admissível para a posição j (1-based):
            # supp_j = {j, ..., N-k+j}
            c_min = j + 1  # j-ésima menor dezena >= j+1 (1-based)
            c_max = self._N - (self._k - j - 1)  # <= N-k+j+1 (1-based) = N-k+j

            # Verifica que todas as dezenas do suporte estao presentes
            esperadas = set(range(c_min, c_max + 1))
            fornecidas = set(dist.keys())
            if fornecidas != esperadas:
                faltando = esperadas - fornecidas
                extras = fornecidas - esperadas
                msg = ""
                if faltando:
                    msg += f"Posicao {j + 1}: dezenas faltando {sorted(faltando)}. "
                if extras:
                    msg += f"Posicao {j + 1}: dezenas extras {sorted(extras)}. "
                raise PositionalError(msg.strip())

            soma = 0.0
            for c, p in dist.items():
                if p < 0:
                    raise PositionalError(
                        f"Posicao {j + 1}: probabilidade negativa para dezena {c}: {p}."
                    )
                soma += p
                P[j, int(c) - 1] = p

            if not np.isclose(soma, 1.0, atol=1e-10):
                raise PositionalError(
                    f"Posicao {j + 1}: distribuicao nao soma 1 (soma={soma})."
                )

        self._P = P
        self._Z = self._calcular_Z(P)

    # ------------------------------------------------------------------
    # Constante de normalizacao Z por DP
    # ------------------------------------------------------------------
    def _calcular_Z(self, P: np.ndarray) -> float:
        """Calcula Z por programacao dinamica em O(k * N).

        F[1][c] = P_1(c)
        F[j][c] = P_j(c) * soma_{c' < c} F[j-1][c']
        Z = soma_c F[k][c]
        """
        # F[j, c] para j=0..k-1, c=0..N-1 (zero-based)
        F = np.zeros((self._k, self._N), dtype=np.float64)

        # Cabeca 0: d1 pode ser 1..(N-k+1) em 1-based -> índice 0..(N-k)
        max_c0 = self._N - self._k
        for c in range(0, max_c0 + 1):
            F[0, c] = P[0, c]

        # Cabecas 1 a k-1
        for j in range(1, self._k):
            min_c = j  # d_j >= j+1 em 1-based -> índice >= j
            max_c = self._N - (self._k - j)  # índice max
            prefix_sum = 0.0
            for c in range(min_c, max_c + 1):
                prefix_sum += F[j - 1, c - 1]
                F[j, c] = P[j, c] * prefix_sum

        Z = float(np.sum(F[self._k - 1, :]))
        if Z <= 0:
            raise PositionalError("Z calculado é zero; verifique as distribuições P_j.")
        return Z

    def z(self) -> float:
        """Retorna a constante de normalização Z."""
        if self._P is None:
            raise PositionalError(
                "Modelo nao ajustado. Chame ajustar_distribuicoes() primeiro."
            )
        if self._Z is None:
            self._Z = self._calcular_Z(self._P)
        return self._Z

    # ------------------------------------------------------------------
    # Probabilidade de um conjunto
    # ------------------------------------------------------------------
    def _validar_conjunto(self, S: tuple[int, ...]) -> tuple[int, ...]:
        """Valida e normaliza um conjunto de dezenas."""
        if len(S) != self._k:
            raise PositionalError(
                f"Conjunto deve conter exatamente {self._k} dezenas, "
                f"recebido: {len(S)}."
            )

        S_ordenado = tuple(sorted(int(d) for d in S))

        if len(set(S_ordenado)) != self._k:
            raise PositionalError(f"Dezenas devem ser distintas: {S_ordenado}.")

        if not all(1 <= d <= self._N for d in S_ordenado):
            raise PositionalError(
                f"Todas as dezenas devem estar em [1, {self._N}]: {S_ordenado}."
            )

        return S_ordenado

    def prob_conjunto(self, S: tuple[int, ...]) -> float:
        """Retorna P_M3(S) = prod_j P_j(d_j) / Z."""
        if self._P is None:
            raise PositionalError(
                "Modelo nao ajustado. Chame ajustar_distribuicoes() primeiro."
            )

        S_ordenado = self._validar_conjunto(S)

        # Produto prod_j P_j(d_j)
        log_q = 0.0
        for j, dez in enumerate(S_ordenado):
            p_val = self._P[j, int(dez) - 1]
            if p_val <= 0.0:
                return 0.0
            log_q += np.log(p_val)

        return float(np.exp(log_q - np.log(self._Z)))

    def log_prob_conjunto(self, S: tuple[int, ...]) -> float:
        """Retorna log P_M3(S) para um conjunto S."""
        p = self.prob_conjunto(S)
        if p <= 0.0:
            return -np.inf
        return float(np.log(p))

    def log_verossimilhança(self, dados: list[tuple[int, ...]]) -> float:
        """Retorna a log-verossimilhança de um conjunto de dados."""
        if self._P is None:
            raise PositionalError(
                "Modelo nao ajustado. Chame ajustar_distribuicoes() primeiro."
            )

        return float(sum(self.log_prob_conjunto(S) for S in dados))

    # ------------------------------------------------------------------
    # Modo do conjunto (exato para M3 via DP)
    # ------------------------------------------------------------------
    def modo(self) -> tuple[int, ...]:
        """Retorna o conjunto de maior probabilidade (modo exato).

        Como Z eh constante, o modo de P_M3 eh o mesmo que o modo de
        prod_j P_j(d_j), encontrado por DP em O(k * N).

        Backtracking reconstrói o caminho ótimo.
        """
        if self._P is None:
            raise PositionalError(
                "Modelo nao ajustado. Chame ajustar_distribuicoes() primeiro."
            )

        # DP para encontrar o modo
        # dp[j, c] = max log-prob acumulado para posicoes 0..j com d_{j+1} = c
        # bp[j, c] = melhor classe da posicao j-1
        neg_inf = -1e30
        dp = np.full((self._k, self._N), neg_inf, dtype=np.float64)
        bp = np.full((self._k, self._N), -1, dtype=np.int32)

        # Caso base: cabeca 0 (d1 em 1..N-k+1)
        max_c0 = self._N - self._k
        for c in range(0, max_c0 + 1):
            if self._P[0, c] > 0:
                dp[0, c] = np.log(self._P[0, c])

        # Transicao
        for j in range(1, self._k):
            min_c = j
            max_c = self._N - (self._k - j)
            best_prev_val = neg_inf
            best_prev_idx = -1

            for c in range(min_c, max_c + 1):
                cand_prev = c - 1
                if dp[j - 1, cand_prev] > best_prev_val:
                    best_prev_val = dp[j - 1, cand_prev]
                    best_prev_idx = cand_prev

                if (
                    best_prev_idx != -1
                    and best_prev_val > neg_inf / 2
                    and self._P[j, c] > 0
                ):
                    dp[j, c] = best_prev_val + np.log(self._P[j, c])
                    bp[j, c] = best_prev_idx

        # Encontrar a melhor classe final
        c_last_candidates = range(self._k - 1, self._N)
        best_c = -1
        best_score = neg_inf
        for c in c_last_candidates:
            if dp[self._k - 1, c] > best_score:
                best_score = dp[self._k - 1, c]
                best_c = c

        if best_c == -1:
            raise PositionalError("Falha ao encontrar modo por DP.")

        # Reconstrucao por backtracking
        selected = [0] * self._k
        curr = best_c
        for j in range(self._k - 1, -1, -1):
            selected[j] = curr
            if j > 0:
                curr = bp[j, curr]

        return tuple(int(d + 1) for d in selected)

    # ------------------------------------------------------------------
    # Probabilidades de inclusao marginal p_i = P(i in S) (Brier score)
    # ------------------------------------------------------------------
    def prob_inclusao(self) -> np.ndarray:
        """Retorna p_i = P(i em S) para cada dezena i em [1, N].

        Calculado exatamente por DP: p_i = soma_{j=1..k} P(d_j = i),
        onde P(d_j = i) eh a margem posicional da dezena i na posicao j,
        normalizada por Z.

        Returns:
            np.ndarray shape (N,) com p_i para cada dezena (índice 0 = dezena 1).
        """
        if self._P is None:
            raise PositionalError(
                "Modelo nao ajustado. Chame ajustar_distribuicoes() primeiro."
            )

        k = self._k
        N = self._N
        P = self._P
        Z = self.z()

        # F_fwd[j, c] = soma sobre d_1<...<d_j=c de prod_{l=1..j} P_l(d_l)
        F_fwd = np.zeros((k, N), dtype=np.float64)
        max_c0 = N - k
        for c in range(0, max_c0 + 1):
            F_fwd[0, c] = P[0, c]
        for j in range(1, k):
            min_c = j
            max_c = N - (k - j)
            prefix_sum = 0.0
            for c in range(min_c, max_c + 1):
                prefix_sum += F_fwd[j - 1, c - 1]
                F_fwd[j, c] = P[j, c] * prefix_sum

        # F_bwd[j, c] = soma sobre d_j=c<d_{j+1}<...<d_k de prod_{l=j..k} P_l(d_l)
        F_bwd = np.zeros((k, N), dtype=np.float64)
        # Ultima posicao j=k-1: d_k em k..N
        for c in range(k - 1, N):
            F_bwd[k - 1, c] = P[k - 1, c]
        for j in range(k - 2, -1, -1):
            min_c = j
            max_c = N - (k - j)
            sufixo_sum = 0.0
            for c in range(max_c, min_c - 1, -1):
                if c + 1 <= N - 1:
                    sufixo_sum += F_bwd[j + 1, c + 1]
                F_bwd[j, c] = P[j, c] * sufixo_sum

        # p_i = soma_{j=0..k-1} (F_fwd[j, i-1] * F_bwd[j, i-1] / P[j, i-1] * P[j, i-1]) / Z
        # = soma_j (F_fwd[j, i-1] * F_bwd[j, i-1]) / Z   (quando P[j,i-1] > 0)
        p_inc = np.zeros(N, dtype=np.float64)
        for j in range(k):
            min_c = j
            max_c = N - (k - j)
            for c in range(min_c, max_c + 1):
                if P[j, c] <= 0:
                    continue
                # Conjuntos onde d_{j+1} = c+1:
                # esquerda = F_fwd[j, c], direita = F_bwd[j, c]
                # O produto F_fwd[j,c] * F_bwd[j,c] conta prod P_l(d_l)
                # para todos os conjuntos com d_{j+1}=c+1, mas P[j,c] eh contado
                # duas vezes (uma em cada lado). Dividir por P[j,c] corrige.
                contribuicao = F_fwd[j, c] * F_bwd[j, c] / P[j, c]
                p_inc[c] += contribuicao

        p_inc /= Z
        # Limpeza numerica
        p_inc = np.clip(p_inc, 0.0, 1.0)
        return p_inc

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------
    @property
    def distribuicoes(self) -> np.ndarray:
        """Retorna copia das distribuicoes positionais P (shape k x N)."""
        if self._P is None:
            raise PositionalError(
                "Modelo nao ajustado. Chame ajustar_distribuicoes() primeiro."
            )
        return self._P.copy()

    @property
    def esta_ajustado(self) -> bool:
        """Retorna True se as distribuicoes ja foram ajustadas."""
        return self._P is not None
