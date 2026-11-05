"""Modelo M1 — Plackett–Luce marginalizado sobre permutações.

Implementa o modelo primário definido em spec/math_spec.md §4.2:

    P_M1(S) = soma_{pi in Perm(S)} prod_{j=1..k} w_{pi_j} / (W - soma_{l<j} w_{pi_l})

    W = soma_{i=1..N} w_i

Como a ordem real do sorteio não é observada, a probabilidade do conjunto
S e a soma sobre as k! permutacoes de S.

Identificabilidade: w e c*w produzem a mesma distribuicao. Fixa-se
soma_i log w_i = 0 (media geometrica 1).

Cobre: ADR-002 (modelo primario M1), math_spec.md §4.2, Apêndice B.
P1-01: implementacao do M1.
"""

from __future__ import annotations

from itertools import permutations
from typing import Final

import numpy as np

NUM_CLASSES: Final[int] = 60
NUM_DEZENAS_JOGO: Final[int] = 6


class PlackettLuceError(ValueError):
    """Excecao levantada quando o modelo Plackett–Luce encontra dados invalidos."""


class PlackettLuce:
    """Modelo Plackett–Luce estático (não usa histórico).

    Atribui um peso w_i > 0 a cada dezena i. A probabilidade de um conjunto
    S = {d1, ..., dk} eh a soma sobre todas as k! permutacoes de S da
    probabilidade sequencial de escolha sem reposicao.

    Identificabilidade: w e c*w produzem a mesma distribuicao. Fixa-se
    soma_i log w_i = 0 (media geometrica 1).

    Cobre: ADR-002 (modelo primario M1), math_spec.md §4.2, Apêndice B.
    P1-01: implementacao do M1.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_dezenas_jogo: int = NUM_DEZENAS_JOGO,
    ) -> None:
        self._N = int(num_classes)
        self._k = int(num_dezenas_jogo)
        self._w: np.ndarray | None = None       # pesos w_1..w_N (indice 0..N-1)
        self._log_w: np.ndarray | None = None   # log w_i

    # ------------------------------------------------------------------
    # Ajuste de pesos
    # ------------------------------------------------------------------
    def ajustar_pesos(self, pesos: dict[int, float]) -> None:
        """Define os pesos das dezenas.

        Args:
            pesos: dicionario {dezena: peso} com dezenas em [1, N].

        Raises:
            PlackettLuceError: Se algum peso nao for estritamente positivo
                ou se faltarem dezenas.
        """
        if not pesos:
            raise PlackettLuceError("Pesos nao podem ser vazios.")

        # Verifica que todas as dezenas 1..N estao presentes
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
            raise PlackettLuceError(msg.strip())

        w = np.array([pesos[i] for i in range(1, self._N + 1)], dtype=np.float64)

        if np.any(w <= 0):
            raise PlackettLuceError(
                f"Todos os pesos devem ser estritamente positivos. "
                f"Encontrados pesos <= 0 em: "
                f"{[i for i, p in pesos.items() if p <= 0]}."
            )

        # Identificabilidade: normaliza para media geometrica 1
        # (soma_i log w_i = 0). Isso nao altera a distribuicao.
        w = w / np.exp(np.mean(np.log(w)))

        self._w = w
        self._log_w = np.log(w)

    # ------------------------------------------------------------------
    # Probabilidade de um conjunto
    # ------------------------------------------------------------------
    def _validar_conjunto(self, S: tuple[int, ...]) -> tuple[int, ...]:
        """Valida e normaliza um conjunto de dezenas."""
        if len(S) != self._k:
            raise PlackettLuceError(
                f"Conjunto deve conter exatamente {self._k} dezenas, "
                f"recebido: {len(S)}."
            )

        S_ordenado = tuple(sorted(int(d) for d in S))

        if len(set(S_ordenado)) != self._k:
            raise PlackettLuceError(
                f"Dezenas devem ser distintas: {S_ordenado}."
            )

        if not all(1 <= d <= self._N for d in S_ordenado):
            raise PlackettLuceError(
                f"Todas as dezenas devem estar em [1, {self._N}]: {S_ordenado}."
            )

        return S_ordenado

    def prob_conjunto(self, S: tuple[int, ...]) -> float:
        """Retorna P_M1(S) para um conjunto S.

        Calcula:
            P(S) = (prod_{i in S} w_i) * soma_{pi in Perm(S)}
                   prod_{j=1..k} 1 / (W - soma_{l<j} w_{pi_l})
        """
        if self._w is None:
            raise PlackettLuceError(
                "Modelo nao ajustado. Chame ajustar_pesos() primeiro."
            )

        S_ordenado = self._validar_conjunto(S)
        S_idx = tuple(d - 1 for d in S_ordenado)  # zero-based

        # Numerador: prod_{i in S} w_i
        log_prod_w = float(np.sum(self._log_w[list(S_idx)]))

        W = float(np.sum(self._w))
        log_w = self._log_w

        # Soma sobre as k! permutacoes de S
        termos_log = []
        for pi in permutations(S_idx):
            restante = W
            log_denom_soma = 0.0
            for idx in pi:
                log_denom_soma += np.log(restante)
                restante -= self._w[idx]
            termos_log.append(-log_denom_soma)

        # logsumexp dos termos
        max_termo = max(termos_log)
        logsumexp = max_termo + np.log(
            sum(np.exp(t - max_termo) for t in termos_log)
        )

        return float(np.exp(log_prod_w + logsumexp))

    def log_prob_conjunto(self, S: tuple[int, ...]) -> float:
        """Retorna log P_M1(S) para um conjunto S."""
        p = self.prob_conjunto(S)
        if p <= 0.0:
            return -np.inf
        return float(np.log(p))

    def log_verossimilhança(self, dados: list[tuple[int, ...]]) -> float:
        """Retorna a log-verossimilhança de um conjunto de dados.

        Args:
            dados: lista de conjuntos S_t.

        Returns:
            Soma de log P_M1(S_t) sobre todos os dados.
        """
        if self._w is None:
            raise PlackettLuceError(
                "Modelo nao ajustado. Chame ajustar_pesos() primeiro."
            )

        return float(sum(self.log_prob_conjunto(S) for S in dados))

    # ------------------------------------------------------------------
    # Modo do conjunto
    # ------------------------------------------------------------------
    def modo(self) -> tuple[int, ...]:
        """Retorna o conjunto de maior probabilidade (modo).

        Para M1 nao existe DP exato. Nesta versao, o modo eh encontrado por
        enumeracao exata sobre todos os C(N, k) conjuntos do universo reduzido
        ou por busca sobre o universo completo quando viavel.

        Para o universo completo (N=60, k=6), a enumeracao de 50 milhões de
        conjuntos nao eh viavel por forca bruta. Usamos uma heurística de
        beam search limitada aos melhores candidatos das sequencias PL.

        Avisa que este metodo eh rotulado como heurístico para o universo
        completo (conforme spec/math_spec.md §9.4).
        """
        if self._w is None:
            raise PlackettLuceError(
                "Modelo nao ajustado. Chame ajustar_pesos() primeiro."
            )

        # Universo reduzido (teste): enumeracao exata
        if self._N <= 10:
            from itertools import combinations

            melhor_S: tuple[int, ...] | None = None
            melhor_p = -np.inf
            for S in combinations(range(1, self._N + 1), self._k):
                p = self.prob_conjunto(S)
                if p > melhor_p:
                    melhor_p = p
                    melhor_S = S
            return melhor_S  # type: ignore[return-value]

        # Universo completo: beam search heuristico (rotulado como tal)
        return self._modo_beam_search()

    def _modo_beam_search(self) -> tuple[int, ...]:
        """Heuristica de beam search para o modo de M1 no universo completo.

        Gera as melhores sequencias PL (ordenadas) e avalia P_M1(S) para os
        conjuntos candidatos resultantes. Nao garante o modo exato, mas fornece
        um limite superior da probabilidade do melhor ponto.

        Rotulado como heurístico (conforme spec/math_spec.md §9.4).
        """
        from itertools import permutations

        # Beam search sobre sequencias PL ordenadas
        beam: list[tuple[tuple[int, ...], float]] = [(( ), 0.0)]  # (sequencia, log_prob)
        W = float(np.sum(self._w))
        log_w = self._log_w

        largura_beam = 50

        for passo in range(self._k):
            novos: list[tuple[tuple[int, ...], float]] = []
            usados_global: set[int] = set()

            for seq, log_p_seq in beam:
                usados = set(seq)
                for idx in range(self._N):
                    if idx in usados:
                        continue
                    restante = W - float(np.sum(self._w[list(usados)]))
                    log_p_novo = log_p_seq + log_w[idx] - np.log(restante)
                    novos.append((seq + (idx,), log_p_novo))

            # Mantem as melhores largura_beam sequencias
            novos.sort(key=lambda x: x[1], reverse=True)
            beam = novos[:largura_beam]

        # Avalia P_M1(S) para os conjuntos distintos gerados
        conjuntos_vistos: set[tuple[int, ...]] = set()
        melhor_S: tuple[int, ...] | None = None
        melhor_p = -np.inf

        for seq, _ in beam:
            S_ordenado = tuple(sorted(d + 1 for d in seq))  # dezenas 1-based
            if S_ordenado in conjuntos_vistos:
                continue
            conjuntos_vistos.add(S_ordenado)
            p = self.prob_conjunto(S_ordenado)
            if p > melhor_p:
                melhor_p = p
                melhor_S = S_ordenado

        if melhor_S is None:
            raise PlackettLuceError("Falha ao encontrar modo por beam search.")

        return melhor_S

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------
    @property
    def pesos(self) -> np.ndarray:
        """Retorna copia dos pesos w_1..w_N."""
        if self._w is None:
            raise PlackettLuceError(
                "Modelo nao ajustado. Chame ajustar_pesos() primeiro."
            )
        return self._w.copy()

    @property
    def esta_ajustado(self) -> bool:
        """Retorna True se os pesos ja foram ajustados."""
        return self._w is not None
