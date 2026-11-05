"""Módulo de baselines para comparação com modelos probabilísticos.

Implementa os baselines definidos na especificação v0.3 (math_spec.md §5):

- B0: Uniforme sobre Ω (piso de "não há sinal").
- B1: 6 dezenas mais frequentes no treino (modelo pontual).
- B2a: Produto posicional empírico com suavização Laplace (M3 com P_j empíricas).
- B2b: Plackett–Luce com pesos proporcionais à frequência global (M1 com w empíricos).
- B3: Concurso anterior (modelo pontual).

Todos os modelos probabilísticos (B0, B2a, B2b) definem uma distribuição
normalizada sobre Ω = {S ⊂ {1..60} : |S| = 6} e suportam log-score de conjunto.
Modelos pontuais (B1, B3) produzem um único conjunto e são avaliados por acertos.

Cobre: ADR-003 (B2a/B2b definidos separadamente), math_spec.md §5 (baselines),
RV-001 a RV-005 (comparação com baselines).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import pandas as pd

from lstm_virada.data import DEZENA_COLS

NUM_CLASSES: Final[int] = 60
NUM_DEZENAS_JOGO: Final[int] = 6
OMEGA_SIZE: Final[int] = 50_063_860  # C(60, 6)
DEFAULT_KAPPA: Final[float] = 1.0  # Suavização Laplace primária para B2a


@dataclass(frozen=True)
class BaselineResult:
    """Resultado de um baseline em um conjunto de teste."""

    name: str
    nll_per_concurso: np.ndarray | None  # -log p(S_t) por concurso (None se pontual)
    mean_nll: float | None
    acertos_por_concurso: np.ndarray  # k_t = |S_chapeu ∩ S_t| por concurso
    medio_acertos: float
    conjunto_predito: tuple[int, ...] | None  # Para modelos pontuais (B1, B3)


class BaselineError(ValueError):
    """Exceção levantada quando um baseline encontra dados inválidos."""


def _validar_concursos(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extrai arrays de concurso e dezenas de um DataFrame validado."""
    if not all(f"d{i}" in df.columns for i in range(1, 7)):
        raise BaselineError("DataFrame não contém colunas d1..d6.")
    if "concurso" not in df.columns:
        raise BaselineError("DataFrame não contém coluna 'concurso'.")

    concursos = df["concurso"].to_numpy(dtype=np.int64)
    dezenas = df[DEZENA_COLS].to_numpy(dtype=np.int32)
    return concursos, dezenas


def _conjunto_para_indices(S: Sequence[int]) -> np.ndarray:
    """Converte um conjunto de 6 dezenas (1..60) em índices zero-based (0..59)."""
    return np.array(sorted(S), dtype=np.int64) - 1


# ==============================================================================
# B0 — Uniforme sobre Ω
# ==============================================================================
class B0Uniforme:
    """Baseline B0: distribuição uniforme sobre todos os conjuntos de 6 dezenas.

    p(S) = 1 / C(60, 6) para todo S ∈ Ω.

    É o piso de "não há sinal" (H1). Não tem parâmetros e não usa histórico.
    """

    def __init__(self) -> None:
        self.name = "B0 (uniforme)"
        self._log_p = -np.log(OMEGA_SIZE)

    def log_score_conjunto(self, dezenas_teste: np.ndarray) -> np.ndarray:
        """Retorna log p(S_t) para cada concurso de teste (mesmo valor para todos)."""
        n = len(dezenas_teste)
        return np.full(n, self._log_p, dtype=np.float64)

    def predizer_conjunto(self) -> tuple[int, ...]:
        """Retorna 6 dezenas sorteadas uniformemente sem reposição (não determinístico)."""
        rng = np.random.default_rng()
        return tuple(int(d) for d in sorted(rng.choice(NUM_CLASSES, size=NUM_DEZENAS_JOGO, replace=False) + 1))

    def avaliar(
        self,
        dezenas_teste: np.ndarray,
    ) -> BaselineResult:
        """Avalia B0 no conjunto de teste."""
        nll = self.log_score_conjunto(dezenas_teste)
        # B0 não tem conjunto predito fixo; acertos são avaliados via amostragem
        # ou deixados para portfólio. Aqui retornamos acertos = 0 (não aplicável).
        acertos = np.zeros(len(dezenas_teste), dtype=np.int32)
        return BaselineResult(
            name=self.name,
            nll_per_concurso=nll,
            mean_nll=float(np.mean(nll)),
            acertos_por_concurso=acertos,
            medio_acertos=0.0,
            conjunto_predito=None,
        )


# ==============================================================================
# B1 — 6 Mais Frequentes (pontual)
# ==============================================================================
class B1MaisFrequentes:
    """Baseline B1: as 6 dezenas de maior frequência no conjunto de treino.

    Modelo pontual: produz um único conjunto. Não tem NLL (sem suavização
    explícita). Desempate por menor dezena.
    """

    def __init__(self) -> None:
        self.name = "B1 (mais frequentes)"
        self._conjunto: tuple[int, ...] | None = None

    def ajustar(self, dezenas_treino: np.ndarray) -> None:
        """Ajusta as 6 dezenas mais frequentes no treino.

        Args:
            dezenas_treino: array (N_treino, 6) com dezenas 1..60.
        """
        contagens = np.zeros(NUM_CLASSES + 1, dtype=np.int64)  # índice 1..60
        for row in dezenas_treino:
            for dez in row:
                contagens[int(dez)] += 1

        # Top 6 por frequência; desempate por menor dezena (ordem natural de argsort estável)
        dezs = np.arange(1, NUM_CLASSES + 1)
        ordem = np.argsort(-contagens[1:], kind="stable")  # decrescente, estável
        top6 = dezs[ordem[:NUM_DEZENAS_JOGO]]
        self._conjunto = tuple(int(d) for d in sorted(top6))

    def conjunto_predito(self) -> tuple[int, ...]:
        if self._conjunto is None:
            raise BaselineError("B1 não foi ajustado. Chame ajustar() primeiro.")
        return self._conjunto

    def avaliar(
        self,
        dezenas_teste: np.ndarray,
    ) -> BaselineResult:
        """Avalia B1 no conjunto de teste por acertos."""
        if self._conjunto is None:
            raise BaselineError("B1 não foi ajustado. Chame ajustar() primeiro.")

        conjunto_set = set(self._conjunto)
        acertos = np.array(
            [len(conjunto_set & set(row.tolist())) for row in dezenas_teste],
            dtype=np.int32,
        )
        return BaselineResult(
            name=self.name,
            nll_per_concurso=None,
            mean_nll=None,
            acertos_por_concurso=acertos,
            medio_acertos=float(np.mean(acertos)),
            conjunto_predito=self._conjunto,
        )


# ==============================================================================
# B2a — Produto Posicional Empírico com Suavização (M3)
# ==============================================================================
class B2aProdutoPosicional:
    """Baseline B2a: M3 com marginais posicionais empíricas suavizadas.

    P_j(c) = (n_{j,c} + κ) / (N_treino + 55·κ) para c ∈ supp_j = {j, ..., 60-6+j}.
    Z = soma sobre todos os conjuntos crescentes de ∏_j P_j(d_j), calculado por DP.

    κ = 1.0 (Laplace) é o valor primário; sem suavização, NLL seria infinita
    para dezenas nunca observadas em uma posição.
    """

    def __init__(self, kappa: float = DEFAULT_KAPPA) -> None:
        if kappa <= 0:
            raise BaselineError(f"kappa deve ser > 0, recebido: {kappa}.")
        self.kappa = float(kappa)
        self.name = f"B2a (produto posicional, κ={kappa})"
        self._P_j: np.ndarray | None = None  # shape (6, 60), apenas supp_j preenchido
        self._Z: float | None = None

    def ajustar(self, dezenas_treino: np.ndarray) -> None:
        """Ajusta as marginais posicionais empíricas com suavização Laplace.

        Args:
            dezenas_treino: array (N_treino, 6) com dezenas 1..60 ordenadas.
        """
        N_treino = len(dezenas_treino)
        kappa = self.kappa

        # Contagens n_{j,c} para cada posição j e dezena c
        contagens = np.zeros((NUM_DEZENAS_JOGO, NUM_CLASSES + 1), dtype=np.float64)
        for row in dezenas_treino:
            for j, dez in enumerate(row):
                contagens[j, int(dez)] += 1.0

        # Suavização: P_j(c) = (n_{j,c} + κ) / (N_treino + 55·κ) para c ∈ supp_j
        # |supp_j| = 55 para todo j
        P = np.zeros((NUM_DEZENAS_JOGO, NUM_CLASSES), dtype=np.float64)
        denominador = N_treino + 55.0 * kappa
        for j in range(NUM_DEZENAS_JOGO):
            for c in range(1, NUM_CLASSES + 1):
                # supp_j (1-based) = {j+1, ..., 60 - (6 - j - 1)} = {j+1, ..., 55 + j}
                # Exemplo: j=0 -> d1 em 1..55; j=5 -> d6 em 6..60.
                c_min = j + 1
                c_max = 55 + j  # 60 - (6 - j - 1)
                if c_min <= c <= c_max:
                    P[j, c - 1] = (contagens[j, c] + kappa) / denominador
                else:
                    P[j, c - 1] = 0.0

        self._P_j = P
        self._Z = self._calcular_Z(P)

    def _calcular_Z(self, P: np.ndarray) -> float:
        """Calcula Z = soma sobre conjuntos crescentes de ∏_j P_j(d_j) por DP.

        DP em O(6 · 60):
        F[1][c] = P_1(c)
        F[j][c] = P_j(c) * soma_{c' < c} F[j-1][c']
        Z = soma_c F[6][c]
        """
        # F[j, c] para j=0..5, c=0..59 (zero-based)
        F = np.zeros((NUM_DEZENAS_JOGO, NUM_CLASSES), dtype=np.float64)

        # Cabeça 0: d1 pode ser 1..55 (índice 0..54)
        for c in range(0, NUM_CLASSES - NUM_DEZENAS_JOGO + 1):
            F[0, c] = P[0, c]

        # Cabeças 1 a 5
        for j in range(1, NUM_DEZENAS_JOGO):
            min_c = j  # d_j >= j+1 em 1-based -> índice >= j
            max_c = NUM_CLASSES - (NUM_DEZENAS_JOGO - j)  # índice max
            prefix_sum = 0.0
            for c in range(min_c, max_c + 1):
                # Acumula F[j-1, c-1] (predecessores estritamente menores)
                prefix_sum += F[j - 1, c - 1]
                F[j, c] = P[j, c] * prefix_sum

        Z = float(np.sum(F[NUM_DEZENAS_JOGO - 1, :]))
        if Z <= 0:
            raise BaselineError("Z calculado é zero; verifique as marginais P_j.")
        return Z

    def log_score_conjunto(self, dezenas_teste: np.ndarray) -> np.ndarray:
        """Retorna log p(S_t) para cada concurso de teste."""
        if self._P_j is None or self._Z is None:
            raise BaselineError("B2a não foi ajustado. Chame ajustar() primeiro.")

        n = len(dezenas_teste)
        log_p = np.zeros(n, dtype=np.float64)
        log_Z = np.log(self._Z)

        for i, row in enumerate(dezenas_teste):
            # Produto ∏_j P_j(d_j)
            log_q = 0.0
            for j, dez in enumerate(row):
                p_val = self._P_j[j, int(dez) - 1]
                if p_val <= 0.0:
                    log_q = -np.inf
                    break
                log_q += np.log(p_val)
            log_p[i] = log_q - log_Z

        return log_p

    def avaliar(
        self,
        dezenas_teste: np.ndarray,
    ) -> BaselineResult:
        """Avalia B2a no conjunto de teste por log-score de conjunto."""
        nll = -self.log_score_conjunto(dezenas_teste)
        # B2a é probabilístico; não tem conjunto predito único para acertos diretos.
        acertos = np.zeros(len(dezenas_teste), dtype=np.int32)
        return BaselineResult(
            name=self.name,
            nll_per_concurso=nll,
            mean_nll=float(np.mean(nll)),
            acertos_por_concurso=acertos,
            medio_acertos=0.0,
            conjunto_predito=None,
        )


# ==============================================================================
# B2b — Plackett–Luce com Pesos Empíricos (M1)
# ==============================================================================
class B2bPLFrequenciaEmpirica:
    """Baseline B2b: M1 (Plackett–Luce) com w_i ∝ frequência global da dezena i.

    P(S) = soma sobre permutações π de ∏_j w_{π_j} / (W - soma_{l<j} w_{π_l}).

    Compartilha a mesma família probabilística de M1, isolando a diferença
    entre "usar frequência diretamente" e "ajustar pesos pela verossimilhança".
    """

    def __init__(self) -> None:
        self.name = "B2b (PL com frequência empírica)"
        self._w: np.ndarray | None = None  # pesos w_1..w_60

    def ajustar(self, dezenas_treino: np.ndarray) -> None:
        """Ajusta pesos proporcionais à frequência global de cada dezena.

        Args:
            dezenas_treino: array (N_treino, 6) com dezenas 1..60.
        """
        frequencias = np.zeros(NUM_CLASSES + 1, dtype=np.float64)
        total_sorteios = 0
        for row in dezenas_treino:
            total_sorteios += 1
            for dez in row:
                frequencias[int(dez)] += 1.0

        # w_i ∝ frequência; todas as 60 dezenas aparecem em algum concurso,
        # então w_i > 0 para todo i (evita suporte zero).
        w = frequencias[1:]  # índice 0..59 = dezena 1..60
        if np.any(w <= 0):
            raise BaselineError("Alguma dezena não aparece no treino; B2b requer w_i > 0.")

        # Normaliza para média geométrica 1 (identificabilidade de escala)
        w = w / np.mean(w)
        self._w = w

    def log_score_conjunto(self, dezenas_teste: np.ndarray) -> np.ndarray:
        """Retorna log p(S_t) para cada concurso de teste via soma sobre permutações.

        Para cada conjunto S, calcula:
        log P(S) = soma_{i∈S} log w_i + logsumexp_π [ -soma_j log(W - soma_{l<j} w_{π_l}) ]
        """
        if self._w is None:
            raise BaselineError("B2b não foi ajustado. Chame ajustar() primeiro.")

        w = self._w
        W = float(np.sum(w))
        n = len(dezenas_teste)
        log_p = np.zeros(n, dtype=np.float64)

        # Pré-calcula log w para eficiência
        log_w = np.log(w)

        import itertools

        for i, row in enumerate(dezenas_teste):
            S = tuple(int(d) - 1 for d in row)  # zero-based
            soma_log_w = float(np.sum([log_w[idx] for idx in S]))

            # Soma sobre as 720 permutações de S
            termos_log = []
            for pi in itertools.permutations(S):
                denom_sum = 0.0
                log_denom_soma = 0.0
                restante = W
                for idx in pi:
                    log_denom_soma += np.log(restante)
                    restante -= w[idx]
                termos_log.append(-log_denom_soma)

            # logsumexp dos termos
            max_termo = max(termos_log)
            logsumexp = max_termo + np.log(
                sum(np.exp(t - max_termo) for t in termos_log)
            )
            log_p[i] = soma_log_w + logsumexp

        return log_p

    def avaliar(
        self,
        dezenas_teste: np.ndarray,
    ) -> BaselineResult:
        """Avalia B2b no conjunto de teste por log-score de conjunto."""
        nll = -self.log_score_conjunto(dezenas_teste)
        acertos = np.zeros(len(dezenas_teste), dtype=np.int32)
        return BaselineResult(
            name=self.name,
            nll_per_concurso=nll,
            mean_nll=float(np.mean(nll)),
            acertos_por_concurso=acertos,
            medio_acertos=0.0,
            conjunto_predito=None,
        )


# ==============================================================================
# B3 — Concurso Anterior (pontual)
# ==============================================================================
class B3ConcursoAnterior:
    """Baseline B3: repete as 6 dezenas do concurso imediatamente anterior ao alvo.

    Modelo pontual: produz um único conjunto por alvo. Não tem NLL.
    """

    def __init__(self) -> None:
        self.name = "B3 (concurso anterior)"
        self._concurso_para_dezenas: dict[int, tuple[int, ...]] = {}

    def ajustar(self, concursos: np.ndarray, dezenas: np.ndarray) -> None:
        """Indexa dezenas por número de concurso.

        Args:
            concursos: array (N,) com números de concurso.
            dezenas: array (N, 6) com dezenas 1..60.
        """
        self._concurso_para_dezenas = {}
        for conc, row in zip(concursos.tolist(), dezenas.tolist()):
            self._concurso_para_dezenas[int(conc)] = tuple(int(d) for d in row)

    def predizer_para_concurso(self, concurso_alvo: int) -> tuple[int, ...]:
        """Retorna as dezenas do concurso imediatamente anterior ao alvo."""
        concurso_anterior = concurso_alvo - 1
        if concurso_anterior not in self._concurso_para_dezenas:
            raise BaselineError(
                f"B3: concurso anterior {concurso_anterior} não encontrado no histórico."
            )
        return self._concurso_para_dezenas[concurso_anterior]

    def avaliar(
        self,
        concursos_teste: np.ndarray,
        dezenas_teste: np.ndarray,
    ) -> BaselineResult:
        """Avalia B3 no conjunto de teste por acertos."""
        acertos = []
        conjunto_predito = None
        for conc, row in zip(concursos_teste.tolist(), dezenas_teste.tolist()):
            try:
                pred = self.predizer_para_concurso(conc)
                conjunto_set = set(pred)
                real_set = set(row)
                acertos.append(len(conjunto_set & real_set))
                conjunto_predito = pred
            except BaselineError:
                acertos.append(0)

        acertos_arr = np.array(acertos, dtype=np.int32)
        return BaselineResult(
            name=self.name,
            nll_per_concurso=None,
            mean_nll=None,
            acertos_por_concurso=acertos_arr,
            medio_acertos=float(np.mean(acertos_arr)) if len(acertos_arr) > 0 else 0.0,
            conjunto_predito=conjunto_predito,
        )


# ==============================================================================
# Fábrica de Baselines
# ==============================================================================
def criar_baseline(nome: str, **kwargs) -> object:
    """Cria um baseline pelo nome.

    Args:
        nome: "B0", "B1", "B2a", "B2b" ou "B3".
        **kwargs: Argumentos adicionais (ex: kappa para B2a).

    Returns:
        Instância do baseline correspondente.

    Raises:
        BaselineError: Se o nome não for reconhecido.
    """
    nome = nome.strip().upper()
    if nome == "B0":
        return B0Uniforme()
    if nome == "B1":
        return B1MaisFrequentes()
    if nome == "B2A":
        return B2aProdutoPosicional(kappa=kwargs.get("kappa", DEFAULT_KAPPA))
    if nome == "B2B":
        return B2bPLFrequenciaEmpirica()
    if nome == "B3":
        return B3ConcursoAnterior()
    raise BaselineError(
        f"Baseline desconhecido: '{nome}'. Use um de: B0, B1, B2a, B2b, B3."
    )


def todos_os_baselines() -> dict[str, object]:
    """Retorna todos os baselines primários (B0, B1, B2a, B2b, B3)."""
    return {
        "B0": B0Uniforme(),
        "B1": B1MaisFrequentes(),
        "B2a": B2aProdutoPosicional(kappa=DEFAULT_KAPPA),
        "B2b": B2bPLFrequenciaEmpirica(),
        "B3": B3ConcursoAnterior(),
    }
