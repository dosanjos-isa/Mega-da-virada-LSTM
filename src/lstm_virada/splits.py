"""Módulo de splitters temporais para walk-forward e inferência.

Responsabilidades:
- Dividir o histórico em treino/teste respeitando a causalidade temporal
  (sem vazamento: treino contém apenas concursos estritamente anteriores ao
  concurso de corte).
- Implementar walk-forward por edição da Mega da Virada (2015-2025): para
  cada edição Y, treina-se com concursos anteriores à Virada Y e testa-se
  exclusivamente na Virada Y.
- Rejeitar concursos sentinela do futuro (nenhuma janela pode conter
  concurso >= as_of).
- Extrair a janela de inferência mais recente para predição prospectiva.

Cobre: RT-006 (vazamento temporal), RV-001 (walk-forward por ano de Virada),
RV-002 (avaliação secundária nos últimos 300 concursos), RV-003 (registro
prospectivo antes do sorteio).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterator, Sequence

import numpy as np
import pandas as pd

from lstm_virada.data import DEZENA_COLS

# Edições da Mega da Virada disponíveis no dataset processado (P0-02).
VIRADA_ANOS: Final[tuple[int, ...]] = tuple(range(2009, 2026))

# Walk-forward secundário: edições 2015 a 2025 (11 folds).
WALK_FORWARD_VIRADA_ANOS: Final[tuple[int, ...]] = tuple(range(2015, 2026))


class SplitError(ValueError):
    """Exceção levantada quando o splitter de dados encontra dados inválidos
    ou tentativa de vazamento temporal."""


@dataclass(frozen=True)
class TemporalSplit:
    """Resultado de uma divisão temporal treino/teste."""

    train_indices: np.ndarray  # índices posicionais (0-based) de treino
    test_indices: np.ndarray   # índices posicionais (0-based) de teste
    as_of_concurso: int        # concurso de corte (primeiro concurso do teste)
    train_concursos: np.ndarray  # números de concurso de treino
    test_concursos: np.ndarray   # números de concurso de teste


@dataclass(frozen=True)
class WalkForwardFold:
    """Um fold do walk-forward por edição da Mega da Virada."""

    virada_ano: int
    concurso_virada: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_concursos: np.ndarray
    test_concursos: np.ndarray


class TemporalSplitter:
    """Splitter temporal simples: divide por concurso de corte (as_of).

    Regras:
    - Treino: concursos estritamente menores que `as_of_concurso`.
    - Teste: concursos maiores ou iguais a `as_of_concurso`.
    - Se `as_of_concurso` for maior que o último concurso disponível, levanta
      SplitError (sentinela futura rejeitada).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.sort_values("concurso").reset_index(drop=True)
        self._concursos = self._df["concurso"].to_numpy(dtype=np.int64)
        self._last_concurso = int(self._concursos[-1])

    def split(self, as_of_concurso: int) -> TemporalSplit:
        """Divide o histórico em treino/teste no concurso de corte.

        Args:
            as_of_concurso: Primeiro concurso do conjunto de teste. Todo
                concurso < as_of_concurso vai para treino.

        Returns:
            TemporalSplit com índices posicionais e números de concurso.

        Raises:
            SplitError: Se `as_of_concurso` for maior que o último concurso
                disponível (sentinela futura) ou for inválido.
        """
        if not isinstance(as_of_concurso, (int, np.integer)):
            raise SplitError(
                f"as_of_concurso deve ser inteiro, recebido: {type(as_of_concurso).__name__}."
            )
        as_of = int(as_of_concurso)

        if as_of < 1:
            raise SplitError(f"as_of_concurso deve ser >= 1, recebido: {as_of}.")

        # Sentinela futura: concurso de corte além do último disponível.
        if as_of > self._last_concurso:
            raise SplitError(
                f"Sentinela futura rejeitada: as_of_concurso={as_of} excede o "
                f"último concurso disponível ({self._last_concurso})."
            )

        # Máscaras booleanas por concurso (não por índice posicional, para
        # garantir causalidade mesmo se houver lacunas numerais).
        train_mask = self._concursos < as_of
        test_mask = self._concursos >= as_of

        train_indices = np.nonzero(train_mask)[0]
        test_indices = np.nonzero(test_mask)[0]

        return TemporalSplit(
            train_indices=train_indices,
            test_indices=test_indices,
            as_of_concurso=as_of,
            train_concursos=self._concursos[train_indices],
            test_concursos=self._concursos[test_indices],
        )

    @property
    def last_concurso(self) -> int:
        """Último concurso disponível no histórico."""
        return self._last_concurso

    @property
    def concursos(self) -> np.ndarray:
        """Números de concurso ordenados."""
        return self._concursos.copy()


class WalkForwardSplitter:
    """Walk-forward por edição da Mega da Virada.

    Para cada edição Y em `virada_anos`:
    - Treino: todos os concursos com número estritamente menor que o concurso
      da Virada Y.
    - Teste: exclusivamente o concurso da Virada Y.

    Isso preserva a causalidade: nunca se usa dados da Virada Y (ou posteriores)
    para treinar o fold da Virada Y.

    Edições de 2009 a 2014 aparecem apenas como treino no primeiro fold
    (2015), nunca como alvo de teste (poder estatístico insuficiente).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        virada_anos: Sequence[int] = WALK_FORWARD_VIRADA_ANOS,
    ) -> None:
        self._df = df.sort_values("concurso").reset_index(drop=True)
        self._concursos = self._df["concurso"].to_numpy(dtype=np.int64)
        self._is_virada = (
            self._df["is_virada"].to_numpy(dtype=np.int32)
            if "is_virada" in self._df.columns
            else np.zeros(len(self._df), dtype=np.int32)
        )
        self._virada_anos = tuple(virada_anos)

        # Mapeia concurso -> virada_ano para as Viradas presentes no dataset.
        self._concurso_para_virada_ano: dict[int, int] = {}
        for _, row in self._df.iterrows():
            conc = int(row["concurso"])
            if int(row["is_virada"]) == 1 and "virada_ano" in self._df.columns:
                v_ano = int(row["virada_ano"])
                if v_ano > 0:
                    self._concurso_para_virada_ano[conc] = v_ano

        # Fallback: se virada_ano não estiver disponível, usa a ordem das
        # Viradas encontradas no dataset (concorrentes com is_virada=1).
        if not self._concurso_para_virada_ano:
            virada_concursos = self._concursos[self._is_virada == 1]
            for i, conc in enumerate(virada_concursos):
                # Atribui anos sequenciais como fallback (não ideal, mas funcional)
                self._concurso_para_virada_ano[int(conc)] = 2009 + i

    def splits(self) -> list[WalkForwardFold]:
        """Gera todos os folds do walk-forward.

        Returns:
            Lista de WalkForwardFold, um por edição de Virada solicitada que
            esteja presente no dataset.

        Raises:
            SplitError: Se nenhuma edição de Virada solicitada for encontrada.
        """
        folds: list[WalkForwardFold] = []

        for ano in self._virada_anos:
            # Encontra o concurso da Virada desse ano.
            concurso_virada = None
            for conc, v_ano in self._concurso_para_virada_ano.items():
                if v_ano == ano:
                    concurso_virada = conc
                    break

            if concurso_virada is None:
                # Edição solicitada não está no dataset; pula silenciosamente.
                continue

            split_result = TemporalSplitter(self._df).split(concurso_virada)

            # O teste do walk-forward é exclusivamente o concurso da Virada Y,
            # não todos os concursos a partir dela (causalidade estrita).
            virada_idx = np.nonzero(self._concursos == concurso_virada)[0]
            if virada_idx.size == 0:
                continue

            folds.append(
                WalkForwardFold(
                    virada_ano=ano,
                    concurso_virada=concurso_virada,
                    train_indices=split_result.train_indices,
                    test_indices=virada_idx,
                    train_concursos=split_result.train_concursos,
                    test_concursos=self._concursos[virada_idx],
                )
            )

        if not folds:
            raise SplitError(
                "Nenhuma edição de Virada solicitada foi encontrada no dataset. "
                f"Solicitadas: {self._virada_anos}."
            )

        return folds

    def fold_for_year(self, virada_ano: int) -> WalkForwardFold:
        """Retorna o fold de uma edição específica da Virada.

        Raises:
            SplitError: Se a edição não estiver disponível.
        """
        for fold in self.splits():
            if fold.virada_ano == virada_ano:
                return fold
        raise SplitError(
            f"Edição da Virada {virada_ano} não disponível para walk-forward."
        )


@dataclass(frozen=True)
class InferenceWindow:
    """Janela de inferência extraída para predição prospectiva."""

    x_seq: np.ndarray  # shape (1, W, 61)
    x_static: np.ndarray  # shape (1, 1)
    last_concurso: int
    expected_target_concurso: int


class InferenceWindowExtractor:
    """Extrai a janela temporal mais recente para inferência operacional.

    A janela contém os W concursos imediatamente anteriores ao último concurso
    disponível no histórico. Não há concurso-alvo (inferência prospectiva).
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 20) -> None:
        if window_size < 1:
            raise SplitError(f"window_size deve ser positivo, recebido: {window_size}.")
        self._df = df.sort_values("concurso").reset_index(drop=True)
        self._window_size = int(window_size)

    def extract(self, alvo_virada: int = 1) -> InferenceWindow:
        """Extrai a janela de inferência mais recente.

        Args:
            alvo_virada: Flag de contexto estático para o sorteio a prever
                (1 = Mega da Virada, 0 = regular).

        Returns:
            InferenceWindow com x_seq, x_static, last_concurso e
            expected_target_concurso.

        Raises:
            SplitError: Se o histórico tiver menos concursos que window_size.
        """
        total_rows = len(self._df)
        if total_rows < self._window_size:
            raise SplitError(
                f"Histórico com {total_rows} concursos é insuficiente para "
                f"janela de inferência de tamanho {self._window_size}."
            )

        recent_df = self._df.iloc[-self._window_size :]
        concursos = recent_df["concurso"].to_numpy(dtype=np.int64)
        last_concurso = int(concursos[-1])

        # Constrói features multi-hot (W, 61) sem is_virada por passo
        # (P0-06: is_virada só aparece no contexto estático x_static).
        dezenas_cols = [f"d{i}" for i in range(1, 7)]
        dezenas = recent_df[dezenas_cols].to_numpy(dtype=np.int32)

        features = np.zeros((self._window_size, 61), dtype=np.float32)
        for r in range(self._window_size):
            for dez in dezenas[r]:
                features[r, int(dez) - 1] = 1.0
            # Índice 60 (is_virada por passo) é deixado em 0.0 por construção.
            # A flag de Virada vai apenas no contexto estático x_static.

        return InferenceWindow(
            x_seq=np.expand_dims(features, axis=0),  # (1, W, 61)
            x_static=np.array([[float(alvo_virada)]], dtype=np.float32),  # (1, 1)
            last_concurso=last_concurso,
            expected_target_concurso=last_concurso + 1,
        )
