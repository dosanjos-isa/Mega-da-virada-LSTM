"""Módulo de geração de jogos da Mega da Virada.

Implementa a geração de conjuntos de 6 dezenas usando modelos probabilísticos
ou baselines, com filtro de diversidade e metadados imutáveis.

Cobre: Geração de jogos, amostragem exata, diversidade de portfólio, metadados imutáveis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Final, Sequence

import hashlib
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from lstm_virada.data import DEZENA_COLS
from lstm_virada.models import (
    PlackettLuce, FisherModel, PositionalModel, SeqPLModel, LSTMSeisCabecas,
    VARIANTE_SET, VARIANTE_CCE,
)

NUM_CLASSES: Final[int] = 60
NUM_DEZENAS_JOGO: Final[int] = 6
DEFAULT_K: Final[int] = 10
DEFAULT_MAX_SHARED: Final[int] = 4
DEFAULT_MAX_ATTEMPTS_FACTOR: Final[int] = 100


class GenerationError(ValueError):
    """Exceção levantada quando a geração de jogos encontra dados inválidos."""
    pass


@dataclass
class GenerationMetadata:
    """Metadados imutáveis de uma geração."""
    timestamp: str
    model_name: str
    model_hash: str | None
    dataset_hash: str | None
    config_hash: str | None
    commit: str | None
    seed: int
    strategy: str
    K: int
    diversity: int | None
    attempts: int
    saturated: bool
    games: list[list[int]]
    temperature: float = 1.0


@dataclass
class GenerationResult:
    """Resultado de uma geração de jogos."""
    games: list[list[int]]
    metadata: GenerationMetadata


class Generator:
    """Gerador de jogos da Mega da Virada.

    Suporta amostragem exata de M0 (uniforme), M1 (Plackett-Luce),
    M2 (Fisher) e M3 (Produto Posicional). Para M4/M5, usa os pesos
    produzidos pela rede neural condicionada na janela de inferencia.
    """

    def __init__(
        self,
        model_name: str = "M0",
        model=None,
        seed: int = 42,
        temperature: float = 1.0,
        janela_inferencia: np.ndarray | None = None,
    ):
        self.model_name = model_name.upper()
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.janela_inferencia = janela_inferencia
        # Cache de distribuicoes por posicao para modelos neurais (M4/M5)
        self._distribuicoes_posicao: list[dict[int, float]] | None = None

    def generate(
        self,
        K: int = DEFAULT_K,
        max_shared_numbers: int = DEFAULT_MAX_SHARED,
        max_attempts_factor: int = DEFAULT_MAX_ATTEMPTS_FACTOR,
        dataset_hash: str | None = None,
        config_hash: str | None = None,
        commit: str | None = None,
    ) -> GenerationResult:
        """Gera K jogos válidos.

        Args:
            K: Número de jogos a gerar.
            max_shared_numbers: Máximo de dezenas compartilhadas entre
                quaisquer dois jogos do portfólio.
            max_attempts_factor: Fator multiplicativo para o limite de
                tentativas antes de declarar saturação.
            dataset_hash: Hash do dataset usado.
            config_hash: Hash da configuração usada.
            commit: Hash do commit Git.

        Returns:
            GenerationResult com os jogos e metadados.
        """
        if K <= 0:
            raise GenerationError("K deve ser maior que zero.")
        if max_shared_numbers < 0 or max_shared_numbers > NUM_DEZENAS_JOGO:
            raise GenerationError(
                f"max_shared_numbers deve estar entre 0 e {NUM_DEZENAS_JOGO}."
            )

        max_attempts = max_attempts_factor * K
        games: list[list[int]] = []
        attempts = 0

        while len(games) < K and attempts < max_attempts:
            attempts += 1
            game = self._sample_one_game()
            if self._is_diverse(game, games, max_shared_numbers):
                games.append(sorted(game))

        saturated = len(games) < K

        metadata = GenerationMetadata(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            model_name=self.model_name,
            model_hash=None,  # Preenchido pelo chamador se houver modelo
            dataset_hash=dataset_hash,
            config_hash=config_hash,
            commit=commit,
            seed=self.seed,
            strategy=self.model_name,
            K=K,
            diversity=max_shared_numbers,
            attempts=attempts,
            saturated=saturated,
            games=games,
            temperature=self.temperature,
        )

        return GenerationResult(games=games, metadata=metadata)

    def _sample_one_game(self) -> list[int]:
        """Amostra um único conjunto de 6 dezenas."""
        if self.model_name == "M0":
            return self._sample_uniform()
        elif self.model_name in ("M1", "M2", "M3"):
            return self._sample_from_model_weights()
        elif self.model_name in ("M4", "M5"):
            return self._sample_from_neural_model()
        else:
            raise GenerationError(f"Modelo não suportado: {self.model_name}")

    def _obter_distribuicoes_posicao(
        self,
    ) -> list[dict[int, float]]:
        """Obtém distribuições de probabilidade por posição para modelos neurais.

        Para M4 (SeqPLModel): faz forward pass pela rede para obter
        pesos w_{j,i} por passo j e dezena i.
        Para M5 (LSTMSeisCabecas): obtém as saídas dos 6 softmax heads.

        Returns:
            Lista de k dicionários {dezena: probabilidade}, um por posição.
        """
        if self.model is None:
            raise GenerationError(
                "Modelo neural não fornecido para obtenção de distribuições."
            )
        if self.janela_inferencia is None:
            raise GenerationError(
                "Janela de inferência não fornecida para modelo neural."
            )

        # Normaliza a janela para (1, W, 61)
        janela = np.asarray(self.janela_inferencia, dtype=np.float32)
        if janela.ndim == 2:
            janela = janela[np.newaxis, :, :]  # (1, W, 61)

        n_passos = NUM_DEZENAS_JOGO
        n_dezenas = self.model._N

        distribuicoes: list[dict[int, float]] = []

        if isinstance(self.model, SeqPLModel):
            # Forward pass pela rede SeqPL: saída (batch, k, N)
            rede = self.model._obter_ou_construir_rede()
            alvos_virada = np.zeros((1, 1), dtype=np.float32)
            pesos = rede.predict(
                [janela, alvos_virada], verbose=0
            )  # (1, k, N)
            pesos = pesos[0]  # (k, N)

            for j in range(n_passos):
                w_j = pesos[j]  # (N,)
                # Normaliza para somar 1 sobre as dezenas disponíveis
                w_j = np.maximum(w_j, 1e-12)
                w_j = w_j / np.sum(w_j)
                dist = {int(i + 1): float(w_j[i]) for i in range(n_dezenas)}
                distribuicoes.append(dist)

        elif isinstance(self.model, LSTMSeisCabecas):
            # Forward pass pela rede LSTM de 6 cabeças
            rede = self.model._obter_ou_construir_rede()
            alvos_virada = np.zeros((1, 1), dtype=np.float32)
            previsoes = rede.predict(
                [janela, alvos_virada], verbose=0
            )  # (1, k, N)
            previsoes = previsoes[0]  # (k, N)

            for j in range(n_passos):
                p_j = previsoes[j]  # (N,)
                p_j = np.maximum(p_j, 1e-12)
                p_j = p_j / np.sum(p_j)
                dist = {int(i + 1): float(p_j[i]) for i in range(n_dezenas)}
                distribuicoes.append(dist)

        else:
            raise GenerationError(
                f"Tipo de modelo neural não suportado: {type(self.model).__name__}"
            )

        self._distribuicoes_posicao = distribuicoes
        return distribuicoes

    def _sample_uniform(self) -> list[int]:
        """Amostra 6 dezenas uniformes sem reposição."""
        return sorted(self.rng.choice(NUM_CLASSES, size=NUM_DEZENAS_JOGO, replace=False) + 1)

    def _sample_from_model_weights(self) -> list[int]:
        """Amostra usando pesos PL/Fisher (M1, M2).

        Para M3, usa _sample_from_positional_model.
        Para M2 (Fisher), usa amostragem recursiva exata por DP (math_spec §9.3).
        Para M1 (PL), usa amostragem sequencial condicionada (exata para PL).
        """
        if self.model_name == "M3":
            return self._sample_from_positional_model()

        # Obtém pesos do modelo (M1, M2)
        if hasattr(self.model, 'pesos'):
            w = self.model.pesos  # array numpy de 60 pesos
        else:
            raise GenerationError(
                f"Modelo {self.model_name} não expõe pesos para amostragem."
            )

        w = np.array(w, dtype=np.float64)
        if np.any(w <= 0):
            raise GenerationError("Pesos devem ser estritamente positivos.")

        if self.model_name == "M2":
            return self._sample_from_fisher(w)

        # M1: amostragem sequencial PL (exata para Plackett–Luce)
        w = w / np.sum(w)
        n_dezenas_modelo = len(w)
        available_indices = list(range(n_dezenas_modelo))
        available_weights = w.copy()
        chosen = []

        for _ in range(NUM_DEZENAS_JOGO):
            if len(available_indices) == 0:
                break
            total = available_weights.sum()
            if total <= 0:
                break
            probs = available_weights / total
            probs = probs / probs.sum()
            idx = self.rng.choice(len(available_indices), p=probs)
            chosen_idx = available_indices[idx]
            chosen.append(chosen_idx + 1)
            available_indices.pop(idx)
            available_weights = np.delete(available_weights, idx)

        return sorted(chosen)

    def _sample_from_fisher(self, w: np.ndarray) -> list[int]:
        """Amostragem recursiva exata para M2 (Fisher).

        Implementa math_spec.md §9.3 / Apêndice A.5:
        Para um conjunto de tamanho k restante e pesos w:
        1. Calcular e_{k-1}(w_{-i}) para cada i por DP.
        2. Amostrar o primeiro elemento i com probabilidade
           π_i = w_i * e_{k-1}(w_{-i}) / (k * e_k(w)).
        3. Remover i, decrementar k, repetir.

        Isso produz cada conjunto S com probabilidade exata
        prod_{i in S} w_i / e_k(w).
        """
        N = len(w)
        k = self.model._k if self.model is not None else NUM_DEZENAS_JOGO
        w = w / np.sum(w)  # normaliza para estabilidade

        def calcular_e_k(w_sub: np.ndarray, k_sub: int) -> float:
            """Calcula e_{k_sub}(w_sub) por programação dinâmica em O(N*k)."""
            e = np.zeros(k_sub + 1, dtype=np.float64)
            e[0] = 1.0
            n_elem = len(w_sub)
            for i in range(n_elem):
                wi = w_sub[i]
                max_j = min(i + 1, k_sub)
                for j in range(max_j, 0, -1):
                    e[j] += wi * e[j - 1]
            return float(e[k_sub])

        chosen = []
        remaining_indices = list(range(N))
        remaining_weights = w.copy()
        k_restante = k

        while k_restante > 0 and len(remaining_indices) > 0:
            e_k = calcular_e_k(remaining_weights, k_restante)
            if e_k <= 0:
                break

            # Para cada i restante, calcular e_{k-1}(w_{-i})
            pesos_amostragem = []
            for idx in range(len(remaining_indices)):
                wi = remaining_weights[idx]
                w_sem_i = np.delete(remaining_weights, idx)
                e_km1 = calcular_e_k(w_sem_i, k_restante - 1)
                pi = wi * e_km1 / (k_restante * e_k)
                pesos_amostragem.append(pi)

            total = sum(pesos_amostragem)
            if total <= 0:
                break
            probs = [p / total for p in pesos_amostragem]

            escolhido_idx = int(self.rng.choice(len(remaining_indices), p=probs))
            escolhido = remaining_indices[escolhido_idx] + 1  # 1-based
            chosen.append(escolhido)

            remaining_indices.pop(escolhido_idx)
            remaining_weights = np.delete(remaining_weights, escolhido_idx)
            k_restante -= 1

        return sorted(chosen)

    def _sample_from_positional_model(self) -> list[int]:
        """Amostra para M3 (Produto Posicional) por amostragem backward exata (DP).

        Usa a matriz P do PositionalModel e delega para _backward_sample_from_P,
        que produz cada conjunto S com probabilidade exata
        prod_j P_j(d_j) / Z (diferente da amostragem sequencial truncada M3-seq).
        """
        if not hasattr(self.model, '_P'):
            raise GenerationError(
                "Modelo M3 não possui matriz _P de probabilidades positionais."
            )

        P = self.model._P  # shape (k, N), índice 0-based para dezenas
        return self._backward_sample_from_P(P)

    def _distribuicoes_para_matriz_P(
        self, distribuicoes: list[dict[int, float]]
    ) -> np.ndarray:
        """Converte uma lista de k dicionários {dezena: probabilidade} em
        uma matriz P (k, N) para uso na amostragem backward DP.

        Args:
            distribuicoes: lista de k dicionários, um por posição j.

        Returns:
            np.ndarray shape (k, N) com P[j, c-1] = probabilidade da dezena c
            na posição j (0-based para c).
        """
        k = len(distribuicoes)
        N = self.model._N if self.model is not None else NUM_CLASSES
        P = np.zeros((k, N), dtype=np.float64)
        for j, dist in enumerate(distribuicoes):
            for d, p in dist.items():
                if 1 <= d <= N:
                    P[j, d - 1] = float(p)
        return P

    def _backward_sample_from_P(self, P: np.ndarray) -> list[int]:
        """Amostragem backward exata por DP (math_spec.md §9.3 / Apêndice A.4).

        Produz um conjunto S com probabilidade exata
        prod_j P_j(d_j) / Z, onde Z é a normalização sobre conjuntos
        crescentes.

        Args:
            P: matriz (k, N) com P_j(c) para cada posição j e dezena c.

        Returns:
            Lista ordenada de k dezenas em [1, N].
        """
        k = P.shape[0]
        N = P.shape[1]

        # --- Passo 1 e 2: DP forward para F e Z ---
        F = np.zeros((k, N), dtype=np.float64)

        # Cabeça 0: d1 em 1..(N-k+1) -> índice 0..(N-k)
        max_c0 = N - k
        for c in range(0, max_c0 + 1):
            F[0, c] = P[0, c]

        # Cabeças 1 a k-1
        for j in range(1, k):
            min_c = j
            max_c = N - (k - j)
            prefix_sum = 0.0
            for c in range(min_c, max_c + 1):
                prefix_sum += F[j - 1, c - 1]
                F[j, c] = P[j, c] * prefix_sum

        Z = float(np.sum(F[k - 1, :]))
        if Z <= 0:
            raise GenerationError("Z calculado é zero; não é possível amostrar.")

        # --- Passo 3: amostrar d_k (última posição, índice k-1) ---
        last_probs = F[k - 1, :] / Z
        last_valid = np.where(last_probs > 0)[0]
        if len(last_valid) == 0:
            raise GenerationError("Nenhuma dezena válida para a última posição.")
        escolhido_c = int(self.rng.choice(last_valid, p=last_probs[last_valid]))
        chosen = [escolhido_c + 1]  # 1-based

        # --- Passo 4: amostrar d_{k-1} .. d_1 de trás para frente ---
        for j in range(k - 2, -1, -1):
            # chosen[-1] eh 1-based; índice zero-based máximo com c < d_{j+1}
            limite_superior = chosen[-1] - 2
            c_min = j
            c_max = min(limite_superior, N - (k - j))

            if c_max < c_min:
                # Não há dezenas válidas; fallback aleatório
                restante = [d for d in range(1, N + 1) if d not in chosen]
                if not restante:
                    break
                escolhido = int(self.rng.choice(restante))
                chosen.append(escolhido)
                continue

            pos_probs = F[j, c_min : c_max + 1]
            total = float(np.sum(pos_probs))
            if total <= 0:
                pos_probs = np.ones(c_max - c_min + 1, dtype=np.float64)
                total = float(len(pos_probs))
            pos_probs = pos_probs / total

            escolhido_idx = int(self.rng.choice(len(pos_probs), p=pos_probs))
            escolhido_c = c_min + escolhido_idx
            chosen.append(escolhido_c + 1)  # 1-based

        chosen.sort()
        return chosen

    def _sample_from_neural_model(self) -> list[int]:
        """Amostra usando rede neural (M4/M5) com backward DP exato.

        Obtém distribuições por posição via forward pass na janela de
        inferência e amostra por backward DP, produzindo cada conjunto com
        probabilidade exata P_M3(S) = prod_j P_j(d_j) / Z.
        """
        if self.model is None:
            raise GenerationError(
                "Modelo neural não fornecido. Passe o modelo treinado."
            )

        # Obter (ou cachear) distribuições por posição
        if self._distribuicoes_posicao is None:
            self._obter_distribuicoes_posicao()

        assert self._distribuicoes_posicao is not None

        P = self._distribuicoes_para_matriz_P(self._distribuicoes_posicao)
        return self._backward_sample_from_P(P)

    def _is_diverse(
        self,
        candidate: list[int],
        existing: list[list[int]],
        max_shared: int,
    ) -> bool:
        """Verifica se o candidato respeita o filtro de diversidade."""
        cand_set = set(candidate)
        for existing_game in existing:
            shared = len(cand_set & set(existing_game))
            if shared > max_shared:
                return False
        return True


def generate_from_model(
    model,
    model_name: str,
    K: int = DEFAULT_K,
    max_shared_numbers: int = DEFAULT_MAX_SHARED,
    seed: int = 42,
    dataset_hash: str | None = None,
    config_hash: str | None = None,
    commit: str | None = None,
    janela_inferencia: np.ndarray | None = None,
) -> GenerationResult:
    """Função de conveniência para gerar jogos com um modelo."""
    generator = Generator(
        model_name=model_name,
        model=model,
        seed=seed,
        janela_inferencia=janela_inferencia,
    )
    return generator.generate(
        K=K,
        max_shared_numbers=max_shared_numbers,
        dataset_hash=dataset_hash,
        config_hash=config_hash,
        commit=commit,
    )


def _to_json_serializable(obj: Any) -> Any:
    """Converte tipos numpy (np.int64, np.float64, etc.) para tipos nativos Python."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [_to_json_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [_to_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    return obj


def save_generation(
    result: GenerationResult,
    output_dir: Path = Path("artifacts/generations"),
) -> Path:
    """Salva uma geração como arquivo JSON imutável.

    Args:
        result: Resultado da geração.
        output_dir: Diretório de saída.

    Returns:
        Caminho do arquivo salvo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cria nome de arquivo com timestamp e hash
    ts = result.metadata.timestamp.replace(":", "-").replace(".", "-")
    content_hash = hashlib.sha256(
        json.dumps(_to_json_serializable(result.games), sort_keys=True).encode()
    ).hexdigest()[:8]
    filename = f"{ts}_{content_hash}.json"
    filepath = output_dir / filename

    # Converte para dict serializável (tipos nativos Python)
    data = {
        "criado_em": result.metadata.timestamp,
        "modelo": result.metadata.model_name,
        "hash_modelo": result.metadata.model_hash,
        "hash_dataset": result.metadata.dataset_hash,
        "hash_config": result.metadata.config_hash,
        "commit": result.metadata.commit,
        "seed": result.metadata.seed,
        "estrategia": result.metadata.strategy,
        "K": result.metadata.K,
        "diversidade": result.metadata.diversity,
        "tentativas": result.metadata.attempts,
        "saturado": result.metadata.saturated,
        "temperatura": result.metadata.temperature,
        "jogos": _to_json_serializable(result.metadata.games),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath


def load_generation(filepath: Path) -> GenerationResult:
    """Carrega uma geração salva."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = GenerationMetadata(
        timestamp=data.get("timestamp", data.get("criado_em", "")),
        model_name=data.get("model_name", data.get("modelo", "")),
        model_hash=data.get("model_hash", data.get("hash_modelo")),
        dataset_hash=data.get("dataset_hash", data.get("hash_dataset")),
        config_hash=data.get("config_hash", data.get("hash_config")),
        commit=data.get("commit"),
        seed=data["seed"],
        strategy=data.get("strategy", data.get("estrategia", "")),
        K=data["K"],
        diversity=data.get("diversity", data.get("diversidade")),
        attempts=data.get("attempts", data.get("tentativas", 0)),
        saturated=data.get("saturated", data.get("saturado", False)),
        games=data.get("games", data.get("jogos", [])),
        temperature=data.get("temperatura", 1.0),
    )

    return GenerationResult(games=metadata.games, metadata=metadata)