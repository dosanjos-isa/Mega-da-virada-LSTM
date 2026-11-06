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

from pathlib import Path
from typing import Final

import numpy as np
import tensorflow as tf

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
            raise FisherError("Modelo nao ajustado. Chame ajustar_pesos() primeiro.")
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
            raise FisherError(f"Dezenas devem ser distintas: {S_ordenado}.")

        if not all(1 <= d <= self._N for d in S_ordenado):
            raise FisherError(
                f"Todas as dezenas devem estar em [1, {self._N}]: {S_ordenado}."
            )

        return S_ordenado

    def prob_conjunto(self, S: tuple[int, ...]) -> float:
        """Retorna P_M2(S) = prod_{i in S} w_i / e_k(w)."""
        if self._w is None:
            raise FisherError("Modelo nao ajustado. Chame ajustar_pesos() primeiro.")

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
            raise FisherError("Modelo nao ajustado. Chame ajustar_pesos() primeiro.")

        return float(sum(self.log_prob_conjunto(S) for S in dados))

    # ------------------------------------------------------------------
    # Treinamento por maxima verossimilhança (MLE)
    # ------------------------------------------------------------------
    def treinar_mle(
        self,
        dados: list[tuple[int, ...]],
        epocas: int = 100,
        taxa_aprendizado: float = 1e-2,
        paciencia: int = 10,
        fracao_validacao: float = 0.10,
        seed: int | None = None,
        verbose: int = 0,
    ) -> dict[str, list[float]]:
        """Ajusta os pesos por maxima verossimilhança usando TensorFlow.

        Maximiza:
            l(θ) = sum_t [sum_{i in S_t} θ_i - log e_k(exp(θ))]

        com a restricao de identificabilidade sum_i θ_i = 0
        (media geometrica dos pesos igual a 1).

        A funcao de particao e_k(w) eh calculada por programacao dinamica
        diferenciavel em O(N * k).

        Args:
            dados: lista de conjuntos S_t (dezenas 1..60).
            epocas: número máximo de épocas (padrão: 100).
            taxa_aprendizado: learning rate do Adam (padrão: 1e-2).
            paciencia: early stopping sobre log-score de validacao (padrão: 10).
            fracao_validacao: fração dos dados usada para validacao (padrão: 0.10).
            seed: semente para reproduzibilidade.
            verbose: 0 = silencioso, 1 = imprime progresso.

        Returns:
            Dicionario com historico de treino e validacao:
            - "loss": log-verossimilhança negativa media por época (treino)
            - "val_loss": log-verossimilhança negativa media por época (validacao)
            - "melhor_epoca": índice da época com melhor val_loss
        """
        if self._w is not None:
            theta_init = np.log(self._w).astype(np.float64)
        else:
            # Inicializacao com frequencias empiricas
            frequencias = np.zeros(self._N, dtype=np.float64)
            for S in dados:
                for d in S:
                    frequencias[int(d) - 1] += 1.0
            frequencias = np.maximum(frequencias, 1.0)
            theta_init = np.log(frequencias).astype(np.float64)

        # Aplicar restricao identidade: theta com media zero
        theta_init = theta_init - np.mean(theta_init)

        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Contagem de ocorrencias de cada dezena nos dados
        contagens = np.zeros(self._N, dtype=np.float64)
        for S in dados:
            for d in S:
                contagens[int(d) - 1] += 1.0

        # Converter dados para indices zero-based
        dados_idx = np.array(
            [[int(d) - 1 for d in S] for S in dados], dtype=np.int32
        )  # shape (n_amostras, k)

        n_amostras = len(dados_idx)

        # Split cronologico: ultimo 10% para validacao
        n_val = max(1, int(n_amostras * fracao_validacao))
        n_treino = n_amostras - n_val

        dados_treino_idx = dados_idx[:n_treino]
        dados_val_idx = dados_idx[n_treino:]

        k = self._k
        N = self._N

        @tf.function
        def _calcular_e_k(w: tf.Tensor) -> tf.Tensor:
            """Calcula e_k(w) por programacao dinamica diferenciavel.

            Args:
                w: shape (N,) com pesos positivos.

            Returns:
                Escalar e_k(w).
            """
            e = tf.zeros(k + 1, dtype=tf.float64)
            e = tf.tensor_scatter_nd_update(e, [[0]], [1.0])

            def cond(i, e):
                return i < N

            def corpo(i, e):
                wi = w[i]
                max_j = tf.minimum(i + 1, k)
                j = max_j

                def cond_j(j, e):
                    return j > 0

                def corpo_j(j, e):
                    e_j = e[j] + wi * e[j - 1]
                    e = tf.tensor_scatter_nd_update(e, [[j]], [e_j])
                    return j - 1, e

                j, e = tf.while_loop(cond_j, corpo_j, [j, e])
                return i + 1, e

            _, e = tf.while_loop(cond, corpo, [0, e])
            return e[k]

        @tf.function
        def _log_verossimilhança_batch(batch_idx: tf.Tensor) -> tf.Tensor:
            """Calcula log P_M2(S) para um batch de conjuntos.

            Args:
                batch_idx: shape (batch_size, k) com indices zero-based.

            Returns:
                shape (batch_size,) com log P_M2(S) para cada conjunto.
            """
            theta_c = theta - tf.reduce_mean(theta)
            w = tf.exp(theta_c)  # shape (N,)

            e_k = _calcular_e_k(w)  # escalar

            # Numerador: prod_{i in S} w_i  ->  sum theta_c[idx]
            log_prod_w = tf.reduce_sum(tf.gather(theta_c, batch_idx), axis=-1)  # (B,)

            # log P(S) = log_prod_w - log(e_k)
            return log_prod_w - tf.math.log(e_k)

        @tf.function
        def _perda_batch(batch_idx: tf.Tensor) -> tf.Tensor:
            """Log-verossimilhança negativa media de um batch."""
            log_probs = _log_verossimilhança_batch(batch_idx)
            return -tf.reduce_mean(log_probs)

        theta = tf.Variable(theta_init, dtype=tf.float64, trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=taxa_aprendizado)

        historico: dict[str, list[float]] = {
            "loss": [],
            "val_loss": [],
            "melhor_epoca": 0,
        }

        melhor_val_loss = float("inf")
        melhor_pesos = None
        contagem_paciencia = 0

        for epoca in range(epocas):
            # Treino
            with tf.GradientTape() as tape:
                perda_treino = _perda_batch(dados_treino_idx)
            grads = tape.gradient(perda_treino, [theta])
            optimizer.apply_gradients(zip(grads, [theta], strict=False))

            # Re-aplicar restricao apos atualizacao
            theta.assign(theta - tf.reduce_mean(theta))

            loss_treino = float(perda_treino)

            # Validacao
            if n_val > 0:
                perda_val = _perda_batch(dados_val_idx)
                val_loss = float(perda_val)
            else:
                val_loss = loss_treino

            historico["loss"].append(loss_treino)
            historico["val_loss"].append(val_loss)

            if verbose >= 1 and (epoca % 10 == 0 or epoca == epocas - 1):
                print(
                    f"  Epoca {epoca:3d}: loss={loss_treino:.6f} "
                    f"val_loss={val_loss:.6f}"
                )

            # Early stopping
            if val_loss < melhor_val_loss:
                melhor_val_loss = val_loss
                melhor_pesos = theta.numpy().copy()
                historico["melhor_epoca"] = epoca
                contagem_paciencia = 0
            else:
                contagem_paciencia += 1
                if contagem_paciencia >= paciencia:
                    if verbose >= 1:
                        print(
                            f"  Early stopping na epoca {epoca} "
                            f"(paciencia {paciencia} esgotada)."
                        )
                    break

        # Restaurar melhores pesos
        if melhor_pesos is not None:
            theta.assign(melhor_pesos)

        # Converter theta final para pesos w e ajustar o modelo
        theta_final = theta.numpy()
        theta_final = theta_final - np.mean(theta_final)
        w_final = np.exp(theta_final)
        w_final = w_final / np.exp(np.mean(np.log(w_final)))

        pesos_dict = {int(i + 1): float(w_final[i]) for i in range(self._N)}
        self.ajustar_pesos(pesos_dict)

        return historico

    # ------------------------------------------------------------------
    # Modo do conjunto (exato para Fisher)
    # ------------------------------------------------------------------
    def modo(self) -> tuple[int, ...]:
        """Retorna o conjunto das k maiores pesos (modo exato do Fisher).

        Como e_k(w) e constante, o modo e simplesmente o conjunto das
        k dezenas de maior peso.
        """
        if self._w is None:
            raise FisherError("Modelo nao ajustado. Chame ajustar_pesos() primeiro.")

        # Indices das k maiores pesos (ordem crescente de dezenas)
        top_k_idx = np.argsort(self._w)[-self._k :]
        top_k_idx = np.sort(top_k_idx)
        return tuple(int(d + 1) for d in top_k_idx)

    # ------------------------------------------------------------------
    # Probabilidades de inclusao marginal p_i = P(i in S) (Brier score)
    # ------------------------------------------------------------------
    def prob_inclusao(self) -> np.ndarray:
        """Retorna p_i = P(i em S) para cada dezena i em [1, N].

        Formula exata (Apêndice A.5 da math_spec.md):
            p_i = w_i * e_{k-1}(w_{-i}) / e_k(w)

        Returns:
            np.ndarray shape (N,) com p_i para cada dezena (índice 0 = dezena 1).
        """
        if self._w is None:
            raise FisherError("Modelo nao ajustado. Chame ajustar_pesos() primeiro.")

        k = self._k
        N = self._N
        w = self._w
        e_k = self.e_k()

        p_inc = np.zeros(N, dtype=np.float64)
        for i in range(N):
            w_sem_i = np.delete(w, i)
            e_km1 = self._calcular_e_k(w_sem_i)
            p_inc[i] = w[i] * e_km1 / (k * e_k)

        p_inc = np.clip(p_inc, 0.0, 1.0)
        return p_inc

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------
    @property
    def pesos(self) -> np.ndarray:
        """Retorna copia dos pesos w_1..w_N."""
        if self._w is None:
            raise FisherError("Modelo nao ajustado. Chame ajustar_pesos() primeiro.")
        return self._w.copy()

    @property
    def esta_ajustado(self) -> bool:
        """Retorna True se os pesos ja foram ajustados."""
        return self._w is not None

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def save(self, filepath: str | Path) -> None:
        """Salva os pesos do modelo em um arquivo JSON.

        Args:
            filepath: Caminho do arquivo de saída.
        """
        if self._w is None:
            raise FisherError("Modelo nao ajustado. Nao ha o que salvar.")

        import json

        caminho = Path(filepath)
        caminho.parent.mkdir(parents=True, exist_ok=True)

        dados = {
            "modelo": "M2",
            "N": self._N,
            "k": self._k,
            "pesos": {str(i + 1): float(self._w[i]) for i in range(self._N)},
            "log_w": {str(i + 1): float(self._log_w[i]) for i in range(self._N)},
        }

        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str | Path) -> FisherModel:
        """Carrega um modelo a partir de um arquivo JSON.

        Args:
            filepath: Caminho do arquivo salvo.

        Returns:
            Instancia de FisherModel com os pesos restaurados.
        """
        import json

        caminho = Path(filepath)
        if not caminho.exists():
            raise FisherError(f"Arquivo nao encontrado: {caminho}")

        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)

        if dados.get("modelo") != "M2":
            raise FisherError(
                f"Arquivo nao contem um modelo M2 (encontrado: {dados.get('modelo')})."
            )

        N = int(dados["N"])
        k = int(dados["k"])
        pesos = {int(d): float(p) for d, p in dados["pesos"].items()}

        modelo = cls(num_classes=N, num_dezenas_jogo=k)
        modelo.ajustar_pesos(pesos)
        return modelo
