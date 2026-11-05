"""Modelo M5 — LSTM de 6 cabeças (Produto Posicional / Pseudo-likelihood).

Implementa o modelo definido em spec/math_spec.md §4.6:

Arquitetura de referência:
    sequência (W, 61) → LSTM(128, return_sequences=True)
    → Dropout(0.2) → LSTM(64) → concat(alvo_virada)
    → Dense(128, relu) → 6 × Dense(60, softmax)

Duas variantes:
- M5-SET: treinada com -sum_j log P_j(d_j) + log Z (likelihood de conjunto).
  Comparável com M1, M2, B2a, B2b por NLL de conjunto.
- M5-CCE: treinada com soma de 6 CCEs (pseudo-likelihood).
  Comparável apenas com B2a-CCE.

Cobre: ADR-002 (M5 como ablacao), math_spec.md §4.6.
P1-05: implementacao do M5.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import tensorflow as tf
from tensorflow import keras

NUM_CLASSES: Final[int] = 60
NUM_DEZENAS_JOGO: Final[int] = 6

VARIANTE_SET = "set"   # Likelihood de conjunto
VARIANTE_CCE = "cce"   # Pseudo-likelihood (soma de CCEs)


class LSTMSeisCabecasError(ValueError):
    """Excecao levantada quando o modelo LSTM de 6 cabecas encontra dados invalidos."""


class LSTMSeisCabecas:
    """Modelo LSTM de 6 cabecas para o Mega da Virada.

    Produz 6 distribuicoes softmax independentes, uma por posição ordenada
    d1 < d2 < ... < d6. Cada cabeça é condicionada apenas no histórico (não
    no prefixo de dezenas já escolhidas), então é um modelo M3 (produto
    posicional), não um modelo PL sequencial.

    Duas variantes de treino:
    - VARIANTE_SET: maximiza a log-verossimilhança de conjunto
      (-sum_j log P_j(d_j) + log Z). Comparável por NLL com M1/M2.
    - VARIANTE_CCE: maximiza a pseudo-likelihood (soma de CCEs).
      Comparável apenas com B2a-CCE.
    """

    def __init__(
        self,
        variante: str = VARIANTE_SET,
        num_classes: int = NUM_CLASSES,
        num_dezenas_jogo: int = NUM_DEZENAS_JOGO,
        window_size: int = 20,
    ) -> None:
        if variante not in (VARIANTE_SET, VARIANTE_CCE):
            raise LSTMSeisCabecasError(
                f"Variante invalida: '{variante}. Use '{VARIANTE_SET}' ou '{VARIANTE_CCE}'."
            )

        self._variante = variante
        self._N = int(num_classes)
        self._k = int(num_dezenas_jogo)
        self._window_size = int(window_size)
        self._modelo: keras.Model | None = None
        self._esta_ajustado = False

    # ------------------------------------------------------------------
    # Construção da rede
    # ------------------------------------------------------------------
    def _construir_rede(self) -> keras.Model:
        """Constrói a rede Keras para o M5 (arquitetura de referência v1)."""
        entrada_seq = keras.layers.Input(
            shape=(self._window_size, 61), name="entrada_seq"
        )
        entrada_static = keras.layers.Input(shape=(1,), name="entrada_static")

        x = keras.layers.LSTM(128, return_sequences=True)(entrada_seq)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.LSTM(64)(x)
        x = keras.layers.Concatenate()([x, entrada_static])
        x = keras.layers.Dense(128, activation="relu")(x)

        # 6 cabeças softmax independentes, uma por posição ordenada
        cabecas = []
        for j in range(self._k):
            cabeca = keras.layers.Dense(
                self._N, activation="softmax", name=f"cabeca_{j+1}"
            )(x)
            cabecas.append(cabeca)

        # Empilha as cabecas em um único tensor de saída (batch, k, N)
        saida = keras.layers.Concatenate(axis=1)(
            [keras.layers.Reshape((1, self._N))(c) for c in cabecas]
        )

        modelo = keras.Model(
            inputs=[entrada_seq, entrada_static],
            outputs=saida,
        )
        return modelo

    def _obter_ou_construir_rede(self) -> keras.Model:
        """Retorna a rede, construindo-a se necessário."""
        if self._modelo is None:
            self._modelo = self._construir_rede()
        return self._modelo

    # ------------------------------------------------------------------
    # Perdas customizadas
    # ------------------------------------------------------------------
    def _perda_cce(self, y_true, y_pred):
        """Soma de CCEs das 6 cabecas (pseudo-likelihood, M5-CCE).

        y_true: (batch, k) índices zero-based
        y_pred: (batch, k, N) probabilidades de cada cabeça
        """
        perda_total = tf.constant(0.0, dtype=tf.float32)
        for j in range(self._k):
            perda_j = keras.losses.sparse_categorical_crossentropy(
                y_true[:, j], y_pred[:, j, :]
            )
            perda_total = perda_total + perda_j
        return tf.reduce_mean(perda_total)

    def _perda_conjunto_set(self, y_true, y_pred):
        """Log-verossimilhança de conjunto (M5-SET).

        log P(S) = sum_j log P_j(d_j) - log Z
        onde Z = soma sobre conjuntos crescentes de prod_j P_j(d_j),
        calculado por DP em O(k * N).

        y_true: (batch, k) índices zero-based
        y_pred: (batch, k, N) probabilidades de cada cabeça
        """
        y_true = tf.cast(y_true, tf.int32)  # garantir int32 para indexação
        batch_size = tf.shape(y_true)[0]
        k = self._k
        N = self._N

        # log P_j(d_j) para cada cabeça e amostra
        # y_pred: (batch, k, N), y_true: (batch, k)
        # Obter log P_j(d_j) para cada (amostra, cabeça)
        indices = tf.stack(
            [tf.range(batch_size)[:, None] * tf.ones((1, k), dtype=tf.int32),
             tf.tile(tf.range(k)[None, :], [batch_size, 1]),
             y_true],
            axis=-1,
        )  # (batch, k, 3): [batch_idx, head_idx, class_idx]

        log_p_j = tf.math.log(
            tf.gather_nd(y_pred, indices) + 1e-12
        )  # (batch, k)
        soma_log_p_j = tf.reduce_sum(log_p_j, axis=1)  # (batch,)

        # Calcular Z por DP para cada amostra no batch
        # Z = soma sobre d1<...<dk de prod_j P_j(d_j)
        # DP: F[j, c] = P_j(c) * soma_{c' < c} F[j-1, c']
        def _calcular_z_amostra(P_amostra):
            """Calcula log Z para uma amostra (P tem shape k x N)."""
            # F[j, c] para j=0..k-1, c=0..N-1
            F = tf.zeros((k, N), dtype=tf.float32)

            # Cabeça 0: d1 em 1..(N-k+1) -> índice 0..(N-k)
            max_c0 = N - k
            mascara_0 = tf.cast(tf.range(N) <= max_c0, tf.float32)
            F0 = P_amostra[0, :] * mascara_0
            F = tf.tensor_scatter_nd_update(
                F,
                [[0, c] for c in range(N)],
                F0,
            )

            # Cabeças 1 a k-1
            for j in range(1, k):
                min_c = j
                max_c = N - (k - j)
                # Soma cumulativa de F[j-1, :] para obter prefix sums
                cumsum_prev = tf.concat(
                    [[0.0], tf.cumsum(F[j - 1, :])[:-1]], axis=0
                )
                # Máscara para c em [min_c, max_c]
                mascara_j = tf.cast(
                    (tf.range(N) >= min_c) & (tf.range(N) <= max_c),
                    tf.float32,
                )
                F_j = P_amostra[j, :] * cumsum_prev * mascara_j
                F = tf.tensor_scatter_nd_update(
                    F,
                    [[j, c] for c in range(N)],
                    F_j,
                )

            Z = tf.reduce_sum(F[k - 1, :])
            return tf.math.log(Z + 1e-12)

        log_Z_batch = tf.map_fn(
            _calcular_z_amostra,
            y_pred,  # (batch, k, N) - itera sobre a primeira dimensão
            fn_output_signature=tf.float32,
        )  # (batch,)

        # log P(S) = sum_j log P_j(d_j) - log Z
        log_p_conjunto = soma_log_p_j - log_Z_batch  # (batch,)

        return -tf.reduce_mean(log_p_conjunto)

    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------
    def treinar(
        self,
        dados: list[tuple[int, ...]],
        janelas: np.ndarray | None = None,
        alvos_virada: np.ndarray | None = None,
        epocas: int = 100,
        taxa_aprendizado: float = 1e-3,
        paciencia: int = 10,
        validacao_split: float = 0.1,
        verbose: int = 1,
    ) -> dict:
        """Treina o modelo M5.

        Args:
            dados: lista de conjuntos S_t (conjuntos-alvo).
            janelas: array (N, W, 61) com janelas de histórico.
            alvos_virada: array (N,) com flags de Virada.
            epocas: número máximo de épocas (padrão: 100).
            taxa_aprendizado: learning rate do Adam.
            paciencia: paciência do early stopping.
            validacao_split: fração de validação.
            verbose: nível de verbosidade do Keras.

        Returns:
            Dicionário com histórico de treino.
        """
        modelo = self._obter_ou_construir_rede()
        modelo.optimizer = keras.optimizers.Adam(learning_rate=taxa_aprendizado)

        n = len(dados)

        # Se janelas não fornecidas, gera dados sintéticos (apenas para teste)
        if janelas is None:
            janelas = np.random.default_rng(42).uniform(
                0, 1, size=(n, self._window_size, 61)
            ).astype(np.float32)
            if alvos_virada is None:
                alvos_virada = np.zeros(n, dtype=np.float32)

        alvos_virada = np.asarray(alvos_virada, dtype=np.float32).reshape(-1, 1)

        # Converte conjuntos-alvo para índices zero-based
        alvos_idx = np.array(
            [[d - 1 for d in S] for S in dados],
            dtype=np.int32,
        )

        # Escolhe a perda baseada na variante
        if self._variante == VARIANTE_SET:
            perda = self._perda_conjunto_set
        else:
            perda = self._perda_cce

        modelo.compile(
            optimizer=modelo.optimizer,
            loss=perda,
        )

        # Callback de early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=paciencia,
            restore_best_weights=True,
            mode="min",
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        historico = modelo.fit(
            x=[janelas, alvos_virada],
            y=alvos_idx,
            epochs=epocas,
            validation_split=validacao_split,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose,
        )

        self._esta_ajustado = True

        return {
            "loss": historico.history.get("loss", []),
            "val_loss": historico.history.get("val_loss", []),
        }

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------
    @property
    def variante(self) -> str:
        """Retorna a variante ('set' ou 'cce')."""
        return self._variante

    @property
    def esta_ajustado(self) -> bool:
        """Retorna True se o modelo foi treinado."""
        return self._esta_ajustado

    @property
    def rede(self) -> keras.Model | None:
        """Retorna o modelo Keras (se construido)."""
        return self._modelo
