"""Modelo M4 — Plackett–Luce Neural Sequencial.

Implementa o modelo definido em spec/math_spec.md §4.5:

    P_ord(pi | historico) = prod_{j=1..k} w_{j,pi_j} / (soma_{i nao em prefixo} w_{j,i})

    P_M4(S | historico) = soma_{pi in Perm(S)} P_ord(pi | historico)

Uma rede (LSTM/GRU) condicionada no histórico e no prefixo de dezenas já
escolhidas produz, a cada passo j, pesos w_{j,i} para as dezenas restantes.

Custo: 720 × 6 avaliações de rede por concurso (para k=6). Isso e caro;
considere usar M3 como primeira ablacao (ADR-002).

Cobre: ADR-002 (M4 como ablacao opcional), math_spec.md §4.5.
P1-04: implementacao do M4.
"""

from __future__ import annotations

from itertools import permutations
from typing import Final

import numpy as np
import tensorflow as tf
from tensorflow import keras

NUM_CLASSES: Final[int] = 60
NUM_DEZENAS_JOGO: Final[int] = 6


class SeqPLError(ValueError):
    """Excecao levantada quando o modelo SeqPL encontra dados invalidos."""


class SeqPLModel:
    """Modelo Plackett–Luce Neural Sequencial.

    Produz pesos condicionados no histórico e no prefixo de dezenas
    já escolhidas. A probabilidade de um conjunto S dado um histórico h
    eh a soma sobre as k! permutacoes de S da probabilidade sequencial.

    Arquitetura de referencia:
        sequência (W, 61) → LSTM(128) → Dropout(0.2) → LSTM(64)
        → concat(alvo_virada) → Dense(128, relu)
        → para cada passo j: Dense(60) → exp() = pesos w_{j,i}
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_dezenas_jogo: int = NUM_DEZENAS_JOGO,
        window_size: int = 20,
    ) -> None:
        self._N = int(num_classes)
        self._k = int(num_dezenas_jogo)
        self._window_size = int(window_size)
        self._modelo: keras.Model | None = None
        self._esta_ajustado = False

    # ------------------------------------------------------------------
    # Construção da rede
    # ------------------------------------------------------------------
    def _construir_rede(self) -> keras.Model:
        """Constrói a rede Keras para o modelo SeqPL."""
        # Entrada sequencial: (batch, W, 61)
        entrada_seq = keras.layers.Input(
            shape=(self._window_size, 61), name="entrada_seq"
        )
        # Entrada estática: alvo_virada (batch, 1)
        entrada_static = keras.layers.Input(shape=(1,), name="entrada_static")

        x = keras.layers.LSTM(128, return_sequences=True)(entrada_seq)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.LSTM(64)(x)
        x = keras.layers.Concatenate()([x, entrada_static])
        x = keras.layers.Dense(128, activation="relu")(x)

        # Saída única: (batch, k * N) com pesos brutos para todos os passos
        logits = keras.layers.Dense(self._k * self._N, activation=None)(x)
        # Ativação exp para garantir pesos positivos
        pesos = keras.layers.Activation(tf.exp)(logits)
        # Reshape para (batch, k, N)
        pesos_reshaped = keras.layers.Reshape((self._k, self._N))(pesos)

        modelo = keras.Model(
            inputs=[entrada_seq, entrada_static],
            outputs=pesos_reshaped,
        )
        return modelo

    def _obter_ou_construir_rede(self) -> keras.Model:
        """Retorna a rede, construindo-a se necessário."""
        if self._modelo is None:
            self._modelo = self._construir_rede()
            self._modelo.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            )
        return self._modelo

    # ------------------------------------------------------------------
    # Perda customizada: NLL de conjunto por marginalização sobre permutações
    # ------------------------------------------------------------------
    def _perda_nll_conjunto(self, y_true, y_pred):
        """Calcula o NLL de conjunto via marginalização sobre permutações.

        Para cada amostra t com conjunto-alvo S_t:
            log P(S_t) = logsumexp_{pi in Perm(S_t)} log P_ord(pi)
            perda_t = -log P(S_t)

        Args:
            y_true: tensor (batch, k) com índices zero-based das dezenas alvo.
            y_pred: tensor (batch, k, N) com pesos w_{j,i} para cada passo j.

        Returns:
            escalar: perda média (NLL) sobre o batch.
        """
        batch_size = tf.shape(y_true)[0]
        k = self._k
        N = self._N

        # Todas as permutações de 0..k-1 (constante)
        perms_list = list(permutations(range(k)))
        perms = tf.constant(perms_list, dtype=tf.int32)  # (n_perms, k)
        tf.shape(perms)[0]

        def log_prob_para_amostra(i):
            """log P(S_i) para a amostra i via logsumexp sobre permutações."""
            alvos_i = tf.cast(y_true[i], tf.int32)  # (k,)

            def log_prob_para_permutacao(perm):
                """log P_ord(pi) para uma permutação."""
                log_prob = tf.constant(0.0, dtype=tf.float32)
                usadas = tf.zeros(N, dtype=tf.bool)

                for j in range(k):
                    idx_alvo = tf.cast(alvos_i[perm[j]], tf.int32)  # índice da dezena
                    w_j = y_pred[i, j, :]  # (N,) pesos do passo j para amostra i

                    # Denominador: soma dos pesos para dezenas não usadas
                    w_disponiveis = tf.where(usadas, tf.zeros_like(w_j), w_j)
                    denominador = tf.reduce_sum(w_disponiveis)

                    log_prob = (
                        log_prob + tf.math.log(w_j[idx_alvo]) - tf.math.log(denominador)
                    )

                    # Marcar idx_alvo como usada
                    usadas = tf.tensor_scatter_nd_update(usadas, [[idx_alvo]], [True])

                return log_prob

            log_probs_perms = tf.map_fn(
                log_prob_para_permutacao,
                perms,
                fn_output_signature=tf.float32,
            )  # (n_perms,)

            # logsumexp
            max_log = tf.reduce_max(log_probs_perms)
            logsumexp = max_log + tf.math.log(
                tf.reduce_sum(tf.exp(log_probs_perms - max_log))
            )
            return logsumexp

        log_probs_batch = tf.map_fn(
            log_prob_para_amostra,
            tf.range(batch_size),
            fn_output_signature=tf.float32,
        )  # (batch,)

        return -tf.reduce_mean(log_probs_batch)

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
        sample_weight: np.ndarray | None = None,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: int = 1,
    ) -> dict:
        """Treina o modelo SeqPL por máxima verossimilhança de conjunto.

        Args:
            dados: lista de conjuntos S_t (conjuntos-alvo).
            janelas: array (N, W, 61) com janelas de histórico. Se None,
                dados sintéticos são gerados (apenas para teste).
            alvos_virada: array (N,) com flags de Virada. Se None, todos zero.
            epocas: número máximo de épocas (padrão: 100).
            taxa_aprendizado: learning rate do Adam.
            paciencia: paciência do early stopping.
            validacao_split: fração de validação (usado apenas se x_val/y_val
                não forem fornecidos).
            sample_weight: pesos por amostra para o treino (shape (N,)).
            x_val: dados de validacao opcionais [x_seq, x_static]. Se fornecido,
                nao usa validacao_split do Keras.
            y_val: alvos de validacao (shape (N_val, k)).
            verbose: nível de verbosidade do Keras.

        Returns:
            Dicionário com histórico de treino.
        """
        modelo = self._obter_ou_construir_rede()
        modelo.optimizer.learning_rate = taxa_aprendizado

        n = len(dados)

        # Se janelas não fornecidas, gera dados sintéticos (apenas para teste)
        if janelas is None:
            janelas = (
                np.random.default_rng(42)
                .uniform(0, 1, size=(n, self._window_size, 61))
                .astype(np.float32)
            )
            if alvos_virada is None:
                alvos_virada = np.zeros(n, dtype=np.float32)

        alvos_virada = np.asarray(alvos_virada, dtype=np.float32).reshape(-1, 1)

        # Converte conjuntos-alvo para índices zero-based
        alvos_idx = np.array(
            [[d - 1 for d in S] for S in dados],
            dtype=np.int32,
        )

        # Compila com perda customizada de NLL de conjunto
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=taxa_aprendizado),
            loss=self._perda_nll_conjunto,
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

        fit_kwargs: dict = {
            "x": [janelas, alvos_virada],
            "y": alvos_idx,
            "epochs": epocas,
            "callbacks": [early_stop, reduce_lr],
            "verbose": verbose,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if x_val is not None and y_val is not None:
            # Validacao explicitamente fornecida (exclui Viradas, D-12)
            x_val_static = np.zeros((len(y_val), 1), dtype=np.float32)
            fit_kwargs["validation_data"] = ([x_val, x_val_static], y_val)
        else:
            fit_kwargs["validation_split"] = validacao_split

        historico = modelo.fit(**fit_kwargs)

        self._esta_ajustado = True

        return {
            "loss": historico.history.get("loss", []),
            "val_loss": historico.history.get("val_loss", []),
        }

    # ------------------------------------------------------------------
    # Probabilidade de um conjunto (com pesos fixos ou pela rede)
    # ------------------------------------------------------------------
    def definir_pesos_sequencia(
        self,
        pesos_sequencia: list[dict[int, float]],
    ) -> None:
        """Define pesos fixos para teste (sem usar a rede neural).

        Args:
            pesos_sequencia: lista de k dicionarios {dezena: peso},
                um por passo j.
        """
        if len(pesos_sequencia) != self._k:
            raise SeqPLError(
                f"Esperados {self._k} passos, recebidos: {len(pesos_sequencia)}."
            )

        self._pesos_fixos = pesos_sequencia
        self._esta_ajustado = True

    def _pesos_para_passo(
        self,
        passo: int,
        usadas: set[int],
    ) -> dict[int, float]:
        """Retorna pesos para o passo dado, excluindo dezenas já usadas."""
        if hasattr(self, "_pesos_fixos"):
            pesos = self._pesos_fixos[passo]
        else:
            # Usar a rede neural (requer histórico, não implementado aqui
            # para o modo de pesos fixos)
            raise SeqPLError(
                "Rede neural não implementada para probabilidade sob consulta. "
                "Use definir_pesos_sequencia() para testes com pesos fixos."
            )

        return {d: w for d, w in pesos.items() if d not in usadas}

    def _obter_pesos_para_janela(
        self,
        janela_x_seq: np.ndarray | None,
        janela_x_static: np.ndarray | None,
    ) -> np.ndarray:
        """Executa forward pass e retorna pesos w_{j,i} (k, N) para uma janela.

        Args:
            janela_x_seq: array (1, W, 61) ou (W, 61). Se None, usa zeros.
            janela_x_static: array (1, 1) ou (1,). Se None, usa zeros.

        Returns:
            np.ndarray shape (k, N) com pesos w_{j,i} para cada passo j.
        """
        rede = self._obter_ou_construir_rede()
        if janela_x_seq is None:
            janela_x_seq = np.zeros((1, self._window_size, 61), dtype=np.float32)
        elif janela_x_seq.ndim == 2:
            janela_x_seq = janela_x_seq[np.newaxis, :, :]
        if janela_x_static is None:
            janela_x_static = np.zeros((1, 1), dtype=np.float32)
        elif janela_x_static.ndim == 1:
            janela_x_static = janela_x_static[np.newaxis, :]

        pesos = rede.predict([janela_x_seq, janela_x_static], verbose=0)  # (1, k, N)
        return pesos[0]  # (k, N)

    def _log_prob_sequencial_para_permutacao(
        self,
        pesos: np.ndarray,
        S_idx: tuple[int, ...],
        pi: tuple[int, ...],
    ) -> float:
        """Calcula log P_ord(pi) para uma permutacao dada os pesos (k, N).

        P_ord(pi) = prod_j w_{j, pi_j} / (soma_{i nao em prefixo} w_{j,i})
        """
        log_prob = 0.0
        usadas: set[int] = set()
        W_total = float(np.sum(pesos, axis=1))  # (k,) soma de todos os pesos por passo

        for j, dez_idx in enumerate(pi):
            w_j = pesos[j]  # (N,)
            # Denominador: soma dos pesos para dezenas nao usadas no passo j
            # Usar W_total[j] menos soma dos pesos das ja usadas
            soma_usadas = float(np.sum([w_j[i] for i in usadas]))
            denominador = W_total[j] - soma_usadas
            if denominador <= 0:
                return float("-inf")
            log_prob += np.log(w_j[dez_idx]) - np.log(denominador)
            usadas.add(dez_idx)

        return log_prob

    def prob_conjunto(
        self,
        S: tuple[int, ...],
        janela_x_seq: np.ndarray | None = None,
        janela_x_static: np.ndarray | None = None,
    ) -> float:
        """Retorna P_M4(S) = soma_{pi in Perm(S)} P_ord(pi).

        Se janela_x_seq for fornecida, usa a rede neural para obter pesos
        condicionados ao historico. Caso contrario, requer que
        definir_pesos_sequencia() tenha sido chamado (modo de teste com pesos
        fixos).
        """
        S_ordenado = tuple(sorted(int(d) for d in S))

        if len(set(S_ordenado)) != self._k:
            raise SeqPLError(f"Dezenas devem ser distintas: {S_ordenado}.")

        if not all(1 <= d <= self._N for d in S_ordenado):
            raise SeqPLError(
                f"Todas as dezenas devem estar em [1, {self._N}]: {S_ordenado}."
            )

        if janela_x_seq is not None or self._modelo is not None:
            # Usar rede neural
            pesos = self._obter_pesos_para_janela(janela_x_seq, janela_x_static)
            S_idx = tuple(d - 1 for d in S_ordenado)

            # logsumexp sobre as k! permutacoes
            termos_log = []
            for pi in permutations(S_idx):
                log_pi = self._log_prob_sequencial_para_permutacao(pesos, S_idx, pi)
                termos_log.append(log_pi)

            if not termos_log:
                return 0.0

            max_termo = max(termos_log)
            logsumexp = max_termo + np.log(
                sum(np.exp(t - max_termo) for t in termos_log)
            )
            return float(np.exp(logsumexp))

        # Modo pesos fixos (comportamento original)
        if not hasattr(self, "_pesos_fixos"):
            raise SeqPLError(
                "Use definir_pesos_sequencia() ou passe janela_x_seq para "
                "calcular probabilidades com a rede neural."
            )

        # Soma sobre as k! permutacoes
        total = 0.0
        for pi in permutations(S_ordenado):
            prob_seq = 1.0
            usadas: set[int] = set()
            for j, dez in enumerate(pi):
                pesos_passo = self._pesos_para_passo(j, usadas)
                W_restante = sum(w for d, w in pesos_passo.items() if d not in usadas)
                if W_restante <= 0:
                    prob_seq = 0.0
                    break
                prob_seq *= pesos_passo[dez] / W_restante
                usadas.add(dez)
            total += prob_seq

        return float(total)

    def log_prob_conjunto(
        self,
        S: tuple[int, ...],
        janela_x_seq: np.ndarray | None = None,
        janela_x_static: np.ndarray | None = None,
    ) -> float:
        """Retorna log P_M4(S) para um conjunto S."""
        p = self.prob_conjunto(
            S, janela_x_seq=janela_x_seq, janela_x_static=janela_x_static
        )
        if p <= 0.0:
            return -np.inf
        return float(np.log(p))

    def log_verossimilhança(self, dados: list[tuple[int, ...]]) -> float:
        """Retorna a log-verossimilhança de um conjunto de dados."""
        return float(sum(self.log_prob_conjunto(S) for S in dados))

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------
    @property
    def esta_ajustado(self) -> bool:
        """Retorna True se o modelo foi ajustado/treinado."""
        return self._esta_ajustado

    @property
    def rede(self) -> keras.Model | None:
        """Retorna o modelo Keras (se construido)."""
        return self._modelo
