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
from pathlib import Path
from typing import Final

import numpy as np
import tensorflow as tf

NUM_CLASSES: Final[int] = 60
NUM_DEZENAS_JOGO: Final[int] = 6

# Todas as 720 permutacoes de indices 0..5 (para k=6)
_PERMUTACOES_K6: tuple[tuple[int, ...], ...] = tuple(
    permutations(range(NUM_DEZENAS_JOGO))
)


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
        self._w: np.ndarray | None = None  # pesos w_1..w_N (indice 0..N-1)
        self._log_w: np.ndarray | None = None  # log w_i

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
            raise PlackettLuceError(f"Dezenas devem ser distintas: {S_ordenado}.")

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
        logsumexp = max_termo + np.log(sum(np.exp(t - max_termo) for t in termos_log))

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
            l(θ) = sum_t log P_M1(S_t; exp(θ))

        com a restricao de identificabilidade sum_i θ_i = 0
        (media geometrica dos pesos igual a 1).

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
            # Inicializar a partir dos pesos atuais, se existirem
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

        # Variavel treinável: theta (log-pesos), shape (N,)
        theta = tf.Variable(theta_init, dtype=tf.float64, trainable=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=taxa_aprendizado)

        @tf.function
        def _log_verossimilhança_batch(
            batch_idx: tf.Tensor,
        ) -> tf.Tensor:
            """Calcula log P_M1(S) para um batch de conjuntos.

            Args:
                batch_idx: shape (batch_size, k) com indices zero-based.

            Returns:
                shape (batch_size,) com log P_M1(S) para cada conjunto.
            """
            # theta_centrado: aplicacao da restricao sum(theta) = 0
            theta_c = theta - tf.reduce_mean(theta)
            w = tf.exp(theta_c)  # shape (N,)
            W = tf.reduce_sum(w)  # escalar

            # batch_size = B, k = 6
            tf.shape(batch_idx)[0]

            # Para cada amostra, obter os pesos das 6 dezenas: (B, 6)
            w_sel = tf.gather(w, batch_idx)  # shape (B, 6)

            # Para cada permutacao pi (720,), reordenar os pesos:
            # w_perm[b, p, j] = w_sel[b, perm_tensor[p, j]]
            # Usar loop Python (executado uma vez durante o tracing)
            w_perm_list = []
            for p in range(len(_PERMUTACOES_K6)):
                w_perm_list.append(
                    tf.gather(w_sel, _PERMUTACOES_K6[p], axis=1)
                )  # (B, 6)
            w_perm = tf.stack(w_perm_list, axis=1)  # (B, 720, 6)

            # Denominadores sequenciais:
            # Para cada passo j, restante_j = W - sum_{l<j} w_perm[l]
            # Usar cumsum ao longo do eixo k (ultimo)
            cumsum_w = tf.cumsum(w_perm, axis=-1)  # (B, 720, 6)
            # restante[j] = W - cumsum_w[j-1] (para j=0, restante = W)
            # Precisamos de: restante_0 = W, restante_1 = W - w[0], ...
            restantes = W - tf.pad(
                cumsum_w[:, :, :-1], [[0, 0], [0, 0], [1, 0]]
            )  # (B, 720, 6)
            # restantes[:, :, j] = W - sum_{l<j} w_perm[l]

            # log dos denominadores: (B, 720, 6)
            log_denom = tf.math.log(restantes)

            # Soma sobre os 6 passos: (B, 720)
            log_denom_soma = tf.reduce_sum(log_denom, axis=-1)

            # Termo de cada permutacao: -log_denom_soma
            termos = -log_denom_soma  # (B, 720)

            # Numerador: prod_{i in S} w_i  ->  sum theta_c[idx]
            log_prod_w = tf.reduce_sum(tf.gather(theta_c, batch_idx), axis=-1)  # (B,)

            # logsumexp sobre as 720 permutacoes
            max_termo = tf.reduce_max(termos, axis=-1, keepdims=True)  # (B, 1)
            logsumexp = max_termo + tf.math.log(
                tf.reduce_sum(tf.exp(termos - max_termo), axis=-1, keepdims=True)
            )  # (B, 1)
            logsumexp = tf.squeeze(logsumexp, axis=-1)  # (B,)

            # log P(S) = log_prod_w + logsumexp
            return log_prod_w + logsumexp

        @tf.function
        def _perda_batch(batch_idx: tf.Tensor) -> tf.Tensor:
            """Log-verossimilhança negativa media de um batch."""
            log_probs = _log_verossimilhança_batch(batch_idx)
            return -tf.reduce_mean(log_probs)

        historico: dict[str, list[float]] = {
            "loss": [],
            "val_loss": [],
            "melhor_epoca": 0,
        }

        melhor_val_loss = float("inf")
        melhor_pesos = None
        contagem_paciencia = 0

        for epoca in range(epocas):
            # Treino: batch completo (full-batch)
            with tf.GradientTape() as tape:
                perda_treino = _perda_batch(dados_treino_idx)
            grads = tape.gradient(perda_treino, [theta])
            optimizer.apply_gradients(zip(grads, [theta], strict=False))

            # Re-aplicar restricao apos atualizacao (projecao no hiperplano)
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
        # Aplicar restricao final
        theta_final = theta_final - np.mean(theta_final)
        w_final = np.exp(theta_final)
        # Normalizar para media geometrica 1 (redundante mas seguro)
        w_final = w_final / np.exp(np.mean(np.log(w_final)))

        pesos_dict = {int(i + 1): float(w_final[i]) for i in range(self._N)}
        self.ajustar_pesos(pesos_dict)

        return historico

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

        # Beam search sobre sequencias PL ordenadas
        beam: list[tuple[tuple[int, ...], float]] = [((), 0.0)]  # (sequencia, log_prob)
        W = float(np.sum(self._w))
        log_w = self._log_w

        largura_beam = 50

        for _passo in range(self._k):
            novos: list[tuple[tuple[int, ...], float]] = []

            for seq, log_p_seq in beam:
                usados = set(seq)
                for idx in range(self._N):
                    if idx in usados:
                        continue
                    restante = W - float(np.sum(self._w[list(usados)]))
                    log_p_novo = log_p_seq + log_w[idx] - np.log(restante)
                    novos.append(((*seq, idx), log_p_novo))

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
    # Amostragem e probabilidades de inclusao (Brier score)
    # ------------------------------------------------------------------
    def amostrar_conjunto(
        self,
        rng: np.random.Generator | None = None,
        temperatura: float = 1.0,
    ) -> tuple[int, ...]:
        """Amostra um conjunto S pela distribuicao PL de M1.

        Amostragem sequencial Plackett–Luce exata (math_spec.md §9.2):
        escolhe a primeira dezena com probabilidade proporcional a w_i/W,
        depois a segunda dentre as restantes com w_i / (W - w_escolhida), etc.

        Args:
            rng: gerador de numeros aleatorios (se None, usa np.random.default_rng).
            temperatura: temperatura T para suavizar pesos (w_i^(1/T)). T=1 na
                avaliacao científica.

        Returns:
            Tupla ordenada de k dezenas em [1, N].
        """
        if self._w is None:
            raise PlackettLuceError(
                "Modelo nao ajustado. Chame ajustar_pesos() primeiro."
            )

        if rng is None:
            rng = np.random.default_rng()

        w = self._w.copy()
        if temperatura != 1.0:
            w = w ** (1.0 / temperatura)

        N = self._N
        k = self._k
        remaining_indices = list(range(N))
        remaining_weights = w.copy()
        chosen = []

        for _ in range(k):
            if len(remaining_indices) == 0:
                break
            total = remaining_weights.sum()
            if total <= 0:
                break
            probs = remaining_weights / total
            probs = probs / probs.sum()
            idx = int(rng.choice(len(remaining_indices), p=probs))
            chosen_idx = remaining_indices[idx]
            chosen.append(chosen_idx + 1)
            remaining_indices.pop(idx)
            remaining_weights = np.delete(remaining_weights, idx)

        return tuple(sorted(chosen))

    def prob_inclusao(
        self,
        n_amostras: int = 100000,
        seed: int = 42,
        temperatura: float = 1.0,
    ) -> np.ndarray:
        """Retorna p_i = P(i em S) para cada dezena i em [1, N].

        Usa Monte Carlo com n_amostras sorteios da distribuicao PL
        (math_spec.md §7.2). Reporta erro de Monte Carlo implicitamente
        pela variancia do estimador.

        Args:
            n_amostras: numero de sorteios Monte Carlo (padrao: 100.000).
            seed: semente para reproduzibilidade.
            temperatura: temperatura T para amostragem.

        Returns:
            np.ndarray shape (N,) com p_i para cada dezena (índice 0 = dezena 1).
        """
        if self._w is None:
            raise PlackettLuceError(
                "Modelo nao ajustado. Chame ajustar_pesos() primeiro."
            )

        rng = np.random.default_rng(seed)
        contagens = np.zeros(self._N, dtype=np.int64)

        for _ in range(n_amostras):
            S = self.amostrar_conjunto(rng=rng, temperatura=temperatura)
            for d in S:
                contagens[d - 1] += 1

        p_inc = contagens.astype(np.float64) / n_amostras
        p_inc = np.clip(p_inc, 0.0, 1.0)
        return p_inc

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

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def save(self, filepath: str | Path) -> None:
        """Salva os pesos do modelo em um arquivo JSON.

        Args:
            filepath: Caminho do arquivo de saída.
        """
        if self._w is None:
            raise PlackettLuceError("Modelo nao ajustado. Nao ha o que salvar.")

        import json

        caminho = Path(filepath)
        caminho.parent.mkdir(parents=True, exist_ok=True)

        dados = {
            "modelo": "M1",
            "N": self._N,
            "k": self._k,
            "pesos": {str(i + 1): float(self._w[i]) for i in range(self._N)},
            "log_w": {str(i + 1): float(self._log_w[i]) for i in range(self._N)},
        }

        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str | Path) -> PlackettLuce:
        """Carrega um modelo a partir de um arquivo JSON.

        Args:
            filepath: Caminho do arquivo salvo.

        Returns:
            Instancia de PlackettLuce com os pesos restaurados.
        """
        import json

        caminho = Path(filepath)
        if not caminho.exists():
            raise PlackettLuceError(f"Arquivo nao encontrado: {caminho}")

        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)

        if dados.get("modelo") != "M1":
            raise PlackettLuceError(
                f"Arquivo nao contem um modelo M1 (encontrado: {dados.get('modelo')})."
            )

        N = int(dados["N"])
        k = int(dados["k"])
        pesos = {int(d): float(p) for d, p in dados["pesos"].items()}

        modelo = cls(num_classes=N, num_dezenas_jogo=k)
        modelo.ajustar_pesos(pesos)
        return modelo
