"""Módulo de treino walk-forward (P1-06).

Orquestra o treino e avaliação de todos os modelos (M1-M5) usando
walk-forward por edição da Mega da Virada (2015-2025).

Baseado em:
- spec/math_spec.md §6 (estimação e treino)
- spec/pre_registration.md §7 (treino e hiperparâmetros)
- spec/pre_registration.md §4.3 (splits do walk-forward)

Regras:
- max_epochs = 100 (fixo)
- Early stopping: paciência 10 sobre validação sem peso
- 10 seeds para M3/M4/M5, 3 reinicializações para M1/M2
- Validação: último 10% cronológico dos concursos regulares do treino
- Causalidade: treino apenas com concursos anteriores ao alvo

Cobre: RV-001 (walk-forward), RV-007 (10 sementes), P1-06.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from lstm_virada.models import (
    NUM_CLASSES,
    NUM_DEZENAS_JOGO,
    VARIANTE_CCE,
    VARIANTE_SET,
    FisherModel,
    LSTMSeisCabecas,
    PlackettLuce,
    PositionalModel,
    SeqPLModel,
)
from lstm_virada.splits import WalkForwardSplitter

MAX_EPOCHS: Final[int] = 100
EARLY_STOPPING_PACIENCIA: Final[int] = 10
SEEDS_MODELO_ESTATICO: Final[int] = 3
SEEDS_MODELO_NEURAL: Final[int] = 10
VALIDACAO_FRAÇÃO: Final[float] = 0.10


class TreinoWalkForwardError(ValueError):
    """Excecao levantada quando o treino walk-forward encontra dados invalidos."""


@dataclass
class ResultadoFold:
    """Resultado de um único fold do walk-forward."""

    virada_ano: int
    concurso_alvo: int
    n_treino: int
    n_teste: int
    concursos_treino: np.ndarray
    epocas_executadas: int
    loss: float
    modelo_nome: str
    variacao: str | None = None
    seed: int | None = None


@dataclass
class ResultadoWalkForward:
    """Resultado completo do treino walk-forward."""

    modelo_nome: str
    variacao: str | None
    n_folds: int
    fold_results: list[ResultadoFold]
    perda_media: float
    perda_desvio: float
    seeds_usadas: list[int]


def _criar_janela_treino(
    dezenas_treino: np.ndarray,
    concursos_treino: np.ndarray,
    window_size: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Cria janelas de histórico e alvos para treino de rede neural.

    Para cada concurso t em concursos_treino (a partir do índice window_size),
    cria uma janela dos window_size concursos anteriores e o alvo é o
    concurso t.

    Args:
        dezenas_treino: array (N_treino, 6) com dezenas 1..60.
        concursos_treino: array (N_treino,) com números de concurso.
        window_size: tamanho da janela W.

    Returns:
        Tupla (janelas, alvos) onde janelas tem shape (N_amostras, W, 61)
        e alvos tem shape (N_amostras, 6) com índices zero-based.
    """
    n = len(dezenas_treino)
    if n <= window_size:
        return (
            np.zeros((0, window_size, 61), dtype=np.float32),
            np.zeros((0, 6), dtype=np.int32),
        )

    # Converte dezenas para multi-hot (N, 61)
    multihot = np.zeros((n, 61), dtype=np.float32)
    for r in range(n):
        for dez in dezenas_treino[r]:
            multihot[r, int(dez) - 1] = 1.0
        # Índice 60 (is_virada por passo) fica em 0.0 (P0-06)

    janelas_list = []
    alvos_list = []

    for t in range(window_size, n):
        janela = multihot[t - window_size : t]  # (W, 61)
        alvo = dezenas_treino[t] - 1  # zero-based
        janelas_list.append(janela)
        alvos_list.append(alvo)

    return (
        np.asarray(janelas_list, dtype=np.float32),
        np.asarray(alvos_list, dtype=np.int32),
    )


def _montar_janela_alvo(
    concurso_alvo: int,
    dezenas_treino: np.ndarray,
    concursos_treino: np.ndarray,
    is_virada_treino: np.ndarray,
    window_size: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrói a janela de inferência para um concurso alvo.

    A janela contem os window_size concursos imediatamente anteriores ao
    alvo, selecionados do conjunto de treino do fold. Preserva causalidade
    estrita (nenhum concurso >= alvo entra na janela).

    Args:
        concurso_alvo: numero do concurso alvo.
        dezenas_treino: array (N_treino, 6) com dezenas 1..60.
        concursos_treino: array (N_treino,) com numeros de concurso.
        is_virada_treino: array (N_treino,) com flags de Virada.
        window_size: tamanho da janela W.

    Returns:
        Tupla (x_seq, x_static) onde x_seq tem shape (1, window_size, 61)
        e x_static tem shape (1, 1).
    """
    n = len(concursos_treino)
    if n < window_size:
        return (
            np.zeros((1, window_size, 61), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )

    # Encontra o indice do ultimo concurso de treino estritamente anterior ao alvo
    indices_anteriores = np.nonzero(concursos_treino < concurso_alvo)[0]
    if len(indices_anteriores) == 0:
        return (
            np.zeros((1, window_size, 61), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )

    idx_ultimo_anterior = indices_anteriores[-1]
    idx_inicio = max(0, idx_ultimo_anterior - window_size + 1)
    janela_indices = np.arange(idx_inicio, idx_ultimo_anterior + 1)

    # Se nao há window_size concursos suficientes, preenche com zeros no início
    n_janela = len(janela_indices)
    x_seq = np.zeros((1, window_size, 61), dtype=np.float32)
    if n_janela > 0:
        multihot = np.zeros((n_janela, 61), dtype=np.float32)
        for r in range(n_janela):
            for dez in dezenas_treino[janela_indices[r]]:
                multihot[r, int(dez) - 1] = 1.0
        x_seq[0, window_size - n_janela :] = multihot

    # x_static: flag de Virada do alvo (sempre 1 para folds de Virada, mas
    # usa o valor do alvo se disponível)
    # Para o fold da Virada, o alvo e uma Virada, então alvo_virada = 1.
    x_static = np.ones((1, 1), dtype=np.float32)

    return x_seq, x_static


def _dividir_treino_validacao(
    janelas: np.ndarray,
    alvos: np.ndarray,
    concursos_treino: np.ndarray,
    is_virada_treino: np.ndarray,
    fraçao_validacao: float = VALIDACAO_FRAÇÃO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide os dados de treino em treino e validação cronológica.

    A validação é o último 10% dos concursos REGULARES do treino
    (não inclui Viradas), preservando a ordem cronológica.

    Args:
        janelas: array (N_amostras, W, 61)
        alvos: array (N_amostras, 6)
        concursos_treino: array (N_amostras,) com números de concurso
        is_virada_treino: array (N_amostras,) com flags de Virada
        fraçao_validacao: fração para validação (padrão: 0.10)

    Returns:
        Tupla (janelas_treino, alvos_treino, janelas_val, alvos_val,
        concursos_treino_ajustados, is_virada_treino_ajustados)
    """
    # Identificar índices de concursos regulares para validação
    regulares_mask = is_virada_treino == 0
    regulares_indices = np.nonzero(regulares_mask)[0]

    if len(regulares_indices) == 0:
        # Sem concursos regulares, usar tudo para treino
        n_val = 0
    else:
        n_val = max(1, int(len(regulares_indices) * fraçao_validacao))
        # Os últimos n_val concursos regulares vão para validação
        indice_corte = regulares_indices[-n_val]
        # Encontrar a posição no array original
        corte_pos = np.nonzero(concursos_treino == concursos_treino[indice_corte])[0]
        corte_pos = (
            corte_pos[0] if len(corte_pos) > 0 else len(concursos_treino) - n_val
        )

    if n_val == 0 or corte_pos <= 0:
        return (
            janelas,
            alvos,
            np.zeros((0, janelas.shape[1], janelas.shape[2]), dtype=np.float32),
            np.zeros((0, alvos.shape[1]), dtype=np.int32),
            concursos_treino,
            is_virada_treino,
        )

    janelas_treino = janelas[:corte_pos]
    alvos_treino = alvos[:corte_pos]
    janelas_val = janelas[corte_pos:]
    alvos_val = alvos[corte_pos:]

    return (
        janelas_treino,
        alvos_treino,
        janelas_val,
        alvos_val,
        concursos_treino,
        is_virada_treino,
    )


def _treinar_modelo_estatico(
    modelo_nome: str,
    dezenas_treino: np.ndarray,
    dezenas_teste: np.ndarray,
    seeds: list[int],
) -> list[ResultadoFold]:
    """Treina modelo estático (M1 ou M2) por máxima verossimilhança.

    Modelos estáticos não usam janelas nem épocas. Ajustam os pesos
    diretamente nos dados de treino e calculam NLL no teste.
    """
    resultados = []

    if modelo_nome == "M1":
        modelo_classe = PlackettLuce
    elif modelo_nome == "M2":
        modelo_classe = FisherModel
    else:
        raise TreinoWalkForwardError(f"Modelo estático não suportado: {modelo_nome}")

    for seed in seeds:
        np.random.seed(seed)
        modelo = modelo_classe()
        modelo.ajustar_pesos(
            {int(d): 1.0 for d in range(1, modelo._N + 1)}
        )  # inicialização uniforme
        # Ajuste pelos dados de treino (máxima verossimilhança)
        # Para M1/M2, o ajuste é iterativo; aqui usamos uma abordagem simples:
        # inicializar com frequências empíricas e fazer algumas iterações
        _ajustar_modelo_estatico(modelo, dezenas_treino)

        # Calcular NLL no teste
        nll = 0.0
        for S in dezenas_teste:
            nll += -modelo.log_prob_conjunto(tuple(int(d) for d in S))
        nll_medio = nll / len(dezenas_teste)

        resultados.append(
            ResultadoFold(
                virada_ano=0,  # será preenchido pelo chamador
                concurso_alvo=0,  # será preenchido pelo chamador
                n_treino=len(dezenas_treino),
                n_teste=len(dezenas_teste),
                epocas_executadas=1,  # modelo estático, não tem épocas
                loss=float(nll_medio),
                modelo_nome=modelo_nome,
                variacao=None,
                seed=seed,
            )
        )

    return resultados


def _ajustar_modelo_estatico(
    modelo: PlackettLuce | FisherModel,
    dezenas_treino: np.ndarray,
) -> None:
    """Ajusta um modelo estático (M1/M2) nos dados de treino.

    Inicializa com frequências empíricas normalizadas por média geométrica 1.
    Para M1, não há solução fechada; a inicialização empírica é um bom ponto
    de partida para testes.
    """
    N = modelo._N

    # Frequências empíricas
    frequencias = np.zeros(N, dtype=np.float64)
    for row in dezenas_treino:
        for dez in row:
            frequencias[int(dez) - 1] += 1.0
    frequencias = np.maximum(frequencias, 1.0)  # evitar zero
    pesos_iniciais = {int(i + 1): float(f) for i, f in enumerate(frequencias)}
    modelo.ajustar_pesos(pesos_iniciais)


def treinar_walkforward_modelo(
    modelo_nome: str,
    dados: dict,
    epocas: int = MAX_EPOCHS,
    seeds: list[int] | None = None,
    window_size: int = 20,
) -> ResultadoWalkForward:
    """Executa treino walk-forward por edição da Mega da Virada.

    Para cada edição Y (2015-2025):
    1. Treina com todos os concursos anteriores à Virada Y
    2. Valida com o último 10% dos concursos regulares do treino
    3. Avalia na Virada Y (teste)

    Args:
        modelo_nome: "M1", "M2", "M3", "M4", "M5-SET" ou "M5-CCE".
        dados: dict com "concursos", "dezenas", "is_virada".
        epocas: número máximo de épocas (padrão: 100).
        seeds: lista de seeds. Se None, usa padrão (10 para neural, 3 para estático).
        window_size: tamanho da janela W para redes neurais.

    Returns:
        ResultadoWalkForward com resultados por fold.

    Raises:
        TreinoWalkForwardError: Se o modelo não for suportado.
    """
    concursos = dados["concursos"]
    dezenas = dados["dezenas"]
    is_virada = dados["is_virada"]

    # Determinar seeds
    if seeds is None:
        if modelo_nome in ("M1", "M2"):
            seeds = list(range(SEEDS_MODELO_ESTATICO))
        else:
            seeds = list(range(SEEDS_MODELO_NEURAL))

    # Criar splitter de walk-forward
    # Criar DataFrame para o splitter
    import pandas as pd

    df = pd.DataFrame(
        {
            "concurso": concursos,
            "d1": dezenas[:, 0],
            "d2": dezenas[:, 1],
            "d3": dezenas[:, 2],
            "d4": dezenas[:, 3],
            "d5": dezenas[:, 4],
            "d6": dezenas[:, 5],
            "is_virada": is_virada,
            "virada_ano": dados.get(
                "virada_ano", np.where(is_virada == 1, concursos, 0)
            ),
        }
    )

    splitter = WalkForwardSplitter(df, virada_anos=tuple(range(2015, 2026)))
    folds = splitter.splits()

    resultados_folds: list[ResultadoFold] = []

    for fold in folds:
        # Índices de treino e teste
        treino_idx = fold.train_indices
        teste_idx = fold.test_indices

        dezenas_treino = dezenas[treino_idx]
        dezenas_teste = dezenas[teste_idx]
        concursos_treino = concursos[treino_idx]
        is_virada_treino = is_virada[treino_idx]
        concurso_alvo = int(concursos[teste_idx[0]])

        if modelo_nome in ("M1", "M2"):
            # Modelo estático: ajuste por máxima verossimilhança (MLE).
            # A avaliação fora da amostra usa o conjunto de teste do fold
            # (dezenas_teste), NÃO o val_loss interno do treino.
            for seed in seeds:
                np.random.seed(seed)
                modelo = PlackettLuce() if modelo_nome == "M1" else FisherModel()
                # MLE real com TensorFlow: Adam, lr=1e-2, max_epochs=100,
                # early stopping paciência 10, validação cronológica 10%.
                historico = modelo.treinar_mle(
                    dados=[tuple(int(d) for d in row) for row in dezenas_treino],
                    epocas=MAX_EPOCHS,
                    taxa_aprendizado=1e-2,
                    paciencia=EARLY_STOPPING_PACIENCIA,
                    fracao_validacao=VALIDACAO_FRAÇÃO,
                    seed=seed,
                    verbose=0,
                )
                epocas_executadas = len(historico["loss"])

                # AVALIAÇÃO FORA DA AMOSTRA: NLL no conjunto de teste do fold.
                nll = 0.0
                for S in dezenas_teste:
                    nll += -modelo.log_prob_conjunto(tuple(int(d) for d in S))
                nll_medio = nll / len(dezenas_teste)

                resultados_folds.append(
                    ResultadoFold(
                        virada_ano=fold.virada_ano,
                        concurso_alvo=concurso_alvo,
                        n_treino=len(dezenas_treino),
                        n_teste=len(dezenas_teste),
                        concursos_treino=concursos_treino,
                        epocas_executadas=epocas_executadas,
                        loss=float(nll_medio),
                        modelo_nome=modelo_nome,
                        variacao=None,
                        seed=seed,
                    )
                )
        else:
            # Modelo neural: criar janelas e treinar
            janelas_treino, alvos_treino = _criar_janela_treino(
                dezenas_treino, concursos_treino, window_size
            )

            if len(janelas_treino) == 0:
                continue

            # Pesos por amostra: Virada = 1.0, regular = alpha (padrão 1.0)
            alpha = 1.0  # configuracao primária do pre-registro (math_spec §6.5)
            pesos_treino = np.where(
                is_virada_treino[window_size:] == 1, 1.0, alpha
            ).astype(np.float32)

            # Dividir treino/validação cronológica (validação exclui Viradas,
            # último 10% dos concursos regulares do treino)
            (
                janelas_treino,
                alvos_treino,
                janelas_val,
                alvos_val,
                _,
                _,
            ) = _dividir_treino_validacao(
                janelas_treino,
                alvos_treino,
                concursos_treino,
                is_virada_treino,
            )

            # Ajustar pesos_treino para o recorte de validação
            n_treino_ajustado = len(janelas_treino)
            pesos_treino_ajustados = pesos_treino[:n_treino_ajustado]

            # x_val: [x_seq, x_static] para validacao explícita sem Viradas
            if len(janelas_val) > 0:
                x_val_seq = janelas_val
                np.zeros((len(alvos_val), 1), dtype=np.float32)
            else:
                x_val_seq = None
                alvos_val = None

            for seed in seeds:
                np.random.seed(seed)

                # Criar modelo baseado no nome
                if modelo_nome in ("M3", "M5-SET", "M5-CCE"):
                    variante = (
                        VARIANTE_SET
                        if modelo_nome in ("M3", "M5-SET")
                        else VARIANTE_CCE
                    )
                    modelo = LSTMSeisCabecas(
                        variante=variante,
                        num_classes=NUM_CLASSES,
                        num_dezenas_jogo=NUM_DEZENAS_JOGO,
                        window_size=window_size,
                    )
                    modelo.treinar(
                        dados=[tuple(int(d) for d in row) for row in dezenas_treino],
                        janelas=janelas_treino,
                        alvos_virada=is_virada_treino[
                            window_size : window_size + len(janelas_treino)
                        ],
                        epocas=epocas,
                        taxa_aprendizado=1e-3,
                        sample_weight=pesos_treino_ajustados,
                        x_val=x_val_seq,
                        y_val=alvos_val,
                        verbose=0,
                    )
                    # Avaliar fora da amostra: NLL no conjunto de teste do fold
                    # usando log_prob_conjunto com janela de inferencia do alvo.
                    nll = 0.0
                    n_teste = len(dezenas_teste)
                    for S in dezenas_teste:
                        x_seq_alvo, x_static_alvo = _montar_janela_alvo(
                            concurso_alvo=concurso_alvo,
                            dezenas_treino=dezenas_treino,
                            concursos_treino=concursos_treino,
                            is_virada_treino=is_virada_treino,
                            window_size=window_size,
                        )
                        nll += -modelo.log_prob_conjunto(
                            tuple(int(d) for d in S),
                            janela_x_seq=x_seq_alvo,
                            janela_x_static=x_static_alvo,
                        )
                    nll_medio = nll / n_teste if n_teste > 0 else 0.0
                elif modelo_nome == "M4":
                    modelo = SeqPLModel(
                        num_classes=NUM_CLASSES,
                        num_dezenas_jogo=NUM_DEZENAS_JOGO,
                        window_size=window_size,
                    )
                    modelo.treinar(
                        dados=[tuple(int(d) for d in row) for row in dezenas_treino],
                        janelas=janelas_treino,
                        alvos_virada=is_virada_treino[
                            window_size : window_size + len(janelas_treino)
                        ],
                        epocas=epocas,
                        taxa_aprendizado=1e-3,
                        paciencia=EARLY_STOPPING_PACIENCIA,
                        sample_weight=pesos_treino_ajustados,
                        x_val=x_val_seq,
                        y_val=alvos_val,
                        verbose=0,
                    )
                    # Avaliar fora da amostra: NLL no conjunto de teste do fold
                    # usando log_prob_conjunto com janela de inferencia do alvo.
                    nll = 0.0
                    n_teste = len(dezenas_teste)
                    for S in dezenas_teste:
                        x_seq_alvo, x_static_alvo = _montar_janela_alvo(
                            concurso_alvo=concurso_alvo,
                            dezenas_treino=dezenas_treino,
                            concursos_treino=concursos_treino,
                            is_virada_treino=is_virada_treino,
                            window_size=window_size,
                        )
                        nll += -modelo.log_prob_conjunto(
                            tuple(int(d) for d in S),
                            janela_x_seq=x_seq_alvo,
                            janela_x_static=x_static_alvo,
                        )
                    nll_medio = nll / n_teste if n_teste > 0 else 0.0
                else:
                    raise TreinoWalkForwardError(f"Modelo não suportado: {modelo_nome}")

                resultados_folds.append(
                    ResultadoFold(
                        virada_ano=fold.virada_ano,
                        concurso_alvo=concurso_alvo,
                        n_treino=len(dezenas_treino),
                        n_teste=len(dezenas_teste),
                        concursos_treino=concursos_treino,
                        epocas_executadas=epocas,
                        loss=float(nll_medio),
                        modelo_nome=modelo_nome,
                        variacao=getattr(modelo, "variante", None),
                        seed=seed,
                    )
                )

    # Agregar resultados
    if not resultados_folds:
        raise TreinoWalkForwardError(
            "Nenhum resultado de fold gerado. Verifique os dados de entrada."
        )

    perdas = [r.loss for r in resultados_folds]
    perda_media = float(np.mean(perdas))
    perda_desvio = float(np.std(perdas))

    return ResultadoWalkForward(
        modelo_nome=modelo_nome,
        variacao=resultados_folds[0].variacao,
        n_folds=len(resultados_folds),
        fold_results=resultados_folds,
        perda_media=perda_media,
        perda_desvio=perda_desvio,
        seeds_usadas=seeds,
    )


def _treinar_m3_empirico(
    modelo: PositionalModel,
    dezenas_treino: np.ndarray,
) -> None:
    """Estima distribuições positionais empíricas dos dados de treino."""
    n = len(dezenas_treino)
    k = modelo._k
    N = modelo._N

    # Contagens n_{j,c} para cada posição j
    contagens = np.zeros((k, N + 1), dtype=np.float64)
    for row in dezenas_treino:
        for j, dez in enumerate(row):
            contagens[j, int(dez)] += 1.0

    # Suavização Laplace kappa=1
    kappa = 1.0
    distribuicoes = []
    for j in range(k):
        dist = {}
        c_min = j + 1
        c_max = N - (k - j - 1)
        denominador = n + 55 * kappa
        for c in range(c_min, c_max + 1):
            n_jc = contagens[j, c]
            dist[c] = (n_jc + kappa) / denominador
        distribuicoes.append(dist)

    modelo.ajustar_distribuicoes(distribuicoes)
