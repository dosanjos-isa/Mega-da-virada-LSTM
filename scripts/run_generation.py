"""Script de execução de geração de jogos.

Uso:
    python scripts/run_generation.py --modelo M1 --K 10 --seed 42 --diversidade 4
    python scripts/run_generation.py --modelo B0 --K 20 --seed 42 --diversidade 3
    python scripts/run_generation.py --modelo M5 --epocas 100 --seeds 10 --K 10 --seed 42

Este script deve ser executado dentro do ambiente de treino (.venv-treino).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Adiciona src ao path
import sys as _sys
from pathlib import Path as _Path

_SRC_DIR = _Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from lstm_virada.data import (
    load_megasena_raw,
    load_virada_raw,
    reconcile_virada_contests,
    unify_series,
    validate_dataset_quality,
    compute_sha256,
)
from lstm_virada.encoding import (
    DEZENA_COLS,
    FEATURES_PER_DRAW,
    create_temporal_windows,
)
from lstm_virada.generation import (
    Generator,
    generate_from_model,
    save_generation,
)
from lstm_virada.models import (
    PlackettLuce,
    FisherModel,
    PositionalModel,
    SeqPLModel,
    LSTMSeisCabecas,
    VARIANTE_SET,
    VARIANTE_CCE,
)

DEFAULT_K: int = 10
DEFAULT_MAX_SHARED: int = 4
DEFAULT_SEED: int = 42
DEFAULT_WINDOW_SIZE: int = 20


def carregar_dados(
    raw_megasena_path: str,
    raw_virada_path: str,
) -> tuple:
    """Carrega e prepara os dados para geração."""
    from pathlib import Path

    df, hash_dataset, manifest = run_pipeline(
        Path(raw_megasena_path),
        Path(raw_virada_path),
        Path("data/processed/dataset_megasena_virada_processed.csv"),
    )

    return df, hash_dataset


def run_pipeline(raw_megasena_path, raw_virada_path, processed_output_path):
    """Wrapper para run_pipeline do data.py."""
    from lstm_virada.data import run_pipeline as _run_pipeline
    return _run_pipeline(
        Path(raw_megasena_path),
        Path(raw_virada_path),
        Path(processed_output_path),
    )


def construir_janela_atual(df, window_size: int = 20):
    """Constrói janela a partir dos últimos W concursos."""
    from lstm_virada.encoding import extract_latest_inference_window
    return extract_latest_inference_window(df, window_size=window_size, alvo_virada=1)


def main():
    parser = argparse.ArgumentParser(
        description="Geração de jogos da Mega da Virada"
    )
    parser.add_argument(
        "--modelo",
        choices=["M0", "M1", "M2", "M3", "M4", "M5-SET", "M5-CCE", "B0", "B1", "B2a", "B2b", "B3"],
        default="M0",
        help="Modelo a usar para geração (padrão: M0 - uniforme)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=DEFAULT_K,
        help="Número de jogos a gerar (padrão: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semente aleatória (padrão: 42)",
    )
    parser.add_argument(
        "--diversidade",
        type=int,
        default=DEFAULT_MAX_SHARED,
        help="Máximo de dezenas compartilhadas entre jogos (padrão: 4)",
    )
    parser.add_argument(
        "--raw-megasena",
        type=str,
        default="data/raw/megasena-download-resultados.csv",
        help="Caminho para CSV da Mega-Sena",
    )
    parser.add_argument(
        "--raw-virada",
        type=str,
        default="data/raw/mega_da_virada.csv",
        help="Caminho para CSV da Mega da Virada",
    )
    parser.add_argument(
        "--dataset-hash",
        type=str,
        default=None,
        help="Hash do dataset (opcional, para rastreabilidade)",
    )
    parser.add_argument(
        "--config-hash",
        type=str,
        default=None,
        help="Hash da configuração (opcional, para rastreabilidade)",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Hash do commit Git (opcional, para rastreabilidade)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/generations",
        help="Diretório de saída para artefatos de geração",
    )
    parser.add_argument(
        "--temperatura",
        type=float,
        default=1.0,
        help="Temperatura para amostragem (padrão: 1.0)",
    )

    parser.add_argument(
        "--artefato-modelo",
        type=str,
        default=None,
        help="Caminho para arquivo de artefato do modelo treinado (opcional)",
    )

    args = parser.parse_args()

    # Carrega dados
    print("Carregando dados...")
    df, hash_dataset = carregar_dados(args.raw_megasena, args.raw_virada)
    print(f"Dataset carregado: {len(df)} concursos, hash: {hash_dataset[:16]}...")

    # Constrói janela atual para inferência
    janela = construir_janela_atual(df, window_size=DEFAULT_WINDOW_SIZE)
    print(f"Janela construída: shape {janela['x_seq'].shape}")

    # Instancia ou carrega modelo
    modelo = None
    janela_inferencia = None

    if args.modelo in ("M1", "M2", "M3", "M4", "M5-SET", "M5-CCE"):
        if args.artefato_modelo:
            # Carregar modelo treinado de artefato
            print(f"Carregando modelo de artefato: {args.artefato_modelo}")
            if args.modelo in ("M1",):
                from lstm_virada.models import PlackettLuce
                modelo = PlackettLuce.load(args.artefato_modelo)
            elif args.modelo in ("M2",):
                from lstm_virada.models import FisherModel
                modelo = FisherModel.load(args.artefato_modelo)
            elif args.modelo in ("M3",):
                from lstm_virada.models import PositionalModel
                modelo = PositionalModel.load(args.artefato_modelo)
            elif args.modelo in ("M4",):
                from lstm_virada.models import SeqPLModel
                modelo = SeqPLModel(
                    num_classes=60,
                    num_dezenas_jogo=6,
                    window_size=DEFAULT_WINDOW_SIZE,
                )
                # Carregar pesos da rede neural
                modelo._obter_ou_construir_rede().load_weights(args.artefato_modelo)
            elif args.modelo in ("M5-SET", "M5-CCE"):
                from lstm_virada.models import LSTMSeisCabecas
                variante = (
                    VARIANTE_SET if args.modelo == "M5-SET" else VARIANTE_CCE
                )
                modelo = LSTMSeisCabecas(
                    variante=variante,
                    num_classes=60,
                    num_dezenas_jogo=6,
                    window_size=DEFAULT_WINDOW_SIZE,
                )
                modelo._obter_ou_construir_rede().load_weights(args.artefato_modelo)
            print(f"Modelo {args.modelo} carregado do artefato.")
        else:
            # Fallback: modelo com pesos iniciais (não treinado)
            print(
                f"AVISO: Nenhum artefato de modelo especificado para "
                f"{args.modelo}. Usando pesos iniciais uniformes (não treinado)."
            )
            if args.modelo == "M1":
                from lstm_virada.models import PlackettLuce
                modelo = PlackettLuce()
                pesos_iniciais = {i: 1.0 for i in range(1, 61)}
                modelo.ajustar_pesos(pesos_iniciais)
            elif args.modelo == "M2":
                from lstm_virada.models import FisherModel
                modelo = FisherModel()
                pesos_iniciais = {i: 1.0 for i in range(1, 61)}
                modelo.ajustar_pesos(pesos_iniciais)
            elif args.modelo == "M3":
                from lstm_virada.models import PositionalModel
                modelo = PositionalModel(num_classes=60, num_dezenas_jogo=6)
                from collections import Counter
                contagens = [Counter() for _ in range(6)]
                df_mega = df[df["is_virada"] == 0]
                for _, row in df_mega.iterrows():
                    for j, col in enumerate(DEZENA_COLS):
                        contagens[j][row[col]] += 1
                dists = []
                for j in range(6):
                    total = sum(contagens[j].values())
                    dist = {
                        d: (contagens[j].get(d, 0) + 1) / (total + 60)
                        for d in range(1, 61)
                    }
                    dists.append(dist)
                modelo.ajustar_distribuicoes(dists)
            elif args.modelo == "M4":
                from lstm_virada.models import SeqPLModel
                modelo = SeqPLModel(
                    num_classes=60,
                    num_dezenas_jogo=6,
                    window_size=DEFAULT_WINDOW_SIZE,
                )
                # Construir rede para forward pass
                modelo._obter_ou_construir_rede()
            elif args.modelo in ("M5-SET", "M5-CCE"):
                from lstm_virada.models import LSTMSeisCabecas
                variante = (
                    VARIANTE_SET if args.modelo == "M5-SET" else VARIANTE_CCE
                )
                modelo = LSTMSeisCabecas(
                    variante=variante,
                    num_classes=60,
                    num_dezenas_jogo=6,
                    window_size=DEFAULT_WINDOW_SIZE,
                )
                modelo._obter_ou_construir_rede()

    # Construir janela de inferência para modelos neurais
    if args.modelo in ("M4", "M5-SET", "M5-CCE") and modelo is not None:
        janela_inferencia = construir_janela_atual(
            df, window_size=DEFAULT_WINDOW_SIZE
        )

    # Gera jogos
    print(f"\nGerando {args.K} jogos com modelo {args.modelo} (seed={args.seed}, diversidade={args.diversidade})...")
    
    resultado = generate_from_model(
        model=modelo,
        model_name=args.modelo,
        K=args.K,
        max_shared_numbers=args.diversidade,
        seed=args.seed,
        dataset_hash=args.dataset_hash or hash_dataset,
        config_hash=args.config_hash,
        commit=args.commit,
        janela_inferencia=janela_inferencia,
    )

    # Salva resultado
    caminho_saida = save_generation(resultado, Path(args.output_dir))
    print(f"\nArquivo salvo: {caminho_saida}")
    print(f"Jogos gerados: {len(resultado.games)}")
    print(f"Saturado: {resultado.metadata.saturated}")
    print(f"Tentativas: {resultado.metadata.attempts}")
    print(f"\nJogos gerados:")
    for i, jogo in enumerate(resultado.games, 1):
        print(f"  Jogo {i:2d}: {jogo}")


if __name__ == "__main__":
    main()