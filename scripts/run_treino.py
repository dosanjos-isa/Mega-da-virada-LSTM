"""Script de execução de treino — rodar apenas dentro do .venv-treino.

Uso:
    python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 42 43 44
    python scripts/run_treino.py --modelo M4 --epocas 100 --seeds 42 43 44 45 46 47 48 49 50 51

Este script:
1. Carrega e prepara os dados históricos.
2. Executa o treinamento walk-forward para o modelo especificado.
3. Salva os modelos treinados em artifacts/modelos/.
4. Registra os resultados em artifacts/resultados/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Adiciona src ao path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    import tensorflow as tf

    TF_DISPONIVEL = True
except ImportError:  # pragma: no cover
    TF_DISPONIVEL = False

from lstm_virada.data import run_pipeline
from lstm_virada.train_walkforward import (
    treinar_walkforward_modelo,
    MAX_EPOCHS,
    EARLY_STOPPING_PACIENCIA,
    SEEDS_MODELO_ESTATICO,
    SEEDS_MODELO_NEURAL,
)
from lstm_virada.models import (
    PlackettLuce,
    FisherModel,
    PositionalModel,
    SeqPLModel,
    LSTMSeisCabecas,
    VARIANTE_SET,
    VARIANTE_CCE,
    NUM_CLASSES,
    NUM_DEZENAS_JOGO,
)


def verificar_ambiente() -> None:
    """Verifica se o TensorFlow está disponível no ambiente atual."""
    if not TF_DISPONIVEL:
        print("=" * 60)
        print("  ERRO: TensorFlow não encontrado neste ambiente.")
        print("=" * 60)
        print()
        print("  Este script deve ser executado dentro do venv de treino.")
        print()
        sys.exit(1)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[INFO] {len(gpus)} GPU(s) detectada(s):")
        for gpu in gpus:
            print(f"         - {gpu.name}")
    else:
        print("[INFO] Nenhuma GPU detectada. Treino em CPU.")


def carregar_dados() -> dict:
    """Carrega e prepara os dados para treinamento.

    Returns:
        Dicionário com "concursos", "dezenas", "is_virada", "virada_ano",
        "hash_dataset".
    """
    print("Carregando e preparando dados...")
    df, hash_dataset, manifest = run_pipeline(
        Path("data/raw/megasena-download-resultados.csv"),
        Path("data/raw/mega_da_virada.csv"),
        Path("data/processed/dataset_megasena_virada_processed.csv"),
    )

    print(f"  Dataset: {len(df)} concursos, hash: {hash_dataset[:16]}...")

    dezenas_cols = ["d1", "d2", "d3", "d4", "d5", "d6"]
    dezenas = df[dezenas_cols].values.astype(np.int32)
    concursos = df["concurso"].values.astype(np.int32)
    is_virada = df["is_virada"].values.astype(np.int32)
    virada_ano = df["virada_ano"].values.astype(np.int32)

    return {
        "concursos": concursos,
        "dezenas": dezenas,
        "is_virada": is_virada,
        "virada_ano": virada_ano,
        "hash_dataset": hash_dataset,
    }


def salvar_modelo_estatico(
    modelo: PlackettLuce | FisherModel,
    modelo_nome: str,
    seed: int,
    output_dir: Path,
) -> Path:
    """Salva um modelo estático (M1/M2) em arquivo JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{modelo_nome}_seed{seed}_{timestamp}.json"
    filepath = output_dir / filename
    modelo.save(filepath)
    return filepath


def salvar_modelo_neural(
    modelo: SeqPLModel | LSTMSeisCabecas,
    modelo_nome: str,
    seed: int,
    output_dir: Path,
) -> Path:
    """Salva um modelo neural (M4/M5) em arquivo Keras."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{modelo_nome}_seed{seed}_{timestamp}.keras"
    filepath = output_dir / filename
    rede = modelo._obter_ou_construir_rede()
    rede.save(str(filepath))
    return filepath


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Treino de modelos para a Mega da Virada"
    )
    parser.add_argument(
        "--modelo",
        choices=["M1", "M2", "M3", "M4", "M5-SET", "M5-CCE"],
        default="M1",
        help="Modelo a treinar (padrão: M1)",
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=MAX_EPOCHS,
        help=f"Número máximo de épocas (padrão: {MAX_EPOCHS})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Sementes para o treino (padrão: 3 para estático, 10 para neural)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        choices=[10, 20, 40],
        default=20,
        help="Tamanho da janela W (padrão: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/modelos",
        help="Diretório para salvar modelos treinados",
    )
    parser.add_argument(
        "--resultados-dir",
        type=str,
        default="artifacts/resultados",
        help="Diretório para salvar resultados do walk-forward",
    )

    args = parser.parse_args()

    verificar_ambiente()

    # Determinar seeds
    if args.seeds is None:
        if args.modelo in ("M1", "M2"):
            seeds = list(range(SEEDS_MODELO_ESTATICO))
        else:
            seeds = list(range(SEEDS_MODELO_NEURAL))
    else:
        seeds = args.seeds

    print()
    print("=" * 60)
    print("  Iniciando treino walk-forward")
    print("=" * 60)
    print(f"  Modelo:        {args.modelo}")
    print(f"  Épocas:        {args.epocas}")
    print(f"  Sementes:      {seeds}")
    print(f"  Janela (W):    {args.window_size}")
    print("=" * 60)
    print()

    # Carregar dados
    dados = carregar_dados()

    # Executar treino walk-forward
    print(f"Executando walk-forward para {args.modelo}...")
    resultado_wf = treinar_walkforward_modelo(
        modelo_nome=args.modelo,
        dados=dados,
        epocas=args.epocas,
        seeds=seeds,
        window_size=args.window_size,
    )

    print()
    print("=" * 60)
    print("  Resultados do walk-forward")
    print("=" * 60)
    print(f"  Modelo:        {resultado_wf.modelo_nome}")
    print(f"  Variação:      {resultado_wf.variacao}")
    print(f"  N folds:       {resultado_wf.n_folds}")
    print(f"  Perda média:   {resultado_wf.perda_media:.6f}")
    print(f"  Perda desvio:  {resultado_wf.perda_desvio:.6f}")
    print(f"  Seeds:         {resultado_wf.seeds_usadas}")
    print("=" * 60)
    print()

    # Salvar modelos treinados (um por fold/seed)
    output_dir = Path(args.output_dir)
    resultados_dir = Path(args.resultados_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resultados_dir.mkdir(parents=True, exist_ok=True)

    print(f"Salvando modelos em {output_dir}...")

    # Para modelos estáticos (M1/M2), salvar um modelo por seed
    # (usando os dados do último fold como exemplo de treinamento final)
    if args.modelo in ("M1", "M2"):
        for seed in seeds:
            np.random.seed(seed)
            modelo = (
                PlackettLuce()
                if args.modelo == "M1"
                else FisherModel()
            )
            # Treinar com todos os dados disponíveis
            todos_dados = [
                tuple(int(d) for d in row) for row in dados["dezenas"]
            ]
            if args.modelo == "M1":
                historico = modelo.treinar_mle(
                    dados=todos_dados,
                    epocas=args.epocas,
                    taxa_aprendizado=1e-2,
                    paciencia=EARLY_STOPPING_PACIENCIA,
                    fracao_validacao=0.10,
                    seed=seed,
                    verbose=0,
                )
            else:
                historico = modelo.treinar_mle(
                    dados=todos_dados,
                    epocas=args.epocas,
                    taxa_aprendizado=1e-2,
                    paciencia=EARLY_STOPPING_PACIENCIA,
                    fracao_validacao=0.10,
                    seed=seed,
                    verbose=0,
                )
            filepath = salvar_modelo_estatico(
                modelo, args.modelo, seed, output_dir
            )
            print(f"  Salvado: {filepath}")

    # Para modelos neurais (M3, M4, M5-SET, M5-CCE), salvar rede Keras
    # treinada com todos os dados disponíveis
    if args.modelo in ("M3", "M4", "M5-SET", "M5-CCE"):
        for seed in seeds:
            np.random.seed(seed)
            if args.modelo == "M3":
                variante = VARIANTE_SET
                modelo = LSTMSeisCabecas(
                    variante=variante,
                    num_classes=NUM_CLASSES,
                    num_dezenas_jogo=NUM_DEZENAS_JOGO,
                    window_size=args.window_size,
                )
            elif args.modelo == "M4":
                modelo = SeqPLModel(
                    num_classes=NUM_CLASSES,
                    num_dezenas_jogo=NUM_DEZENAS_JOGO,
                    window_size=args.window_size,
                )
            else:  # M5-SET / M5-CCE
                variante = (
                    VARIANTE_SET if args.modelo == "M5-SET" else VARIANTE_CCE
                )
                modelo = LSTMSeisCabecas(
                    variante=variante,
                    num_classes=NUM_CLASSES,
                    num_dezenas_jogo=NUM_DEZENAS_JOGO,
                    window_size=args.window_size,
                )

            todos_dados = [
                tuple(int(d) for d in row) for row in dados["dezenas"]
            ]
            # Construir janelas com todos os dados
            n = len(todos_dados)
            multihot = np.zeros((n, 61), dtype=np.float32)
            for r in range(n):
                for dez in todos_dados[r]:
                    multihot[r, int(dez) - 1] = 1.0

            janelas_list = []
            alvos_list = []
            for t in range(args.window_size, n):
                janela = multihot[t - args.window_size : t]
                alvo = np.array(todos_dados[t], dtype=np.int32) - 1
                janelas_list.append(janela)
                alvos_list.append(alvo)

            if len(janelas_list) > 0:
                janelas_arr = np.asarray(janelas_list, dtype=np.float32)
                alvos_arr = np.asarray(alvos_list, dtype=np.int32)
                alvos_virada_arr = np.zeros(len(alvos_arr), dtype=np.float32)

                modelo.treinar(
                    dados=todos_dados,
                    janelas=janelas_arr,
                    alvos_virada=alvos_virada_arr,
                    epocas=args.epocas,
                    taxa_aprendizado=1e-3,
                    verbose=0,
                )

                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                filename = f"{args.modelo}_seed{seed}_{timestamp}.keras"
                filepath = output_dir / filename
                rede = modelo._obter_ou_construir_rede()
                rede.save(str(filepath))
                print(f"  Salvado: {filepath}")
            else:
                print(
                    f"  AVISO: dados insuficientes para treino de "
                    f"{args.modelo} com window_size={args.window_size}"
                )

    # Salvar resultados do walk-forward em JSON
    resultados = {
        "modelo_nome": resultado_wf.modelo_nome,
        "variacao": resultado_wf.variacao,
        "n_folds": resultado_wf.n_folds,
        "perda_media": resultado_wf.perda_media,
        "perda_desvio": resultado_wf.perda_desvio,
        "seeds_usadas": resultado_wf.seeds_usadas,
        "epocas": args.epocas,
        "window_size": args.window_size,
        "hash_dataset": dados["hash_dataset"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "folds": [
            {
                "virada_ano": f.virada_ano,
                "concurso_alvo": f.concurso_alvo,
                "n_treino": f.n_treino,
                "n_teste": f.n_teste,
                "loss": f.loss,
                "seed": f.seed,
                "epocas_executadas": f.epocas_executadas,
            }
            for f in resultado_wf.fold_results
        ],
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    resultados_file = resultados_dir / f"{args.modelo}_{timestamp}.json"
    with open(resultados_file, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"Resultados salvos em: {resultados_file}")

    print()
    print("Treino concluído com sucesso!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
