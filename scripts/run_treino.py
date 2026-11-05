"""Script de execução de treino — rodar apenas dentro do .venv-treino.

Uso (dentro do venv ativado):
    python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10

Uso (fora do venv, via activar_treino):
    .\scripts\activar_treino.ps1
    python scripts/run_treino.py --modelo M1 --epocas 100 --seeds 10

Este script verifica se o TensorFlow está disponível antes de iniciar o treino.
Se não estiver, encerra com instruções para ativar o venv correto.
"""

from __future__ import annotations

import argparse
import sys

try:
    import tensorflow as tf

    TF_DISPONIVEL = True
except ImportError:  # pragma: no cover
    TF_DISPONIVEL = False


def verificar_ambiente() -> None:
    """Verifica se o TensorFlow está disponível no ambiente atual."""
    if not TF_DISPONIVEL:
        print("=" * 60)
        print("  ERRO: TensorFlow não encontrado neste ambiente.")
        print("=" * 60)
        print()
        print("  Este script deve ser executado dentro do venv de treino.")
        print()
        print("  Para ativar:")
        print("    .\\scripts\\activar_treino.ps1")
        print("    python scripts/run_treino.py ...")
        print()
        print("  Ou, se quiser treinar em CPU (sem GPU):")
        print("    .venv-treino\\Scripts\\Activate.ps1")
        print("    pip uninstall tensorflow-directml")
        print("    pip install tensorflow>=2.15")
        print()
        sys.exit(1)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[INFO] {len(gpu)} GPU(s) detectada(s) via DirectML:")
        for gpu in gpus:
            print(f"         - {gpu.name}")
    else:
        print("[INFO] Nenhuma GPU detectada. Treino em CPU.")


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
        default=100,
        help="Número máximo de épocas (padrão: 100)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        help="Sementes para o treino (padrão: 10 sementes)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        choices=[10, 20, 40],
        default=20,
        help="Tamanho da janela W (padrão: 20)",
    )

    args = parser.parse_args()

    verificar_ambiente()

    print()
    print("=" * 60)
    print("  Iniciando treino walk-forward")
    print("=" * 60)
    print(f"  Modelo:        {args.modelo}")
    print(f"  Épocas:        {args.epocas}")
    print(f"  Sementes:      {args.seeds}")
    print(f"  Janela (W):    {args.window_size}")
    print("=" * 60)
    print()

    # Aqui será integrada a chamada ao train_walkforward quando
    # a Fase 3 estiver completa e integrada ao pipeline de treino.
    print("[AVISO] Integração com train_walkforward em implementação.")
    print("        Este script está pronto para receber a chamada.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
