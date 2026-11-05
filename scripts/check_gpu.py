"""Script de verificação de GPU via DirectML para o projeto Mega da Virada.

Executa antes do treino para confirmar que o TensorFlow consegue enxergar a GPU
via DirectML. Se não encontrar GPU, o treino cairá em CPU silenciosamente.

Uso:
    python scripts/check_gpu.py
"""

from __future__ import annotations

import sys

import tensorflow as tf


def verificar_gpu() -> bool:
    """Verifica se o TensorFlow enxerga dispositivos GPU via DirectML.

    Returns:
        True se pelo menos uma GPU foi detectada, False caso contrário.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("[AVISO] Nenhuma GPU detectada pelo TensorFlow/DirectML.")
        print("       O treino será executado em CPU.")
        print("")
        print("       Para habilitar GPU no Windows:")
        print("         1. pip uninstall tensorflow")
        print("         2. pip install -r requirements-gpu.txt")
        print("         3. Reinicie este script.")
        return False

    print(f"[OK] {len(gpu)} GPU(s) detectada(s) via DirectML:")
    for gpu in gpus:
        print(f"       - {gpu.name} ({gpu.device_type})")

    # Teste rápido de alocação de tensor na GPU
    try:
        with tf.device("/GPU:0"):
            _ = tf.constant([1.0, 2.0, 3.0])
        print("[OK] Teste de alocação na GPU: bem-sucedido.")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERRO] Falha ao alocar tensor na GPU: {exc}")
        return False

    return True


def main() -> int:
    """Ponto de entrada do script."""
    print("=" * 60)
    print("  Verificação de GPU — Mega da Virada / DirectML")
    print("=" * 60)
    print()

    sucesso = verificar_gpu()

    print()
    if sucesso:
        print(">>> Ambiente pronto para treino com GPU.")
        return 0

    print(">>> Ambiente rodará em CPU. Treino funcionará, mas mais lento.")
    print()
    print("  Para habilitar GPU (DirectML) no Windows:")
    print("    1. Use Python 3.11 ou 3.12 (não 3.13).")
    print("    2. pip uninstall tensorflow")
    print("    3. pip install -r requirements-gpu.txt")
    print("    4. Reinicie este script.")
    print()
    print("  Alternativa: use WSL2 com TensorFlow CUDA (Linux) no Python 3.13.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
