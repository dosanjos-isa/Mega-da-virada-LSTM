"""Script de verificação de GPU para o projeto Mega da Virada.

Verifica três níveis de funcionalidade da GPU:
1. GPU disponível para o sistema (via nvidia-smi / bibliotecas CUDA)
2. GPU reconhecida pelo TensorFlow
3. Operação efetivamente executada na GPU

Uso:
    python scripts/check_gpu.py
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys

import tensorflow as tf


def verificar_gpu_sistema() -> bool:
    """Verifica se a GPU está disponível para o sistema (nvidia-smi / libcuda).

    Returns:
        True se uma GPU NVIDIA foi detectada pelo sistema.
    """
    try:
        resultado = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if resultado.returncode == 0 and resultado.stdout.strip():
            print("[OK] GPU disponível para o sistema (nvidia-smi):")
            for linha in resultado.stdout.strip().splitlines():
                print(f"       - {linha}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: verificar se libcuda.so está acessível
    try:
        ctypes.CDLL("libcuda.so.1")
        print("[OK] Biblioteca libcuda.so.1 encontrada no sistema.")
        return True
    except OSError:
        pass

    print("[AVISO] Nenhuma GPU NVIDIA detectada pelo sistema.")
    return False


def verificar_gpu_tensorflow() -> bool:
    """Verifica se o TensorFlow reconhece a GPU.

    Returns:
        True se o TensorFlow listou pelo menos uma GPU.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("[AVISO] Nenhuma GPU reconhecida pelo TensorFlow.")
        print("       O treino será executado em CPU.")
        return False

    print(f"[OK] {len(gpus)} GPU(s) reconhecida(s) pelo TensorFlow:")
    for gpu in gpus:
        print(f"       - {gpu.name} ({gpu.device_type})")
    return True


def testar_operacao_gpu() -> bool:
    """Executa uma operação tensorial explicitamente na GPU.

    Returns:
        True se a operação foi executada na GPU.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[ERRO] Não é possível testar operação na GPU sem GPU reconhecida.")
        return False

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.config.set_soft_device_placement(False)

    try:
        with tf.device("/GPU:0"):
            a = tf.random.normal((1024, 1024))
            b = tf.random.normal((1024, 1024))
            resultado = tf.matmul(a, b)

        dispositivo = str(resultado.device).upper()
        if "GPU:0" in dispositivo:
            print("[OK] Teste de operação na GPU: bem-sucedido.")
            print(f"       Dispositivo: {resultado.device}")
            print(f"       Formato: {resultado.shape}")
            return True
        else:
            print(f"[ERRO] Operação executada em {resultado.device}, não na GPU.")
            return False
    except Exception as exc:  # noqa: BLE001
        print(f"[ERRO] Falha ao executar operação na GPU: {exc}")
        return False


def main() -> int:
    """Ponto de entrada do script."""
    print("=" * 60)
    print("  Verificação de GPU — Mega da Virada / WSL2 + CUDA")
    print("=" * 60)
    print()
    print(f"TensorFlow: {tf.__version__}")
    print()

    # Nível 1: GPU no sistema
    passo1 = verificar_gpu_sistema()
    print()

    # Nível 2: GPU no TensorFlow
    passo2 = verificar_gpu_tensorflow()
    print()

    # Nível 3: Operação na GPU (só se o TF reconheceu a GPU)
    passo3 = False
    if passo2:
        passo3 = testar_operacao_gpu()
        print()

    print("=" * 60)
    print("  RESUMO DA VALIDAÇÃO")
    print("=" * 60)
    print(f"  1. GPU no sistema:          {'OK' if passo1 else 'FALHA'}")
    print(f"  2. GPU no TensorFlow:       {'OK' if passo2 else 'FALHA'}")
    print(f"  3. Operação na GPU:         {'OK' if passo3 else 'FALHA'}")
    print()

    if passo1 and passo2 and passo3:
        print(">>> AMBIENTE VALIDADO — GPU RTX 4050 pronta para treinamento.")
        return 0
    elif passo1 and passo2:
        print(">>> GPU PARCIALMENTE VALIDADA — TensorFlow reconhece a GPU,")
        print("    mas a operação explícita na GPU falhou.")
        return 1
    else:
        print(">>> GPU NÃO VALIDADA — Treino será executado em CPU.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
