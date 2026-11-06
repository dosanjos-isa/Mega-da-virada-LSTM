#!/usr/bin/env bash
# Configuração do venv de TREINO com GPU (CUDA) no WSL2
# Venv criado no disco nativo do WSL (~) para evitar lentidão do OneDrive (/mnt/c)
set -euo pipefail

VENV_DIR="$HOME/.venv-treino-wsl"
PROJECT_HOST="/mnt/c/Users/Matheus Pinheiro/OneDrive - Am Tech/Área de Trabalho/Mega da virada LSTM"

echo "============================================================"
echo "  Configurando venv GPU no disco nativo do WSL (~)"
echo "============================================================"
echo "Venv: $VENV_DIR"
echo "Projeto: $PROJECT_HOST"
echo

# 0. Verificar GPU
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  /'

# 1. Criar venv
if [ -d "$VENV_DIR" ]; then rm -rf "$VENV_DIR"; fi
python3 -m venv --without-pip "$VENV_DIR"
"$VENV_DIR/bin/python" /tmp/get-pip.py --quiet
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# 2. Instalar tensorflow + CUDA via pip
"$VENV_DIR/bin/pip" install \
    tensorflow \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12

# 3. Stack do projeto
"$VENV_DIR/bin/pip" install numpy pandas scipy scikit-learn pyyaml

# 4. Pacote lstm_virada (ignorar restricao de Python >=3.11)
"$VENV_DIR/bin/pip" install -e "$PROJECT_HOST" --ignore-requires-python --no-deps

# 5. Testar GPU
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
"$VENV_DIR/bin/python" -c "
import tensorflow as tf
print('tf OK', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', len(gpus))
for g in gpus:
    print(' -', g.name)
with tf.device('/GPU:0'):
    x = tf.constant([1.0, 2.0, 3.0])
print('[OK] Teste de alocacao na GPU: sucesso')
"

echo
echo "============================================================"
echo "  Concluido!"
echo "============================================================"
echo "Para ativar: source $VENV_DIR/bin/activate"
echo "Para treinar: source $VENV_DIR/bin/activate && cd /mnt/c/... && python scripts/run_treino.py ..."
