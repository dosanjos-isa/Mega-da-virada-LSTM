#!/usr/bin/env bash
# Configuração do venv de TREINO com GPU (CUDA) no WSL2
# Venv criado no disco nativo do WSL (~) para evitar lentidão do OneDrive (/mnt/c)
set -euo pipefail

VENV_DIR="$HOME/.venvs/lstm-virada-tf221"
PROJECT_HOST="/mnt/c/Users/Matheus Pinheiro/OneDrive - Am Tech/Área de Trabalho/Mega da virada LSTM"

echo "============================================================"
echo "  Configurando venv GPU no disco nativo do WSL (~)"
echo "============================================================"
echo "Venv: $VENV_DIR"
echo "Projeto: $PROJECT_HOST"
echo

# 0. Verificar GPU
echo "GPU detectada pelo nvidia-smi:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  /'

# 1. Criar venv (sem pip, instalado depois via get-pip.py pois apt está travado)
if [ -d "$VENV_DIR" ]; then
    echo "Aviso: $VENV_DIR já existe. Removendo e recriando..."
    rm -rf "$VENV_DIR"
fi
python3.10 -m venv --without-pip "$VENV_DIR"
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
"$VENV_DIR/bin/python" /tmp/get-pip.py --quiet
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# 2. Instalar tensorflow com suporte a CUDA (versão 2.21.0, mesma do ambiente Windows)
echo ""
echo "Instalando tensorflow[and-cuda]==2.21.0 (isso pode demorar)..."
"$VENV_DIR/bin/pip" install 'tensorflow[and-cuda]==2.21.0'

# 3. Stack do projeto (dependências base)
echo ""
echo "Instalando dependências base do projeto..."
"$VENV_DIR/bin/pip" install numpy pandas scipy scikit-learn pyyaml

# 4. Pacote lstm_virada (ignorar restricao de Python >=3.11 e matplotlib que precisa de 3.11+)
echo ""
echo "Instalando pacote lstm-virada em modo editável..."
"$VENV_DIR/bin/pip" install -e "$PROJECT_HOST" --ignore-requires-python --no-deps

# 5. Testar GPU
echo ""
echo "Testando detecção de GPU pelo TensorFlow..."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib"
for d in "$VENV_DIR"/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$d" ]; then
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$d"
    fi
done

"$VENV_DIR/bin/python" -c "
import tensorflow as tf
print('TF version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs detectadas:', len(gpus))
for g in gpus:
    print('  -', g.name)
if gpus:
    with tf.device('/GPU:0'):
        x = tf.constant([1.0, 2.0, 3.0])
    print('[OK] Teste de alocacao na GPU: sucesso')
else:
    print('[AVISO] Nenhuma GPU detectada pelo TensorFlow')
"

echo
echo "============================================================"
echo "  Concluido!"
echo "============================================================"
echo "Para ativar: source $VENV_DIR/bin/activate"
echo "Para LD_LIBRARY_PATH: source scripts/activar_treino_wsl.sh"
echo "Para treinar: source scripts/activar_treino_wsl.sh && python scripts/run_treino.py ..."
