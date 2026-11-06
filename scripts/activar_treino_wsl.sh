#!/usr/bin/env bash
# ==============================================================================
# Ativação do ambiente de TREINO com GPU (CUDA) no WSL2 — Mega da Virada LSTM
#
# Uso (dentro do WSL):
#   source /home/matheus_pinheiro/.venvs/lstm-virada-tf221/bin/activate
#   source scripts/activar_treino_wsl.sh   # define LD_LIBRARY_PATH para CUDA
#
# Ou, direto do PowerShell (Windows):
#   wsl -d Ubuntu-22.04 -- bash -c "source /home/matheus_pinheiro/.venvs/lstm-virada-tf221/bin/activate && source scripts/activar_treino_wsl.sh && python scripts/run_treino.py ..."
# ==============================================================================

VENV_DIR="$HOME/.venvs/lstm-virada-tf221"
PROJECT_HOST="/mnt/c/Users/Matheus Pinheiro/OneDrive - Am Tech/Área de Trabalho/Mega da virada LSTM"

# Verificar se o ambiente existe
if [ ! -d "$VENV_DIR" ]; then
    echo "ERRO: Ambiente virtual não encontrado em $VENV_DIR"
    echo "Execute primeiro: scripts/configurar_venv_wsl_gpu_native.sh"
    return 1 2>/dev/null || exit 1
fi

# Construir LD_LIBRARY_PATH com todas as libs CUDA dos pacotes nvidia + driver WSL
LIB_DIRS="/usr/lib/wsl/lib"
for d in "$VENV_DIR"/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$d" ]; then
        LIB_DIRS="$LIB_DIRS:$d"
    fi
done
export LD_LIBRARY_PATH="$LIB_DIRS${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PROJECT_ROOT="$PROJECT_HOST"
export PYTHONPATH="$PROJECT_HOST/src:${PYTHONPATH:-}"

echo "============================================================"
echo "  Ambiente de TREINO (GPU/CUDA) ATIVO — WSL2"
echo "============================================================"
echo "  Venv:      $VENV_DIR"
echo "  Projeto:   $PROJECT_HOST"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""
echo "  Para treinar:"
echo "    python scripts/run_treino.py --modelo M5-SET --epocas 100 --seeds 10"
echo ""
echo "  Para verificar a GPU:"
echo "    python scripts/check_gpu.py"
echo ""
echo "  Para sair:"
echo "    deactivate"
echo "============================================================"
