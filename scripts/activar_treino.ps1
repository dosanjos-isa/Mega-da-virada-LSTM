# ==============================================================================
# Script de ativação do ambiente de TREINO (venv dedicado ao TensorFlow)
#
# Uso:
#   .\scripts\activar_treino.ps1
#
# O que faz:
#   1. Verifica se o venv .venv-treino existe; se não, cria.
#   2. Instala/atualiza as dependências de treino (requirements-gpu.txt).
#   3. Ativa o ambiente no PowerShell atual.
#
# Após ativar, rode o treino com:
#   python scripts/run_treino.py
#
# Para sair do ambiente:
#   deactivate
# ==============================================================================

$ErrorActionPreference = "Stop"

$venvDir = Join-Path $PSScriptRoot "..\.venv-treino"
$requirementsFile = Join-Path $PSScriptRoot "..\requirements-gpu.txt"
$pythonExe = "python"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Ativando ambiente de TREINO (TensorFlow / DirectML)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verificar se o venv existe
if (-Not (Test-Path $venvDir)) {
    Write-Host "[1/3] Criando venv de treino em: $venvDir" -ForegroundColor Yellow
    python -m venv $venvDir
    Write-Host "    Venv criado." -ForegroundColor Green
} else {
    Write-Host "[1/3] Venv de treino já existe: $venvDir" -ForegroundColor Green
}

# 2. Instalar dependências
$pipExe = Join-Path $venvDir "Scripts\pip.exe"
Write-Host "[2/3] Instalando dependências de treino..." -ForegroundColor Yellow
& $pipExe install --upgrade pip
& $pipExe install -r $requirementsFile
Write-Host "    Dependências instaladas." -ForegroundColor Green

# 3. Ativar o ambiente
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
Write-Host "[3/3] Ativando ambiente..." -ForegroundColor Yellow
& $activateScript

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Ambiente de treino ATIVO." -ForegroundColor Green
Write-Host "  Rode:  python scripts/run_treino.py" -ForegroundColor Green
Write-Host "  Sair:  deactivate" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
