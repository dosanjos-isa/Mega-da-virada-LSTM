#!/usr/bin/env bash
set -euo pipefail

VENV="/home/matheus_pinheiro/.venv-treino-wsl"
PROJECT="/mnt/c/Users/Matheus Pinheiro/OneDrive - Am Tech/Área de Trabalho/Mega da virada LSTM"

export PYTHONPATH="$PROJECT/src"
export CUDA_VISIBLE_DEVICES=""

cd "$PROJECT"

echo "============================================================"
echo "  P5-6: Geração prospectiva 2026 (todo o histórico 1-3074)"
echo "============================================================"
"$VENV/bin/python" <<'PYEOF'
import sys, json, time
from pathlib import Path
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, "src")

from lstm_virada.data import run_pipeline
from lstm_virada.models import PlackettLuce
from lstm_virada.generation import generate_from_model, save_generation

PROJECT = Path(".")
DF, HASH, MANIFEST = run_pipeline(
    PROJECT / "data/raw/megasena-download-resultados.csv",
    PROJECT / "data/raw/mega_da_virada.csv",
    PROJECT / "data/processed/dataset_megasena_virada_processed.csv",
)

DEZ_COLS = ["d1","d2","d3","d4","d5","d6"]
dezenas = DF[DEZ_COLS].values.astype(np.int32)
concursos = DF["concurso"].values.astype(np.int32)

# ---------- Treino com TODO o histórico (1-3074) ----------
print("Treinando M1 com todo o histórico (1-3074)...")
t0 = time.time()
modelo = PlackettLuce()
modelo.ajustar_pesos({int(d): 1.0 for d in range(1, 61)})
historico = modelo.treinar_mle(
    dados=[tuple(int(d) for d in row) for row in dezenas],
    epocas=100,
    taxa_aprendizado=1e-2,
    paciencia=10,
    fracao_validacao=0.10,
    seed=42,
    verbose=0,
)
t1 = time.time()
print(f"  Treinado em {t1-t0:.1f}s, {len(historico['loss'])} épocas")

# ---------- Geração prospectiva K=10 ----------
print("\nGerando K=10 jogos (seed=42, diversidade=4)...")
resultado = generate_from_model(
    model=modelo,
    model_name="M1",
    K=10,
    max_shared_numbers=4,
    seed=42,
    dataset_hash=HASH,
    janela_inferencia=None,
)

print(f"\nJogos gerados: {len(resultado.games)}")
print(f"Saturado: {resultado.metadata.saturated}")
print(f"Tentativas: {resultado.metadata.attempts}")
for i, jogo in enumerate(resultado.games, 1):
    print(f"  Jogo {i:2d}: {sorted(jogo)}")

# ---------- Salvar artefato ----------
output_dir = PROJECT / "artifacts" / "generations"
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
caminho = save_generation(resultado, output_dir)
print(f"\nArquivo salvo: {caminho}")

# Salvar também resumo JSON
resumo = {
    "modelo": "M1",
    "K": 10,
    "seed": 42,
    "diversidade": 4,
    "dataset_hash": HASH,
    "treino_concursos": int(concursos[-1]),
    "treino_total": len(dezenas),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "jogos": [sorted(j) for j in resultado.games],
}
resumo_path = output_dir / f"resumo_prospectivo_2026_{timestamp}.json"
with open(resumo_path, "w", encoding="utf-8") as f:
    json.dump(resumo, f, indent=2, ensure_ascii=False)
print(f"Resumo salvo: {resumo_path}")

print("\n=== P5-6 CONCLUÍDO ===")
PYEOF
