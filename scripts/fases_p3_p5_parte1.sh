#!/usr/bin/env bash
set -euo pipefail

VENV="/home/matheus_pinheiro/.venv-treino-wsl"
PROJECT="/mnt/c/Users/Matheus Pinheiro/OneDrive - Am Tech/Área de Trabalho/Mega da virada LSTM"

export PYTHONPATH="$PROJECT/src"
export CUDA_VISIBLE_DEVICES=""

cd "$PROJECT"

echo "============================================================"
echo "  Fases P3-P5: Execucao direta (CPU)"
echo "============================================================"
"$VENV/bin/python" <<'PYEOF'
import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")

from lstm_virada.data import run_pipeline
from lstm_virada.models import PlackettLuce
from lstm_virada.metrics import (
    log_score_conjunto, delta_log_score_parado,
    teste_wilcoxon_parado, correcao_holm, bootstrap_bca_ic_medio,
)

PROJECT = Path(".")
DF, HASH, MANIFEST = run_pipeline(
    PROJECT / "data/raw/megasena-download-resultados.csv",
    PROJECT / "data/raw/mega_da_virada.csv",
    PROJECT / "data/processed/dataset_megasena_virada_processed.csv",
)
print(f"Dataset: {len(DF)} concursos, hash {HASH[:16]}...")

DEZ_COLS = ["d1","d2","d3","d4","d5","d6"]
dezenas = DF[DEZ_COLS].values.astype(np.int32)
concursos = DF["concurso"].values.astype(np.int32)
is_virada = DF["is_virada"].values.astype(np.int32)

# ---------- Teste A: concursos regulares 2773-3074, excluindo 2810 e 2955 ----------
teste_a_mask = (concursos >= 2773) & (concursos <= 3074) & (is_virada == 0)
for c in (2810, 2955):
    teste_a_mask &= concursos != c
teste_a_concursos = concursos[teste_a_mask]
teste_a_dezenas = dezenas[teste_a_mask]
print(f"Teste A: {len(teste_a_concursos)} concursos ({teste_a_concursos[0]}-{teste_a_concursos[-1]})")

# ---------- Treino: concursos 1-2772 ----------
treino_mask = concursos <= 2772
treino_dezenas = dezenas[treino_mask]
print(f"Treino: {len(treino_dezenas)} concursos (1-{concursos[treino_mask][-1]})")

# ---------- B0: uniforme ----------
omega = 50_063_860
log_p_b0 = np.full(len(teste_a_dezenas), -np.log(omega))
ls_b0 = log_score_conjunto(log_p_b0)
print(f"B0 log_score: {ls_b0:.4f} nats")

# ---------- B2b: M1 com pesos empíricos ----------
freq = np.zeros(60, dtype=np.float64)
for row in treino_dezenas:
    for d in row:
        freq[int(d)-1] += 1
freq = np.maximum(freq, 1.0)
pesos_emp = {int(i+1): float(f) for i, f in enumerate(freq)}
modelo_b2b = PlackettLuce()
modelo_b2b.ajustar_pesos(pesos_emp)
log_p_b2b = np.array([modelo_b2b.log_prob_conjunto(tuple(int(d) for d in S)) for S in teste_a_dezenas])
ls_b2b = log_score_conjunto(log_p_b2b)
print(f"B2b log_score: {ls_b2b:.4f} nats")

# ---------- M1: treinar com MLE ----------
print("Treinando M1 com MLE (pode demorar)...")
t0 = time.time()
modelo_m1 = PlackettLuce()
modelo_m1.ajustar_pesos({int(d): 1.0 for d in range(1, 61)})
historico = modelo_m1.treinar_mle(
    dados=[tuple(int(d) for d in row) for row in treino_dezenas],
    epocas=100,
    taxa_aprendizado=1e-2,
    paciencia=10,
    fracao_validacao=0.10,
    seed=42,
    verbose=0,
)
t1 = time.time()
print(f"M1 treinado em {t1-t0:.1f}s, {len(historico['loss'])} épocas")
log_p_m1 = np.array([modelo_m1.log_prob_conjunto(tuple(int(d) for d in S)) for S in teste_a_dezenas])
ls_m1 = log_score_conjunto(log_p_m1)
print(f"M1 log_score: {ls_m1:.4f} nats")

# ---------- H1: M1 vs B0 ----------
delta_h1 = delta_log_score_parado(log_p_m1, log_p_b0)
print(f"\n=== H1: M1 vs B0 (Teste A) ===")
print(f"  M1 LS: {ls_m1:.4f}, B0 LS: {ls_b0:.4f}, Delta medio: {ls_m1-ls_b0:.4f} nats")
p_h1 = teste_wilcoxon_parado(delta_h1, alternativa="maior")
print(f"  Wilcoxon unilateral: p={p_h1:.6f}, rejeitada (alpha=0.05)={p_h1 < 0.05}")
ic = bootstrap_bca_ic_medio(delta_h1, n_bootstrap=10000, seed=42)
print(f"  IC 95% delta: [{ic[0]:.4f}, {ic[1]:.4f}] nats/concurso")

# ---------- S2: M1 vs B2b ----------
delta_s2 = delta_log_score_parado(log_p_m1, log_p_b2b)
print(f"\n=== S2: M1 vs B2b (Teste A) ===")
print(f"  M1 LS: {ls_m1:.4f}, B2b LS: {ls_b2b:.4f}, Delta medio: {ls_m1-ls_b2b:.4f} nats")
p_s2 = teste_wilcoxon_parado(delta_s2, alternativa="maior")
print(f"  Wilcoxon unilateral: p={p_s2:.6f}, rejeitada (alpha=0.05)={p_s2 < 0.05}")
ic_s2 = bootstrap_bca_ic_medio(delta_s2, n_bootstrap=10000, seed=42)
print(f"  IC 95% delta: [{ic_s2[0]:.4f}, {ic_s2[1]:.4f}] nats/concurso")

# ---------- Correcao de Holm (S1, S2, S3, S4, S5) ----------
# S1 requer M3 (rede neural, lento na CPU) - pular por agora
# S3, S4, S5 requerem acertos/portfolio (Teste B) - pular por agora
print(f"\n=== Correcao de Holm ===")
# Apenas S2 disponível; Holm com 1 teste nao altera o p-valor
print(f"  S2 p-valor bruto: {p_s2:.6f}")
print(f"  (Holm com 1 teste: p-ajustado = p-bruto)")

# ---------- Sanidade: rótulos permutados ----------
print(f"\n=== Sanidade: rótulos permutados ===")
rng = np.random.default_rng(42)
perm_idx = rng.permutation(len(teste_a_dezenas))
log_p_m1_perm = log_p_m1[perm_idx]
delta_perm = delta_log_score_parado(log_p_m1_perm, log_p_b0)
p_perm = teste_wilcoxon_parado(delta_perm, alternativa="maior")
print(f"  M1(permutado) vs B0: p={p_perm:.6f}, rejeitada={p_perm < 0.05}")
print(f"  Esperado: NAO rejeitada (modelo nao deve vencer com rótulos permutados)")

# ---------- Sanidade: normalização ----------
print(f"\n=== Sanidade: normalização ===")
# M1 deve somar 1 no universo
from lstm_virada.models import log_prob_conjunto_todos  # se existir
# Pular: verificar se M1.soma_probabilidades() existe
try:
    soma = modelo_m1.soma_probabilidades()
    print(f"  M1 soma probabilidades: {soma:.10f} (deve ser ~1.0)")
except AttributeError:
    print("  Metodo soma_probabilidades nao disponível (pular)")

print("\n=== FASES P3-P5 (parte 1: H1, S2, sanidade) CONCLUÍDAS ===")
PYEOF
