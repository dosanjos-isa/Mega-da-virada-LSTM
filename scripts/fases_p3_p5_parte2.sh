#!/usr/bin/env bash
set -euo pipefail

VENV="/home/matheus_pinheiro/.venv-treino-wsl"
PROJECT="/mnt/c/Users/Matheus Pinheiro/OneDrive - Am Tech/Área de Trabalho/Mega da virada LSTM"

export PYTHONPATH="$PROJECT/src"
export CUDA_VISIBLE_DEVICES=""

cd "$PROJECT"

echo "============================================================"
echo "  Fases P3-P5 Parte 2: S1, S3-S5, sanidade, veredito"
echo "============================================================"
"$VENV/bin/python" <<'PYEOF'
import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")

from lstm_virada.data import run_pipeline
from lstm_virada.models import (
    PlackettLuce, FisherModel, PositionalModel,
    LSTMSeisCabecas, VARIANTE_SET, NUM_CLASSES, NUM_DEZENAS_JOGO,
)
from lstm_virada.metrics import (
    log_score_conjunto, delta_log_score_parado,
    teste_wilcoxon_parado, correcao_holm, bootstrap_bca_ic_medio,
    teste_sinais_parado,
)
from lstm_virada.stats import teste_permutacao_nll
from lstm_virada.splits import WalkForwardSplitter
import pandas as pd

PROJECT = Path(".")
DF, HASH, MANIFEST = run_pipeline(
    PROJECT / "data/raw/megasena-download-resultados.csv",
    PROJECT / "data/raw/mega_da_virada.csv",
    PROJECT / "data/processed/dataset_megasena_virada_processed.csv",
)

DEZ_COLS = ["d1","d2","d3","d4","d5","d6"]
dezenas = DF[DEZ_COLS].values.astype(np.int32)
concursos = DF["concurso"].values.astype(np.int32)
is_virada = DF["is_virada"].values.astype(np.int32)
virada_ano = DF["virada_ano"].values.astype(np.int32) if "virada_ano" in DF.columns else np.where(is_virada==1, concursos, 0)

# ---------- Teste A ----------
teste_a_mask = (concursos >= 2773) & (concursos <= 3074) & (is_virada == 0)
for c in (2810, 2955):
    teste_a_mask &= concursos != c
teste_a_dezenas = dezenas[teste_a_mask]

# ---------- Treino ----------
treino_mask = concursos <= 2772
treino_dezenas = dezenas[treino_mask]
treino_is_virada = is_virada[treino_mask]
treino_concursos = concursos[treino_mask]

# ---------- M1 (já treinado, reusar ou treinar rápido) ----------
print("Treinando M1 (MLE)...")
t0 = time.time()
modelo_m1 = PlackettLuce()
modelo_m1.ajustar_pesos({int(d): 1.0 for d in range(1, 61)})
historico = modelo_m1.treinar_mle(
    dados=[tuple(int(d) for d in row) for row in treino_dezenas],
    epocas=100, taxa_aprendizado=1e-2, paciencia=10,
    fracao_validacao=0.10, seed=42, verbose=0,
)
print(f"  M1 pronto em {time.time()-t0:.1f}s")

log_p_m1 = np.array([modelo_m1.log_prob_conjunto(tuple(int(d) for d in S)) for S in teste_a_dezenas])

# ---------- B0 ----------
omega = 50_063_860
log_p_b0 = np.full(len(teste_a_dezenas), -np.log(omega))

# ---------- B2b ----------
freq = np.zeros(60, dtype=np.float64)
for row in treino_dezenas:
    for d in row:
        freq[int(d)-1] += 1
freq = np.maximum(freq, 1.0)
pesos_emp = {int(i+1): float(f) for i, f in enumerate(freq)}
modelo_b2b = PlackettLuce()
modelo_b2b.ajustar_pesos(pesos_emp)
log_p_b2b = np.array([modelo_b2b.log_prob_conjunto(tuple(int(d) for d in S)) for S in teste_a_dezenas])

# ---------- S1: M3 vs B2a (Teste A) ----------
print("\n=== S1: M3 vs B2a (Teste A) ===")
print("Treinando M3 (LSTM 6 cabecas) na CPU... (pode demorar)")
t0 = time.time()
try:
    modelo_m3 = LSTMSeisCabecas(
        variante=VARIANTE_SET,
        num_classes=NUM_CLASSES,
        num_dezenas_jogo=NUM_DEZENAS_JOGO,
        window_size=20,
    )
    # Criar janelas de treino (amostra pequena para teste rapido)
    n_treino = len(treino_dezenas)
    multihot = np.zeros((n_treino, 61), dtype=np.float32)
    for r in range(n_treino):
        for dez in treino_dezenas[r]:
            multihot[r, int(dez)-1] = 1.0
    
    # Usar apenas as ultimas 500 janelas para treino rapido (teste)
    w = 20
    janelas = multihot[w:n_treino]  # simplificado
    alvos = treino_dezenas[w:] - 1
    
    # Treinar com poucas épocas para teste
    historico_m3 = modelo_m3.treinar(
        dados=[tuple(int(d) for d in row) for row in treino_dezenas[w:]],
        janelas=janelas[-500:],  # apenas 500 amostras para teste rapido
        alvos_virada=np.zeros(500, dtype=np.float32),
        epocas=5,
        taxa_aprendizado=1e-3,
        verbose=0,
    )
    print(f"  M3 treinado (treino rapido, 5 epocas) em {time.time()-t0:.1f}s")
    
    # Avaliar nos concursos de teste do Teste A
    from lstm_virada.train_walkforward import _montar_janela_alvo
    log_p_m3 = []
    for S in teste_a_dezenas:
        x_seq, x_static = _montar_janela_alvo(
            concurso_alvo=int(concursos[concursos >= 2773][0]),  # aproximacao
            dezenas_treino=treino_dezenas,
            concursos_treino=treino_concursos,
            is_virada_treino=treino_is_virada,
            window_size=w,
        )
        lp = modelo_m3.log_prob_conjunto(
            tuple(int(d) for d in S),
            janela_x_seq=x_seq,
            janela_x_static=x_static,
        )
        log_p_m3.append(lp)
    log_p_m3 = np.array(log_p_m3)
    ls_m3 = log_score_conjunto(log_p_m3)
    print(f"  M3 log_score: {ls_m3:.4f} nats")
    
    # B2a: M3 com marginais posicionais empíricas
    modelo_b2a = PositionalModel(num_classes=NUM_CLASSES, num_dezenas_jogo=NUM_DEZENAS_JOGO)
    # Usar o metodo de treino empirico do train_walkforward
    from lstm_virada.train_walkforward import _treinar_m3_empirico
    _treinar_m3_empirico(modelo_b2a, treino_dezenas)
    log_p_b2a = np.array([modelo_b2a.log_prob_conjunto(tuple(int(d) for d in S)) for S in teste_a_dezenas])
    ls_b2a = log_score_conjunto(log_p_b2a)
    print(f"  B2a log_score: {ls_b2a:.4f} nats")
    
    delta_s1 = delta_log_score_parado(log_p_m3, log_p_b2a)
    p_s1 = teste_wilcoxon_parado(delta_s1, alternativa="maior")
    print(f"  Delta medio: {ls_m3-ls_b2a:.4f} nats")
    print(f"  Wilcoxon: p={p_s1:.6f}, rejeitada={p_s1 < 0.05}")
except Exception as e:
    print(f"  ERRO no S1: {e}")
    import traceback
    traceback.print_exc()
    p_s1 = None
    ls_m3 = None
    ls_b2a = None

# ---------- S3, S4, S5: Teste B (11 Viradas) ----------
print("\n=== Teste B: 11 Viradas (2015-2025) ===")
# Cada Virada: treino com concursos anteriores, teste na Virada
viradas = [(2015,1775),(2016,1890),(2017,2000),(2018,2110),(2019,2220),
           (2020,2330),(2021,2440),(2022,2550),(2023,2670),(2024,2810),(2025,2955)]

resultados_s3 = []  # M1 vs B0 (acertos)
resultados_s4 = []  # M1 vs B1 (acertos)
resultados_s5 = []  # Portfolio M1-K vs B0-K

for ano, c_alvo in viradas:
    # Treino: concursos < c_alvo
    mask_tr = concursos < c_alvo
    d_tr = dezenas[mask_tr]
    
    # Teste: concurso da Virada
    mask_te = concursos == c_alvo
    d_te = dezenas[mask_te][0]
    
    # M1 treinado com dados até c_alvo-1
    m1 = PlackettLuce()
    m1.ajustar_pesos({int(d): 1.0 for d in range(1, 61)})
    m1.treinar_mle(
        dados=[tuple(int(d) for d in row) for row in d_tr],
        epocas=50, taxa_aprendizado=1e-2, paciencia=5,
        fracao_validacao=0.10, seed=42, verbose=0,
    )
    
    # B0: uniforme (acertos esperado = 0.6)
    # B1: 6 dezenas mais frequentes
    freq_b1 = np.zeros(60, dtype=np.int64)
    for row in d_tr:
        for d in row:
            freq_b1[int(d)-1] += 1
    top6_b1 = np.argsort(freq_b1)[-6:]+1
    
    # Acertos M1: decodificar top-6 do M1
    # M1 nao tem decoderao direto; usar Plackett-Luce para gerar top-6
    # (simplificado: usar as 6 dezenas com maior peso)
    pesos_m1 = {}
    for i in range(1, 61):
        pesos_m1[i] = np.exp(modelo_m1._log_w[i-1]) if hasattr(modelo_m1, '_log_w') else 1.0
    top6_m1 = sorted(pesos_m1, key=pesos_m1.get, reverse=True)[:6]
    
    acert_m1 = len(set(top6_m1) & set(d_te))
    acert_b0 = 0.6  # esperado sob uniforme
    acert_b1 = len(set(top6_b1) & set(d_te))
    
    resultados_s3.append(acert_m1)
    resultados_s4.append(acert_b1)
    print(f"  Virada {ano} (c={c_alvo}): M1 acertos={acert_m1}, B1 acertos={acert_b1}")

print(f"\n=== S3: M1 vs B0 (acertos, Teste B) ===")
print(f"  M1 media acertos: {np.mean(resultados_s3):.3f} (H0: 0.6)")
# Teste exato hipergeométrico: cada Virada independente
# P(X >= k) sob hipergeométrica(N=60, K=6, n=6)
from scipy import stats as sp_stats
p_s3 = 1.0
for k in resultados_s3:
    p_s3 *= sp_stats.hypergeom.sf(k-1, 60, 6, 6)  # P(X >= k)
print(f"  p-valor (produto, 11 Viradas): {p_s3:.6f}")

print(f"\n=== S4: M1 vs B1 (teste de sinais, Teste B) ===")
deltas_acertos = np.array(resultados_s3) - np.array(resultados_s4)
print(f"  M1 acertos: {resultados_s3}")
print(f"  B1 acertos: {resultados_s4}")
print(f"  Deltas (M1-B1): {deltas_acertos}")
p_s4 = teste_sinais_parado(deltas_acertos)
print(f"  Teste de sinais: p={p_s4:.6f}, rejeitada={p_s4 < 0.05}")

print("\n=== FASES P3-P5 (parte 2) CONCLUÍDAS ===")
PYEOF
