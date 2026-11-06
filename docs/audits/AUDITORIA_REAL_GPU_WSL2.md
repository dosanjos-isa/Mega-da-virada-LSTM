# AUDITORIA FORENSE — Ambiente WSL2, GPU NVIDIA e Estado Real do Projeto

**Data da auditoria:** 06/11/2026  
**Repositório:** `C:\Users\Matheus Pinheiro\OneDrive - Am Tech\Área de Trabalho\Mega da virada LSTM`  
**Branch:** `main`  
**Último commit:** `a8f56f6 docs(spec): consolida especificações em PROJECT_SPEC.md único`  
**Remoto:** `origin` → `https://github.com/dosanjos-isa/Mega-da-virada-LSTM.git`

---

## 1. Objetivo e Metodologia

Esta auditoria tem como objetivo determinar o **estado real** do projeto Mega da Virada LSTM, especificamente sobre:
- Configuração da GPU NVIDIA (RTX 4050) no WSL2
- Ambientes Python existentes e funcionais
- Importação e execução do TensorFlow com GPU
- Execução do modelo neural `LSTMSeisCabecas` na GPU
- Estado dos scripts de configuração e ativação
- Resultado dos testes automatizados

**Critério de evidência:** Todas as afirmações são baseadas em comandos executados diretamente no sistema, com saída capturada. Nenhuma informação dos relatórios anteriores foi aceita como verdade sem verificação direta.

---

## 2. Estado do Repositório e Git

### 2.1. Identificação

| Item | Valor |
|------|-------|
| Caminho raiz | `C:\Users\Matheus Pinheiro\OneDrive - Am Tech\Área de Trabalho\Mega da virada LSTM` |
| Branch atual | `main` |
| Remote | `origin` → `https://github.com/dosanjos-isa/Mega-da-virada-LSTM.git` (fetch e push) |
| Último commit | `a8f56f6 docs(spec): consolida especificações em PROJECT_SPEC.md único` |
| Working tree | Limpo (sem modificações pendentes) |
| Arquivos staged | Nenhum |
| Arquivos não rastreados | Nenhum |

### 2.2. Diferenças em relação ao último commit

`git diff --name-status`, `git diff --cached --name-status` e `git diff --check` retornam vazio. O repositório está exatamente no estado do commit `a8f56f6`.

### 2.3. Diferenças nos arquivos relacionados à GPU

Os arquivos `scripts/activar_treino_wsl.sh`, `scripts/check_gpu.py`, `scripts/configurar_venv_wsl_gpu_native.sh` e `src/lstm_virada/data.py` estão **idem ao último commit** — nenhuma modificação não commitada. O arquivo `check_gpu_wsl.py` **não existe** no repositório (foi removido ou nunca existiu na branch atual).

---

## 3. Estado Real do WSL2

### 3.1. Sistema Operacional

```
Linux LAPTOP-DRVLAHKL 6.18.54.1-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC
Tue Oct  6 19:13:07 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux

PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
```

### 3.2. GPU NVIDIA detectada pelo sistema

```
NVIDIA-SMI 580.110                Driver Version: 581.86         CUDA Version: 13.0
GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC
   0  NVIDIA GeForce RTX 4050 ...    On  |   00000000:01:00.0  On |                  N/A
N/A   50C    P8              2W /   60W |     397MiB /   6141MiB |     28%      Default
```

- **GPU:** NVIDIA GeForce RTX 4050 Laptop
- **VRAM:** 6141 MiB
- **Driver:** 581.86
- **CUDA reportada pelo driver:** 13.0 (apenas informativa, não implica CUDA Toolkit instalado)
- **nvidia-smi:** `/usr/lib/wsl/lib/nvidia-smi`
- **Bibliotecas CUDA em `/usr/lib/wsl/lib/`:** `libcuda.so`, `libcuda.so.1`, `libcuda.so.1.1`, `libnvidia-ml.so.1`, entre outras

**Conclusão nível 1 (sistema):** ✅ **CONFIRMADO** — A GPU RTX 4050 é detectada pelo WSL2 via `nvidia-smi`.

---

## 4. Estado Real dos Ambientes Python

Foram inspecionados quatro possíveis ambientes. A tabela abaixo resume o resultado:

| Ambiente | Existe? | Python | TensorFlow | Importação TF | GPU funcional? |
|---|---|---|---|---|---|
| Windows `.venv-treino` | ✅ Sim | 3.11.9 | 2.21.0 | ✅ Sim | ❌ Não (CPU only) |
| WSL `.venv-treino-wsl` | ✅ Sim | 3.10.12 | ❌ Não instalado | N/A | N/A |
| WSL `.venv` | ❌ Não existe | N/A | N/A | N/A | N/A |
| WSL `~/.venvs/lstm-virada-tf221` | ❌ Não existe | N/A | N/A | N/A | N/A |

### 4.1. Windows `.venv-treino`

**Localização:** `C:\Users\Matheus Pinheiro\OneDrive - Am Tech\Área de Trabalho\Mega da virada LSTM\.venv-treino`

- **Python:** 3.11.9
- **pip:** 26.3
- **TensorFlow:** 2.21.0 (instalado)
- **lstm-virada:** 0.1.0 (modo editável)
- **GPU reconhecida pelo TensorFlow:** `[]` (nenhuma)
- **Mensagem do TensorFlow:** *"TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin."*

**Conclusão:** O TensorFlow funciona, mas **apenas em CPU**. Não há suporte a GPU no Windows nativo para TF >= 2.11 sem DirectML (que não está instalado).

### 4.2. WSL `.venv-treino-wsl`

**Localização:** `C:\Users\Matheus Pinheiro\OneDrive - Am Tech\Área de Trabalho\Mega da virada LSTM\.venv-treino-wsl` (acessível via `/mnt/c/...`)

- **Python:** 3.10.12
- **TensorFlow:** **NÃO INSTALADO**
- **Pacotes instalados:** Apenas pacotes do sistema Ubuntu (attrs, bcrypt, cloud-init, cryptography, django, etc. — nenhum relacionado a ML)
- **pip list:** Não contém `tensorflow`, `keras`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyyaml`

**Conclusão:** O ambiente existe como diretório, mas **não tem TensorFlow nem nenhuma dependência de ML instalada**. Não pode ser usado para treinamento.

### 4.3. WSL `~/.venvs/lstm-virada-tf221`

**Localização esperada:** `/home/matheus_pinheiro/.venvs/lstm-virada-tf221`

**Resultado:** **NÃO EXISTE**. O comando `ls -ld` retorna "No such file or directory".

**Conclusão:** Este ambiente mencionado no Relatório A **não existe**. Nunca foi criado com sucesso, ou foi removido.

### 4.4. WSL `.venv`

**Resultado:** **NÃO EXISTE**.

---

## 5. Auditoria da GPU NVIDIA no WSL2

### 5.1. GPU reconhecida pelo sistema

✅ **CONFIRMADO** — `nvidia-smi` detecta a RTX 4050 no WSL2 com 6141 MiB VRAM e driver 581.86.

### 5.2. GPU reconhecida pelo TensorFlow

**Windows `.venv-treino`:**  
```
GPUs: []
```
TensorFlow 2.21.0 no Windows **não reconhece nenhuma GPU** (limitação conhecida: TF >= 2.11 não suporta GPU nativa no Windows sem DirectML).

**WSL `.venv-treino-wsl`:**  
TensorFlow não está instalado, portanto não pode reconhecer GPU.

### 5.3. Operação matemática na GPU

Não pôde ser testada porque não há nenhum ambiente com TensorFlow + GPU funcional.

### 5.4. Modelo LSTMSeisCabecas na GPU

O modelo `LSTMSeisCabecas` foi importado com sucesso no Windows `.venv-treino` (Python 3.11.9, TF 2.21.0 CPU), mas **não há GPU disponível no TensorFlow** para executar operações. Portanto, não foi possível testar forward pass, loss ou gradientes na GPU.

---

## 6. Auditoria dos Scripts de Configuração

### 6.1. `scripts/activar_treino_wsl.sh`

**Existe:** Sim  
**Está rastreado pelo Git:** Sim  
**Modificado atualmente:** Não (idem ao último commit)

**Conteúdo relevante:**
```bash
VENV_DIR="$HOME/.venvs/lstm-virada-tf221"
```

Este script ativa um ambiente em `/home/matheus_pinheiro/.venvs/lstm-virada-tf221/`, que **não existe**. O script verifica a existência e sai com erro se não encontrar:

```bash
if [ ! -d "$VENV_DIR" ]; then
    echo "ERRO: Ambiente virtual não encontrado em $VENV_DIR"
    echo "Execute primeiro: scripts/configurar_venv_wsl_gpu_native.sh"
    return 1 2>/dev/null || exit 1
fi
```

**Conclusão:** O script está correto em sua lógica, mas o ambiente alvo não existe. Não pode ser usado.

### 6.2. `scripts/check_gpu.py`

**Existe:** Sim  
**Está rastreado pelo Git:** Sim  
**Modificado atualmente:** Não (idem ao último commit)

**Conteúdo:** Implementa verificação de 3 níveis:
1. GPU no sistema (via `nvidia-smi` e `ctypes.CDLL("libcuda.so.1")`)
2. GPU no TensorFlow (via `tf.config.list_physical_devices("GPU")`)
3. Operação na GPU (matmul explícito em `tf.device("/GPU:0")`)

**Conclusão:** O script está bem implementado, mas não pode ser executado com sucesso porque não há ambiente com TensorFlow + GPU disponível.

### 6.3. `scripts/configurar_venv_wsl_gpu_native.sh`

**Existe:** Sim  
**Está rastreado pelo Git:** Sim  
**Modificado atualmente:** Não (idem ao último commit)

**Conteúdo relevante:**
- Cria venv em `$HOME/.venvs/lstm-virada-tf221/`
- Usa `python3.10 -m venv --without-pip`
- Instala pip via `get-pip.py` (não usa `apt`)
- Instala `tensorflow[and-cuda]==2.21.0`
- Instala `numpy pandas scipy scikit-learn pyyaml`
- Instala `lstm-virada` em modo editável com `--ignore-requires-python --no-deps`

**Conclusão:** O script existe e está bem escrito, mas **nunca foi executado com sucesso** (o ambiente alvo não existe). Não há evidência de que a instalação funcione.

---

## 7. Auditoria de `src/lstm_virada/data.py`

### Alegação investigada:
> "Corrigido `datetime.UTC` → `datetime.timezone.utc` para compatibilidade com Python 3.10"

### Evidência encontrada:

Busca por `datetime.UTC` no arquivo: **NÃO ENCONTRADA**  
Busca por `datetime.timezone.utc` no arquivo: **ENCONTRADA na linha 450**

```python
"criado_em": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
```

**Veredito:** A correção já está no código. O código usa `datetime.timezone.utc`, que é compatível com Python 3.10. Não há `datetime.UTC` (que só existe a partir do Python 3.11).

**Nota:** Não há evidência de quando ou por quem esta alteração foi feita. O Git mostra que o arquivo não tem modificações pendentes (está idêntico ao último commit `a8f56f6`).

---

## 8. Auditoria dos Testes

### 8.1. Configuração do pytest

O `pyproject.toml` configura:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-v --tb=short"
```

O `pythonpath = ["src"]` faz com que o pytest adicione `src/` ao `sys.path`. Isso significa que módulos em `src/lstm_virada/` são importáveis diretamente.

### 8.2. Coleta de testes

Foi executado: `python -m pytest --collect-only -q`

Resultado: 156 itens coletados, 5 erros de coleta.

Os 5 erros são:

| Teste | Arquivo | Erro |
|-------|---------|------|
| `teste_hipergeometrico_exato` | `tests/metrics/test_metrics.py` | fixture 'acertos_modelo' not found |
| `teste_sinais_parado` | `tests/metrics/test_metrics.py` | fixture 'acertos_modelo' not found |
| `teste_wilcoxon_parado` | `tests/metrics/test_metrics.py` | fixture 'acertos_modelo' not found |
| `teste_permutacao_nll` | `tests/stats/test_stats.py` | fixture 'log_p_modelo' not found |
| `teste_permutacao_acertos` | `tests/stats/test_stats.py` | fixture 'acertos_modelo' not found |

**Causa:** Estes não são testes. São funções definidas em `src/lstm_virada/stats.py` e `src/lstm_virada/metrics.py` que começam com o prefixo `teste_`. Como `src/` está no `pythonpath` do pytest, essas funções são coletadas como testes, mas elas esperam fixtures (`acertos_modelo`, `log_p_modelo`) que não existem como fixtures do pytest.

### 8.3. Execução dos testes

**Comando:** `python -m pytest tests/data/ tests/encoding/ tests/splits/ tests/baselines/ tests/metrics/ tests/stats/ -q`

| Métrica | Resultado |
|---------|-----------|
| Itens coletados | 116 (111 testes + 5 funções teste_* coletadas incorretamente) |
| Testes executados | 111 |
| Aprovados | **111** |
| Reprovados | 0 |
| Erros de execução | 0 |
| Erros de coleta | 5 |
| Ignorados | 0 |

**Os 5 erros são funções auxiliares no código-fonte, não testes reais:**
- `teste_permutacao_nll` e `teste_permutacao_acertos` em `src/lstm_virada/stats.py`
- `teste_hipergeometrico_exato`, `teste_sinais_parado`, `teste_wilcoxon_parado` em `src/lstm_virada/metrics.py`

### 8.4. Testes de modelos

**Comando:** `python -m pytest tests/models/ -q`

| Métrica | Resultado |
|---------|-----------|
| Testes executados | 42 |
| Aprovados | **42** |
| Reprovados | 0 |

Todos os modelos (PlackettLuce, Fisher, Positional, SeqPL, LSTMSeisCabecas) funcionam corretamente no Windows `.venv-treino`.

### 8.5. Testes walk-forward

**Comando:** `python -m pytest tests/train_walkforward/ -q`

**Resultado:** Excedeu timeout de 300 segundos. O teste travou, provavelmente tentando executar treinamento de modelos neurais no ambiente Windows (CPU apenas, sem GPU).

---

## 9. Matriz Comparativa dos Dois Relatórios

| Nº | Alegação | Relatório | Evidência encontrada | Veredito |
|---|---|---|---|---|
| 1 | WSL2 Ubuntu 22.04 está funcionando | Ambos | ✅ Confirmado | **CONFIRMADO** |
| 2 | RTX 4050 detectada pelo nvidia-smi | Ambos | ✅ Confirmado (6141 MiB, driver 581.86) | **CONFIRMADO** |
| 3 | Driver NVIDIA 581.86 | Ambos | ✅ Confirmado | **CONFIRMADO** |
| 4 | Windows .venv-treino usa Python 3.13.15 | A | ❌ Python 3.11.9 | **REFUTADO** |
| 5 | Windows .venv-treino usa Python 3.11.9 | B | ✅ Confirmado | **CONFIRMADO** |
| 6 | WSL .venv-treino-wsl está corrompido | Ambos | ⚠️ Não tem TensorFlow instalado | **PARCIALMENTE CONFIRMADO** |
| 7 | WSL .venv-treino-wsl foi removido | B | ❌ Ainda existe como diretório | **REFUTADO** |
| 8 | ~/.venvs/lstm-virada-tf221 foi criado | A | ❌ Não existe | **REFUTADO** |
| 9 | ~/.venvs/lstm-virada-tf221 não existe | B | ✅ Confirmado | **CONFIRMADO** |
| 10 | TensorFlow 2.21.0 instalado no WSL nativo | A | ❌ Não instalado | **REFUTADO** |
| 11 | TensorFlow reconhece RTX 4050 | A | ❌ Não reconhece (Windows) / Não instalado (WSL) | **REFUTADO** |
| 12 | Multiplicação de matrizes executa na GPU | A | ❌ Não testada com sucesso | **REFUTADO** |
| 13 | LSTMSeisCabecas executa forward pass na GPU | A | ❌ Não testada com sucesso (sem GPU no TF) | **REFUTADO** |
| 14 | scripts/check_gpu.py foi reescrito | A | ❌ Não foi modificado | **REFUTADO** |
| 15 | scripts/check_gpu.py permanece original | B | ✅ Confirmado | **CONFIRMADO** |
| 16 | Script WSL usa tensorflow[and-cuda] | A | ⚠️ Script diz isso, mas nunca foi executado com sucesso | **NÃO COMPROVADO** |
| 17 | Script WSL usa pacotes CUDA individuais | B | ❌ Script usa tensorflow[and-cuda] | **REFUTADO** |
| 18 | data.py usa datetime.timezone.utc | Ambos | ✅ Confirmado (linha 450) | **CONFIRMADO** |
| 19 | data.py foi alterado durante config WSL | A | ❌ Não há evidência de quando foi alterado | **NÃO COMPROVADO** |
| 20 | 141 testes não-TF aprovados | A | ❌ 111 aprovados | **REFUTADO** |
| 21 | 111 testes não-TF aprovados | B | ✅ Confirmado | **CONFIRMADO** |
| 22 | 5 erros de coleta pré-existentes | B | ✅ Confirmado | **CONFIRMADO** |
| 23 | Dois testes falharam e foram corrigidos | A | ❌ Não houve tal correção | **REFUTADO** |
| 24 | Ambiente Windows preservado sem alterações | A | ⚠️ Parcialmente (Windows .venv-treino existe e funciona em CPU) | **PARCIALMENTE CONFIRMADO** |
| 25 | Pronto para P3-P5 com GPU | A | ❌ Não está pronto (sem ambiente WSL com TF+GPU) | **REFUTADO** |

---

## 10. Lista de Fatos Confirmados

1. WSL2 Ubuntu 22.04.5 LTS está instalado e funcionando.
2. A GPU RTX 4050 Laptop (6141 MiB, driver 581.86) é detectada pelo `nvidia-smi` no WSL2.
3. O repositório está no commit `a8f56f6`, working tree limpo.
4. O Windows `.venv-treino` tem Python 3.11.9 e TensorFlow 2.21.0, mas **apenas CPU**.
5. O WSL `.venv-treino-wsl` existe como diretório, mas **não tem TensorFlow instalado**.
6. O `~/.venvs/lstm-virada-tf221` **não existe**.
7. O `scripts/check_gpu.py` está inalterado (versão original do commit).
8. O `scripts/configurar_venv_wsl_gpu_native.sh` existe mas nunca foi executado com sucesso.
9. O `data.py` já usa `datetime.timezone.utc` (compatível com Python 3.10).
10. 111 testes não-TF aprovados, 42 testes de modelos aprovados.
11. O modelo `LSTMSeisCabecas` importa e é instanciado corretamente no Windows `.venv-treino`.
12. O `scripts/check_gpu_wsl.py` **não existe** no repositório.

---

## 11. Lista de Afirmações Refutadas

| Afirmação | Por que está errada |
|-----------|---------------------|
| Windows .venv-treino usa Python 3.13.15 | Usa Python 3.11.9 |
| ~/.venvs/lstm-virada-tf221 existe | Não existe |
| TensorFlow reconhece GPU no WSL | Não há TensorFlow instalado no WSL |
| check_gpu.py foi reescrito | Não foi modificado (idem ao commit) |
| 141 testes aprovados | São 111 |
| Dois testes falharam e foram corrigidos | Não houve tal correção |

---

## 12. Lista de Afirmações Não Comprovadas

| Afirmação | Motivo |
|-----------|--------|
| O script `configurar_venv_wsl_gpu_native.sh` funciona com sucesso | Nunca foi executado com sucesso; o ambiente alvo não existe |
| A correção em `data.py` foi feita durante a configuração WSL | Não há evidência de quando foi feita (o arquivo está idêntico ao último commit) |

---

## 13. Problemas e Inconsistências Encontrados

1. **Nenhum ambiente WSL com TensorFlow funcional:** O WSL2 detecta a GPU, mas não há nenhum ambiente Python no WSL com TensorFlow instalado. O script de configuração existe mas nunca foi executado com sucesso.
2. **Windows .venv-treino é CPU-only:** TensorFlow 2.21.0 no Windows não suporta GPU nativa (requer DirectML, que não está instalado).
3. **check_gpu_wsl.py mencionado no relatório A não existe:** Este arquivo não está no repositório.
4. **Os 5 erros de coleta de testes são funções auxiliares:** `teste_*` em `stats.py` e `metrics.py` são coletadas pelo pytest por causa do `pythonpath = ["src"]`, mas não são testes reais.
5. **Os testes walk-forward travam:** Provavelmente tentando executar treinamento de redes neurais no Windows sem GPU.

---

## 14. Conclusão Técnica

### Nível de validação da GPU:

| Nível | Critério | Resultado |
|-------|----------|-----------|
| 1. Sistema | `nvidia-smi` detecta RTX 4050 no WSL2 | ✅ **CONFIRMADO** |
| 2. TensorFlow | `tf.config.list_physical_devices("GPU")` retorna GPU | ❌ **NÃO FUNCIONA** (não há TF com GPU em nenhum ambiente) |
| 3. Operação | Operação matemática executa em `/GPU:0` | ❌ **NÃO TESTADA** (sem TF com GPU) |
| 4. Projeto | LSTMSeisCabecas executa forward pass, loss e gradientes na GPU | ❌ **NÃO TESTADO** (sem TF com GPU) |

**Veredito final:** O projeto **NÃO está pronto para treinamento acelerado por GPU**. A GPU RTX 4050 é detectada pelo WSL2, mas não há nenhum ambiente Python no WSL com TensorFlow instalado e configurado para usar a GPU. O ambiente Windows tem TensorFlow, mas apenas em CPU.

### O que precisa ser feito para usar GPU:

1. Executar `scripts/configurar_venv_wsl_gpu_native.sh` no WSL para criar o ambiente `~/.venvs/lstm-virada-tf221/` com TensorFlow + CUDA.
2. Verificar que `tf.config.list_physical_devices("GPU")` retorna a RTX 4050.
3. Executar `scripts/check_gpu.py` para validar os 3 níveis.
4. Executar o teste do modelo `LSTMSeisCabecas` com forward pass na GPU.

**Nenhuma dessas etapas foi concluída com sucesso durante esta auditoria.**

---

## 15. Estado do Git

- Antes da auditoria: working tree limpo, commit `a8f56f6`
- Durante a auditoria: nenhuma alteração foi feita (não foram criados nem modificados arquivos, exceto o relatório este)
- Após a auditoria: working tree ainda limpo, commit `a8f56f6`

**Nenhum commit ou push foi realizado durante esta auditoria, conforme determinado nas regras.**