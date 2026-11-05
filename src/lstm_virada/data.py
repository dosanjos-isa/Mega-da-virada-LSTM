"""Módulo de ingestão e validação dos dados da Mega-Sena e Mega da Virada.

Responsabilidades:
- Carregar dados brutos das fontes da Caixa e da Mega da Virada.
- Reconciliar concursos de Mega da Virada contra a base oficial (RT-002).
- Unificar o histórico em uma série temporal única sem duplicatas, marcando
  is_virada = 1 para as Viradas e 0 para regulares (RT-001).
- Rotular as Mega da Virada com virada_ano (edição 2009-2025), independente do
  ano civil da data do sorteio (P0-02).
- Executar validações rigorosas de integridade e qualidade (RT-003).
- Versionar e exportar o dataset processado com hash SHA-256 e manifesto JSON
  (RT-004, P0-01, P0-03).

Cobre: RT-001, RT-002, RT-003, RT-004, P0-01, P0-02, P0-03
"""

import datetime
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

DEZENA_COLS: Final[list[str]] = [f"d{i}" for i in range(1, 7)]
FINAL_COLUMNS: Final[list[str]] = [
    "concurso", "data", "ano", "virada_ano", *DEZENA_COLS, "is_virada"
]

# Mapeamento concurso -> virada_ano (edição da Mega da Virada).
# Fonte: mega_da_virada.csv. A data do sorteio nao e usada para rotular a
# virada_ano porque a Virada de 2025 ocorreu em 2026-01-01 e as de 2009-2014
# ocorreram em 30/12; o ano civil quebraria o walk-forward.
VIRADA_ANO_MAP: Final[dict[int, int]] = {
    1140: 2009,
    1245: 2010,
    1350: 2011,
    1455: 2012,
    1560: 2013,
    1665: 2014,
    1775: 2015,
    1890: 2016,
    2000: 2017,
    2110: 2018,
    2220: 2019,
    2330: 2020,
    2440: 2021,
    2550: 2022,
    2670: 2023,
    2810: 2024,
    2955: 2025,
}


class DataValidationError(Exception):
    """Exceção levantada quando os dados violam qualquer critério de qualidade de RT-003."""

    pass


def compute_sha256(filepath: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo de forma segura em chunks de 64 KB.

    Args:
        filepath: Caminho para o arquivo em disco.

    Returns:
        String hexadecimal com o hash SHA-256.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_megasena_raw(filepath: Path) -> pd.DataFrame:
    """Carrega o histórico de concursos da Mega-Sena a partir do CSV bruto.

    Suporta tanto o formato padronizado (com colunas Ano, Concurso, Data, N1..N6)
    quanto o formato bruto original da Caixa (cabeçalho Concurso,Data,Números com
    8 campos nas linhas).

    Args:
        filepath: Caminho para o arquivo CSV da Mega-Sena.

    Returns:
        DataFrame com colunas normalizadas: ['concurso', 'data', 'ano', 'd1'..'d6'].
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    # Inspeciona a primeira linha para decidir o parser adequado
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
        first_line = f.readline().strip()

    header_parts = [p.strip() for p in first_line.split(",")]

    if len(header_parts) == 3 and "números" in header_parts[2].lower():
        # Formato bruto original da Caixa (8 campos por linha de dados)
        df = pd.read_csv(
            filepath,
            skiprows=1,
            header=None,
            names=["concurso", "data", "d1", "d2", "d3", "d4", "d5", "d6"],
            encoding="utf-8-sig",
        )
    else:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        # Normaliza nomes de colunas para minúsculas
        rename_map = {}
        for col in df.columns:
            c_lower = col.strip().lower()
            if c_lower in ["concurso", "concurso_numero"]:
                rename_map[col] = "concurso"
            elif c_lower in ["data", "data_sorteio"]:
                rename_map[col] = "data"
            elif c_lower in ["ano"]:
                rename_map[col] = "ano"
            elif c_lower in [f"n{i}" for i in range(1, 7)]:
                rename_map[col] = f"d{c_lower[1]}"
            elif c_lower in [f"d{i}" for i in range(1, 7)]:
                rename_map[col] = c_lower

        df = df.rename(columns=rename_map)

    # Conversões e ordenação das dezenas
    df["concurso"] = pd.to_numeric(df["concurso"], errors="raise").astype(int)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="raise")
    if "ano" not in df.columns:
        df["ano"] = df["data"].dt.year
    else:
        df["ano"] = pd.to_numeric(df["ano"], errors="raise").astype(int)

    for col in DEZENA_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    # Garante que as dezenas estejam em ordem crescente estrita em cada sorteio
    dezenas = df[DEZENA_COLS].values
    dezenas_sorted = np.sort(dezenas, axis=1)
    for i, col in enumerate(DEZENA_COLS):
        df[col] = dezenas_sorted[:, i]

    df = df.sort_values("concurso").reset_index(drop=True)
    return df[["concurso", "data", "ano", *DEZENA_COLS]]


def load_virada_raw(filepath: Path) -> pd.DataFrame:
    """Carrega o CSV da Mega da Virada.

    Args:
        filepath: Caminho para o arquivo CSV da Mega da Virada.

    Returns:
        DataFrame com colunas normalizadas: ['ano', 'concurso', 'd1'..'d6'].
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    rename_map = {}
    for col in df.columns:
        c_lower = col.strip().lower()
        if c_lower in ["concurso", "concurso_numero"]:
            rename_map[col] = "concurso"
        elif c_lower in ["ano"]:
            rename_map[col] = "ano"
        elif c_lower in ["data"]:
            rename_map[col] = "data"
        elif c_lower in [f"n{i}" for i in range(1, 7)]:
            rename_map[col] = f"d{c_lower[1]}"
        elif c_lower in [f"d{i}" for i in range(1, 7)]:
            rename_map[col] = c_lower

    df = df.rename(columns=rename_map)

    df["ano"] = pd.to_numeric(df["ano"], errors="raise").astype(int)

    # Trata concurso permitindo nulo temporário caso precise reconciliar via RT-002
    if "concurso" in df.columns:
        df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")

    for col in DEZENA_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    # Ordena as dezenas crescentemente
    dezenas = df[DEZENA_COLS].values
    dezenas_sorted = np.sort(dezenas, axis=1)
    for i, col in enumerate(DEZENA_COLS):
        df[col] = dezenas_sorted[:, i]

    return df


def reconcile_virada_contests(
    df_virada: pd.DataFrame, df_megasena: pd.DataFrame
) -> pd.DataFrame:
    """Reconcilia e confere os concursos da Mega da Virada contra a base da Caixa.

    Atende ao requisito RT-002:
    - Se algum número de concurso estiver vazio/nulo (como ocorria em 2023 e 2024),
      localiza o concurso realizado em 31/12 daquele ano no histórico oficial.
    - Confere se as 6 dezenas da Mega da Virada coincidem exatamente com o concurso
      correspondente na base oficial da Caixa.

    Args:
        df_virada: DataFrame carregado da Mega da Virada.
        df_megasena: DataFrame carregado do histórico geral oficial da Caixa.

    Returns:
        DataFrame da Mega da Virada com todos os concursos preenchidos e validados.
    """
    df_reconciled = df_virada.copy()

    for idx, row in df_reconciled.iterrows():
        ano = int(row["ano"])
        concurso = row["concurso"]

        # Se concurso estiver ausente ou NaN, localiza pelo dia 31/12 do ano
        if pd.isna(concurso) or concurso <= 0:
            match_caixa = df_megasena[
                (df_megasena["data"].dt.year == ano)
                & (df_megasena["data"].dt.month == 12)
                & (df_megasena["data"].dt.day == 31)
            ]
            if match_caixa.empty:
                raise DataValidationError(
                    f"[RT-002] Não foi encontrado concurso oficial em 31/12/{ano} "
                    f"para a Mega da Virada do ano {ano}."
                )
            concurso_id = int(match_caixa.iloc[0]["concurso"])
            df_reconciled.at[idx, "concurso"] = concurso_id
        else:
            concurso_id = int(concurso)

        # Valida contra o histórico oficial
        oficial = df_megasena[df_megasena["concurso"] == concurso_id]
        if oficial.empty:
            raise DataValidationError(
                f"[RT-002] Concurso {concurso_id} da Mega da Virada ({ano}) "
                f"não existe no histórico oficial da Mega-Sena."
            )

        # Confere se as dezenas batem
        dezenas_virada = [int(row[col]) for col in DEZENA_COLS]
        dezenas_oficial = [int(oficial.iloc[0][col]) for col in DEZENA_COLS]
        if dezenas_virada != dezenas_oficial:
            raise DataValidationError(
                f"[RT-002] Divergência de dezenas no concurso {concurso_id} ({ano}): "
                f"Virada={dezenas_virada} vs Oficial Caixa={dezenas_oficial}."
            )

    df_reconciled["concurso"] = df_reconciled["concurso"].astype(int)
    return df_reconciled


def unify_series(df_megasena: pd.DataFrame, df_virada: pd.DataFrame) -> pd.DataFrame:
    """Unifica a série em um histórico único sem duplicatas.

    Atende ao requisito RT-001:
    - As Mega da Virada já fazem parte do histórico da Mega-Sena e NÃO devem ser
      duplicadas.
    - O concurso correspondente apenas recebe is_virada = 1, e os demais recebem
      is_virada = 0.
    - A série é estritamente ordenada pelo número do concurso.

    Atende ao requisito P0-02:
    - Adiciona a coluna virada_ano (edição 2009-2025), independente do ano civil
      da data do sorteio.
    - Corrigi a Virada de 2025 (concurso 2955): virada_ano = 2025, não 2026.

    Args:
        df_megasena: DataFrame do histórico geral da Caixa.
        df_virada: DataFrame validado e reconciliado da Mega da Virada.

    Returns:
        DataFrame unificado com colunas:
        ['concurso', 'data', 'ano', 'virada_ano', 'd1', 'd2', 'd3', 'd4', 'd5',
        'd6', 'is_virada']
    """
    df_unified = df_megasena.copy()

    virada_contests = set(df_virada["concurso"].astype(int).tolist())

    df_unified["is_virada"] = df_unified["concurso"].apply(
        lambda c: 1 if c in virada_contests else 0
    )

    # P0-02: rotula virada_ano para os concursos de Virada; 0 para regulares.
    df_unified["virada_ano"] = df_unified["concurso"].apply(
        lambda c: VIRADA_ANO_MAP.get(int(c), 0)
    )

    # Sanidade: toda virada deve ter virada_ano > 0; todo regular deve ter 0.
    viradas = df_unified[df_unified["is_virada"] == 1]
    if (viradas["virada_ano"] == 0).any():
        bad = viradas[viradas["virada_ano"] == 0]["concurso"].tolist()
        raise DataValidationError(
            f"[P0-02] Concursos marcados como virada sem virada_ano: {bad}"
        )
    regulares = df_unified[df_unified["is_virada"] == 0]
    if (regulares["virada_ano"] != 0).any():
        bad = regulares[regulares["virada_ano"] != 0]["concurso"].tolist()
        raise DataValidationError(
            f"[P0-02] Concursos regulares com virada_ano != 0: {bad}"
        )

    # Sanidade: Virada de 2025 (concurso 2955) deve ter virada_ano = 2025.
    v2025 = df_unified[df_unified["concurso"] == 2955]
    if not v2025.empty and int(v2025.iloc[0]["virada_ano"]) != 2025:
        raise DataValidationError(
            "[P0-02] Virada de 2025 (concurso 2955) deve ter virada_ano = 2025."
        )

    df_unified = df_unified.sort_values("concurso").reset_index(drop=True)
    return df_unified[FINAL_COLUMNS]


def validate_dataset_quality(df: pd.DataFrame) -> None:
    """Executa a bateria estrita de validações de qualidade.

    Atende ao requisito RT-003:
    As validações devem interromper o pipeline com DataValidationError em caso de falha:
    1. Exatamente seis dezenas distintas por concurso (sem repetições).
    2. Todas as dezenas no intervalo [1, 60].
    3. Dezenas ordenadas de forma estritamente crescente (d1 < d2 < ... < d6).
    4. Números de concurso estritamente crescentes e contínuos sem lacunas (1 a N).
    5. Datas não decrescentes ao longo do tempo.
    6. Ausência de concursos duplicados.
    7. Ausência de valores nulos em qualquer coluna obrigatória.

    Args:
        df: DataFrame unificado a ser validado.

    Raises:
        DataValidationError: Se qualquer regra de qualidade for violada.
    """
    # 1. Checagem de valores nulos
    null_counts = df[FINAL_COLUMNS].isnull().sum()
    if null_counts.any():
        raise DataValidationError(
            f"[RT-003] Valores nulos encontrados no dataset: {null_counts.to_dict()}"
        )

    # 2. Ausência de concursos duplicados
    if df["concurso"].duplicated().any():
        dups = df[df["concurso"].duplicated()]["concurso"].tolist()
        raise DataValidationError(f"[RT-003] Concursos duplicados encontrados: {dups}")

    # 3. Números de concurso estritamente crescentes e sem lacunas
    concursos = df["concurso"].values
    n_contests = len(df)
    expected_contests = np.arange(1, n_contests + 1)
    if not np.array_equal(concursos, expected_contests):
        diff = np.where(concursos != expected_contests)[0]
        first_bad = diff[0]
        raise DataValidationError(
            f"[RT-003] Quebra na continuidade dos concursos no índice {first_bad}: "
            f"esperado {expected_contests[first_bad]}, encontrado {concursos[first_bad]}."
        )

    # 4. Dezenas no intervalo [1, 60]
    dezenas_matrix = df[DEZENA_COLS].values
    if (dezenas_matrix < 1).any() or (dezenas_matrix > 60).any():
        raise DataValidationError(
            "[RT-003] Existem dezenas fora do intervalo válido [1, 60]."
        )

    # 5. Exatamente 6 dezenas distintas e estritamente crescentes por concurso
    differences = np.diff(dezenas_matrix, axis=1)
    if (differences <= 0).any():
        invalid_rows = np.where((differences <= 0).any(axis=1))[0]
        bad_contest = concursos[invalid_rows[0]]
        raise DataValidationError(
            f"[RT-003] Dezenas não são estritamente crescentes ou contêm duplicatas "
            f"no concurso {bad_contest}: {dezenas_matrix[invalid_rows[0]]}."
        )

    # 6. Datas crescentes (não decrescentes)
    datas = pd.to_datetime(df["data"])
    if not datas.is_monotonic_increasing:
        decreasing_indices = np.where(datas.diff() < pd.Timedelta(0))[0]
        bad_idx = decreasing_indices[0]
        raise DataValidationError(
            f"[RT-003] Quebra na ordem cronológica das datas no concurso "
            f"{concursos[bad_idx]} (Data: {datas.iloc[bad_idx]} anterior a {datas.iloc[bad_idx - 1]})."
        )

    # 7. virada_ano consistente com is_virada
    if "virada_ano" in df.columns:
        viradas = df[df["is_virada"] == 1]
        if (viradas["virada_ano"] == 0).any():
            bad = viradas[viradas["virada_ano"] == 0]["concurso"].tolist()
            raise DataValidationError(
                f"[P0-02] Concursos marcados como virada sem virada_ano: {bad}"
            )
        regulares = df[df["is_virada"] == 0]
        if (regulares["virada_ano"] != 0).any():
            bad = regulares[regulares["virada_ano"] != 0]["concurso"].tolist()
            raise DataValidationError(
                f"[P0-02] Concursos regulares com virada_ano != 0: {bad}"
            )


def export_processed_dataset(
    df: pd.DataFrame,
    output_path: Path,
    raw_hashes: dict[str, str] | None = None,
    commit: str | None = None,
) -> tuple[str, dict]:
    """Exporta o dataset processado final de forma atômica e retorna seu hash SHA-256.

    Atende ao requisito RT-004 e P0-03:
    - Gravação segura e atômica via arquivo temporário.
    - Retorna o hash SHA-256 do arquivo gerado para fins de versionamento e rastreabilidade.
    - Gera um manifesto JSON ao lado do dataset com hashes, proveniência, commit,
      versões e timestamp.

    Args:
        df: DataFrame validado a ser persistido.
        output_path: Caminho de destino do arquivo processado (ex: .csv).
        raw_hashes: dicionário {caminho_raw: sha256} para registrar no manifesto.
        commit: hash do commit Git atual para rastreabilidade.

    Returns:
        Tupla (hash SHA-256 do arquivo salvo, dicionário do manifesto).
    """
    import sys

    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_path.with_suffix(f".tmp_{os.getpid()}")

    df.to_csv(temp_file, index=False, encoding="utf-8")
    os.replace(temp_file, output_path)

    dataset_hash = compute_sha256(output_path)

    # P0-03: manifesto JSON ao lado do dataset.
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "criado_em": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "dataset_processado": str(output_path),
        "sha256_dataset": dataset_hash,
        "raw_hashes": raw_hashes or {},
        "commit": commit,
        "versoes": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "nota_proveniencia": (
            "arquivo bruto megasena-download-resultados.csv ja esta normalizado; "
            "hash inicial 088db01f... registrado no notebook 00 nao confere com o "
            "atual 043a16a7...; commit 8efe765 contem a versao normalizada; "
            "original nao esta no Git. Proveniencia parcial."
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return dataset_hash, manifest


def run_pipeline(
    raw_megasena_path: Path,
    raw_virada_path: Path,
    processed_output_path: Path,
) -> tuple[pd.DataFrame, str, dict]:
    """Executa o pipeline completo de ingestão e validação (Tarefa T1).

    Orquestra os requisitos RT-001 a RT-004 e P0-01 a P0-03:
    1. Carrega dados brutos imutáveis e registra seus hashes (P0-01).
    2. Reconcilia os concursos da Mega da Virada contra a base Caixa (RT-002).
    3. Unifica a série temporal sem duplicações com flag is_virada (RT-001).
    4. Rotula as Viradas com virada_ano (P0-02).
    5. Executa a validação de qualidade estrita (RT-003).
    6. Exporta o dataset final com hash SHA-256 e manifesto JSON (RT-004, P0-03).

    Args:
        raw_megasena_path: Caminho para o CSV da Mega-Sena Caixa.
        raw_virada_path: Caminho para o CSV da Mega da Virada.
        processed_output_path: Caminho do arquivo de saída processado.

    Returns:
        Tupla (DataFrame processado e validado, hash SHA-256, manifesto).
    """
    import subprocess

    # 1. Carga dos dados brutos e registro de hashes (P0-01)
    raw_hashes = {
        str(raw_megasena_path): compute_sha256(raw_megasena_path),
        str(raw_virada_path): compute_sha256(raw_virada_path),
    }

    df_mega = load_megasena_raw(raw_megasena_path)
    df_virada = load_virada_raw(raw_virada_path)

    # 2. Reconciliação (RT-002)
    df_virada_reconciled = reconcile_virada_contests(df_virada, df_mega)

    # 3. Unificação (RT-001) + rotulagem virada_ano (P0-02)
    df_unified = unify_series(df_mega, df_virada_reconciled)

    # 4. Validação de qualidade (RT-003)
    validate_dataset_quality(df_unified)

    # 5. Exportação com hash SHA-256 e manifesto JSON (RT-004, P0-03)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None

    dataset_hash, manifest = export_processed_dataset(
        df_unified, processed_output_path, raw_hashes=raw_hashes, commit=commit
    )

    return df_unified, dataset_hash, manifest
