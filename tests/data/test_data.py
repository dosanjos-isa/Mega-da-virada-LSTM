"""Testes unitários e de integração para o módulo data.py.

Cobre os requisitos de engenharia de dados:
- RT-001: Série única sem duplicação e marcação de is_virada.
- RT-002: Reconciliação e conferência dos concursos da Virada contra a Caixa.
- RT-003: Validações estritas de qualidade e integridade do pipeline.
- RT-004: Dados brutos imutáveis e exportação versionada com SHA-256.
"""

import importlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Carregamento dinâmico do pytest: evita erro de 'missing-import' em linters/IDEs
# quando o ambiente de desenvolvimento ainda não teve `pip install -r requirements.txt` executado.
pytest: Any
try:
    pytest = importlib.import_module("pytest")
except ImportError:  # pragma: no cover
    class _PytestStub:
        @staticmethod
        def fixture(func: Any) -> Any:
            return func

        @staticmethod
        def raises(expected_exception: type[Exception], match: str | None = None) -> Any:
            class _RaisesContext:
                def __enter__(self) -> "_RaisesContext":
                    return self

                def __exit__(
                    self,
                    exc_type: type[BaseException] | None,
                    exc_val: BaseException | None,
                    exc_tb: Any,
                ) -> bool:
                    if exc_type is None:
                        raise AssertionError(
                            f"Esperada exceção {expected_exception.__name__}, mas nenhuma foi levantada."
                        )
                    if not issubclass(exc_type, expected_exception):
                        return False
                    if match and exc_val is not None:
                        if not re.search(match, str(exc_val)):
                            raise AssertionError(
                                f"Exceção levantada, mas mensagem '{exc_val}' não casa com padrão '{match}'."
                            )
                    return True

            return _RaisesContext()

    pytest = _PytestStub()

from lstm_virada.data import (
    DEZENA_COLS,
    FINAL_COLUMNS,
    DataValidationError,
    compute_sha256,
    export_processed_dataset,
    load_megasena_raw,
    load_virada_raw,
    reconcile_virada_contests,
    run_pipeline,
    unify_series,
    validate_dataset_quality,
)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MEGASENA_CSV = RAW_DIR / "megasena-download-resultados.csv"
VIRADA_CSV = RAW_DIR / "mega_da_virada.csv"


# ---------------------------------------------------------------------------
# Fixtures auxiliares
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Retorna um DataFrame mínimo válido para testes de qualidade."""
    return pd.DataFrame({
        "concurso": [1, 2, 3],
        "data": pd.to_datetime(["1996-03-10", "1996-03-17", "1996-03-24"]),
        "ano": [1996, 1996, 1996],
        "d1": [4, 9, 10],
        "d2": [5, 37, 11],
        "d3": [30, 39, 29],
        "d4": [33, 41, 30],
        "d5": [41, 43, 36],
        "d6": [52, 49, 47],
        "is_virada": [0, 0, 0],
    })


# ---------------------------------------------------------------------------
# Testes de Utilidades e Hash (RT-004)
# ---------------------------------------------------------------------------
def test_compute_sha256(tmp_path: Path) -> None:
    # cobre: RT-004
    test_file = tmp_path / "test.txt"
    test_file.write_text("LSTM Mega da Virada", encoding="utf-8")
    hash_str = compute_sha256(test_file)
    assert isinstance(hash_str, str)
    assert len(hash_str) == 64
    assert hash_str == compute_sha256(test_file)


# ---------------------------------------------------------------------------
# Testes de Carga de Dados (RT-001, RT-002)
# ---------------------------------------------------------------------------
def test_load_megasena_raw_success() -> None:
    # cobre: RT-001
    assert MEGASENA_CSV.exists(), f"Arquivo não encontrado: {MEGASENA_CSV}"
    df = load_megasena_raw(MEGASENA_CSV)

    assert len(df) == 3074
    assert df["concurso"].iloc[0] == 1
    assert df["concurso"].iloc[-1] == 3074
    for col in DEZENA_COLS:
        assert col in df.columns
        assert pd.api.types.is_integer_dtype(df[col])


def test_load_megasena_raw_file_not_found(tmp_path: Path) -> None:
    # cobre: RT-001
    non_existent = tmp_path / "inexistente.csv"
    with pytest.raises(FileNotFoundError):
        load_megasena_raw(non_existent)


def test_load_virada_raw_success() -> None:
    # cobre: RT-002
    assert VIRADA_CSV.exists(), f"Arquivo não encontrado: {VIRADA_CSV}"
    df = load_virada_raw(VIRADA_CSV)

    assert len(df) == 17
    assert 2009 in df["ano"].values
    assert 2025 in df["ano"].values


# ---------------------------------------------------------------------------
# Testes de Reconciliação dos Concursos da Virada (RT-002)
# ---------------------------------------------------------------------------
def test_reconcile_virada_contests_success() -> None:
    # cobre: RT-002
    df_mega = load_megasena_raw(MEGASENA_CSV)
    df_virada = load_virada_raw(VIRADA_CSV)

    df_reconciled = reconcile_virada_contests(df_virada, df_mega)

    assert len(df_reconciled) == 17
    assert df_reconciled["concurso"].isnull().sum() == 0

    assert int(df_reconciled[df_reconciled["ano"] == 2009]["concurso"].iloc[0]) == 1140
    assert int(df_reconciled[df_reconciled["ano"] == 2023]["concurso"].iloc[0]) == 2670
    assert int(df_reconciled[df_reconciled["ano"] == 2024]["concurso"].iloc[0]) == 2810
    assert int(df_reconciled[df_reconciled["ano"] == 2025]["concurso"].iloc[0]) == 2955


def test_reconcile_virada_contests_fills_missing_via_date() -> None:
    # cobre: RT-002
    df_mega = load_megasena_raw(MEGASENA_CSV)
    df_virada = load_virada_raw(VIRADA_CSV)

    # Simula concurso 2023 com valor NaN / ausente
    df_virada_missing = df_virada.copy()
    idx_2023 = df_virada_missing[df_virada_missing["ano"] == 2023].index[0]
    df_virada_missing.at[idx_2023, "concurso"] = np.nan

    reconciled = reconcile_virada_contests(df_virada_missing, df_mega)
    concurso_2023 = reconciled[reconciled["ano"] == 2023]["concurso"].iloc[0]
    assert concurso_2023 == 2670


def test_reconcile_virada_contests_divergent_numbers_raises() -> None:
    # cobre: RT-002
    df_mega = load_megasena_raw(MEGASENA_CSV)
    df_virada = load_virada_raw(VIRADA_CSV)

    # Altera propositalmente uma dezena da Virada para provocar divergência
    df_corrompido = df_virada.copy()
    df_corrompido.at[0, "d1"] = 59

    with pytest.raises(DataValidationError, match=r"Divergência de dezenas"):
        reconcile_virada_contests(df_corrompido, df_mega)


# ---------------------------------------------------------------------------
# Testes de Unificação da Série (RT-001)
# ---------------------------------------------------------------------------
def test_unify_series_no_duplicates_and_flag() -> None:
    # cobre: RT-001
    df_mega = load_megasena_raw(MEGASENA_CSV)
    df_virada = load_virada_raw(VIRADA_CSV)
    df_virada_reconciled = reconcile_virada_contests(df_virada, df_mega)

    df_unified = unify_series(df_mega, df_virada_reconciled)

    assert len(df_unified) == len(df_mega)
    assert df_unified["concurso"].duplicated().sum() == 0

    assert (df_unified["is_virada"] == 1).sum() == 17
    assert (df_unified["is_virada"] == 0).sum() == len(df_mega) - 17

    for col in FINAL_COLUMNS:
        assert col in df_unified.columns


# ---------------------------------------------------------------------------
# Testes de Validação de Qualidade (RT-003)
# ---------------------------------------------------------------------------
def test_validate_dataset_quality_success(sample_valid_df: pd.DataFrame) -> None:
    # cobre: RT-003
    validate_dataset_quality(sample_valid_df)


def test_validate_dataset_quality_rejects_duplicates(
    sample_valid_df: pd.DataFrame,
) -> None:
    # cobre: RT-003
    bad_df = sample_valid_df.copy()
    bad_df.at[2, "concurso"] = 2
    with pytest.raises(DataValidationError, match=r"duplicados"):
        validate_dataset_quality(bad_df)


def test_validate_dataset_quality_rejects_gaps(sample_valid_df: pd.DataFrame) -> None:
    # cobre: RT-003
    bad_df = sample_valid_df.copy()
    bad_df.at[2, "concurso"] = 4
    with pytest.raises(DataValidationError, match=r"continuidade"):
        validate_dataset_quality(bad_df)


def test_validate_dataset_quality_rejects_out_of_range_low(
    sample_valid_df: pd.DataFrame,
) -> None:
    # cobre: RT-003
    bad_df = sample_valid_df.copy()
    bad_df.at[0, "d1"] = 0
    with pytest.raises(DataValidationError, match=r"intervalo válido"):
        validate_dataset_quality(bad_df)


def test_validate_dataset_quality_rejects_out_of_range_high(
    sample_valid_df: pd.DataFrame,
) -> None:
    # cobre: RT-003
    bad_df = sample_valid_df.copy()
    bad_df.at[0, "d6"] = 61
    with pytest.raises(DataValidationError, match=r"intervalo válido"):
        validate_dataset_quality(bad_df)


def test_validate_dataset_quality_rejects_repeated_numbers(
    sample_valid_df: pd.DataFrame,
) -> None:
    # cobre: RT-003
    bad_df = sample_valid_df.copy()
    bad_df.at[0, "d2"] = 4
    with pytest.raises(DataValidationError, match=r"estritamente crescentes"):
        validate_dataset_quality(bad_df)


def test_validate_dataset_quality_rejects_decreasing_dates(
    sample_valid_df: pd.DataFrame,
) -> None:
    # cobre: RT-003
    bad_df = sample_valid_df.copy()
    bad_df.at[1, "data"] = pd.to_datetime("1996-03-01")
    with pytest.raises(DataValidationError, match=r"ordem cronológica"):
        validate_dataset_quality(bad_df)


def test_validate_dataset_quality_rejects_nulls(sample_valid_df: pd.DataFrame) -> None:
    # cobre: RT-003
    bad_df = sample_valid_df.copy()
    bad_df.at[1, "d3"] = np.nan
    with pytest.raises(DataValidationError, match=r"nulos"):
        validate_dataset_quality(bad_df)


# ---------------------------------------------------------------------------
# Testes de Exportação e Versionamento (RT-004)
# ---------------------------------------------------------------------------
def test_export_processed_dataset(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    # cobre: RT-004
    out_file = tmp_path / "processed.csv"
    h = export_processed_dataset(sample_valid_df, out_file)

    assert out_file.exists()
    assert len(h) == 64
    assert h == compute_sha256(out_file)

    reloaded = pd.read_csv(out_file)
    assert len(reloaded) == len(sample_valid_df)


# ---------------------------------------------------------------------------
# Teste de Integração Ponta a Ponta (RT-001 a RT-004)
# ---------------------------------------------------------------------------
def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    # cobre: RT-001, RT-002, RT-003, RT-004
    out_processed = tmp_path / "dataset_megasena_virada_processed.csv"

    df_final, dataset_hash = run_pipeline(MEGASENA_CSV, VIRADA_CSV, out_processed)

    assert len(df_final) == 3074
    assert (df_final["is_virada"] == 1).sum() == 17

    assert out_processed.exists()
    assert len(dataset_hash) == 64
    assert dataset_hash == compute_sha256(out_processed)
