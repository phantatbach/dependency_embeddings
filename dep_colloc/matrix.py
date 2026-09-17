"""Build dense-compatible or sparse matrices from collocation pair counts."""

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse


MatrixOutputFormat = Literal["dataframe", "sparse_npz", "matrix_market"]
SavedMatrixPaths = Tuple[str, str, str]


def _parse_colloc_count_line(
    line: str,
    line_number: int,
    input_path: Path,
) -> tuple[str, str, int]:
    try:
        pair_text, count_text = line.rsplit("\t", 1)
        target, context = pair_text.split(" ", 1)
        count = int(count_text)
    except ValueError as exc:
        raise ValueError(
            f"Malformed collocation count line {line_number} in {input_path}: {line!r}"
        ) from exc

    if count <= 0:
        raise ValueError(
            f"Collocation count must be positive at line {line_number} "
            f"in {input_path}: {count}"
        )
    return target, context, count


def _index_colloc_items(input_path: Path) -> tuple[Dict[str, int], Dict[str, int], int]:
    row_indices: Dict[str, int] = {}
    column_indices: Dict[str, int] = {}
    entry_count = 0

    with input_path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            target, context, _ = _parse_colloc_count_line(line, line_number, input_path)
            if target not in row_indices:
                row_indices[target] = len(row_indices)
            if context not in column_indices:
                column_indices[context] = len(column_indices)
            entry_count += 1

    return row_indices, column_indices, entry_count


def _write_labels(path: Path, labels: Dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for label in labels:
            output.write(f"{label}\n")


def _build_sparse_matrix(
    input_path: Path,
    row_indices: Dict[str, int],
    column_indices: Dict[str, int],
    entry_count: int,
) -> sparse.csr_matrix:
    rows = np.empty(entry_count, dtype=np.int64)
    columns = np.empty(entry_count, dtype=np.int64)
    values = np.empty(entry_count, dtype=np.int64)

    position = 0
    with input_path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            target, context, count = _parse_colloc_count_line(
                line, line_number, input_path
            )
            rows[position] = row_indices[target]
            columns[position] = column_indices[context]
            values[position] = count
            position += 1

    matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(row_indices), len(column_indices)),
        dtype=np.int64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _write_matrix_market(
    input_path: Path,
    output_path: Path,
    row_indices: Dict[str, int],
    column_indices: Dict[str, int],
    entry_count: int,
) -> None:
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        output.write("%%MatrixMarket matrix coordinate integer general\n")
        output.write("% dependency collocation count matrix\n")
        output.write(
            f"{len(row_indices)} {len(column_indices)} {entry_count}\n"
        )
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            target, context, count = _parse_colloc_count_line(
                line, line_number, input_path
            )
            output.write(
                f"{row_indices[target] + 1} {column_indices[context] + 1} {count}\n"
            )


def create_raw_colloc_matrix(
    input_path: str,
    output_prefix: Optional[str] = None,
    output_format: MatrixOutputFormat = "sparse_npz",
) -> Union[pd.DataFrame, SavedMatrixPaths]:
    """Create a target-by-context count matrix from a collocation count file.

    The input may be either syntactic or path collocation counts and must contain
    `target context<TAB>count` lines. `dataframe` returns a pandas sparse
    DataFrame. `sparse_npz` writes a SciPy CSR matrix. `matrix_market` writes a
    coordinate-format matrix without assembling the full sparse matrix in RAM.
    Saved formats also write `.rows.txt` and `.columns.txt` label files.
    """
    if output_format not in {"dataframe", "sparse_npz", "matrix_market"}:
        raise ValueError(
            "output_format must be one of: dataframe, sparse_npz, matrix_market"
        )

    input_file = Path(input_path)
    prefix = (
        Path(output_prefix)
        if output_prefix
        else input_file.with_name(f"{input_file.stem}_matrix")
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)

    row_indices, column_indices, entry_count = _index_colloc_items(input_file)
    if not row_indices or not column_indices:
        raise ValueError(f"No collocation counts found in {input_file}")

    if output_format == "matrix_market":
        matrix_path = prefix.with_suffix(".mtx")
        rows_path = prefix.with_suffix(".rows.txt")
        columns_path = prefix.with_suffix(".columns.txt")
        _write_matrix_market(
            input_file,
            matrix_path,
            row_indices,
            column_indices,
            entry_count,
        )
        _write_labels(rows_path, row_indices)
        _write_labels(columns_path, column_indices)
        return str(matrix_path), str(rows_path), str(columns_path)

    matrix = _build_sparse_matrix(
        input_file,
        row_indices,
        column_indices,
        entry_count,
    )
    row_labels = list(row_indices)
    column_labels = list(column_indices)

    if output_format == "dataframe":
        return pd.DataFrame.sparse.from_spmatrix(
            matrix,
            index=row_labels,
            columns=column_labels,
        )

    matrix_path = prefix.with_suffix(".npz")
    rows_path = prefix.with_suffix(".rows.txt")
    columns_path = prefix.with_suffix(".columns.txt")
    sparse.save_npz(matrix_path, matrix, compressed=True)
    _write_labels(rows_path, row_indices)
    _write_labels(columns_path, column_indices)
    return str(matrix_path), str(rows_path), str(columns_path)
