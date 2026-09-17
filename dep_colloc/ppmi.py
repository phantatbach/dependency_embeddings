import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from scipy import io, sparse
from dep_colloc.utils import VALID_TOKEN_FORMATS


def _calculate_ppmi_sparse(
    count_matrix: sparse.spmatrix,
    min_count: Optional[int] = None,
) -> sparse.csr_matrix:
    """Calculate PPMI for nonzero cells of a sparse count matrix."""
    matrix = count_matrix.tocsr().astype(np.float64)
    matrix.sum_duplicates()
    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("The count matrix must have a positive total")

    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    column_sums = np.asarray(matrix.sum(axis=0)).ravel()
    coo = matrix.tocoo()
    denominators = row_sums[coo.row] * column_sums[coo.col]

    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log2((coo.data * total) / denominators)

    keep = np.isfinite(values) & (values > 0)
    if min_count is not None:
        keep &= coo.data >= min_count

    ppmi = sparse.coo_matrix(
        (values[keep], (coo.row[keep], coo.col[keep])),
        shape=matrix.shape,
    ).tocsr()
    ppmi.eliminate_zeros()
    return ppmi


def calculate_ppmi_dataframe(
    count_df: pd.DataFrame,
    min_count: Optional[int] = None,
) -> pd.DataFrame:
    """Convert a dense or pandas-sparse count DataFrame to sparse PPMI."""
    if all(isinstance(dtype, pd.SparseDtype) for dtype in count_df.dtypes):
        count_matrix = count_df.sparse.to_coo().tocsr()
    else:
        count_matrix = sparse.csr_matrix(count_df.to_numpy())

    ppmi_matrix = _calculate_ppmi_sparse(count_matrix, min_count=min_count)
    return pd.DataFrame.sparse.from_spmatrix(
        ppmi_matrix,
        index=count_df.index,
        columns=count_df.columns,
    )


def calculate_ppmi_sparse_file(
    count_matrix_path: str,
    output_path: Optional[str] = None,
    min_count: Optional[int] = None,
) -> str:
    """Convert a sparse `.npz` or Matrix Market count matrix to PPMI `.npz`."""
    input_file = Path(count_matrix_path)
    output_file = (
        Path(output_path)
        if output_path
        else input_file.with_name(f"{input_file.stem}_ppmi.npz")
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if input_file.suffix == ".npz":
        count_matrix = sparse.load_npz(input_file)
    elif input_file.suffix == ".mtx":
        count_matrix = io.mmread(input_file)
    else:
        raise ValueError("count_matrix_path must end in .npz or .mtx")
    ppmi_matrix = _calculate_ppmi_sparse(count_matrix, min_count=min_count)
    sparse.save_npz(output_file, ppmi_matrix, compressed=True)
    return str(output_file)

def PPMI_colloc_df(dep_colloc_path,
                   lemma_pos_freq_path,
                   min_count=None,
                   mode=None):
    
    # 1) load collocation counts
    df = pd.read_csv(dep_colloc_path, index_col=0)

    # 2) load row frequencies f_i
    pos_freq = {}
    with open(lemma_pos_freq_path, encoding='utf-8') as f:
        for line in f:
            key, val = line.rstrip().split('\t', 1)
            pos_freq[key] = int(val)
    row_freq = (pd.Series(pos_freq)
                .reindex(df.index) # Re-index to match the index of pos_freq with that of df
                .fillna(0)
                .astype(int))

    # 3) load column frequencies f_j
    if mode not in VALID_TOKEN_FORMATS:
        raise ValueError(f"Invalid mode: must be one of {sorted(VALID_TOKEN_FORMATS)}")

    col_freq = (row_freq
                .reindex(df.columns) # Re-index to match the index of pos_freq with that of df
                .fillna(0)
                .astype(int))

    # 4) total N
    N = df.values.sum()

    # 5) prepare empty PPMI frame
    ppmi = pd.DataFrame(
        np.zeros(df.shape, dtype=float),
        index=df.index,
        columns=df.columns
    )

    # 6) compute PPMI
    for i in df.index:
        f_i = row_freq.at[i]
        if f_i == 0:
            continue
        for j in df.columns:
            n_ij = df.at[i, j]
            if min_count is not None and n_ij < min_count:
                continue
            f_j = col_freq.at[j]
            if n_ij > 0 and f_j > 0:
                pmi = np.log2((n_ij * N) / (f_i * f_j))
                ppmi.at[i, j] = max(pmi, 0.0)

    return ppmi
