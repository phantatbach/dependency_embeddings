# Dependency Embeddings

This repository provides an end-to-end workflow for extracting dependency
collocations from a parsed corpus and using them in two ways:

1. Train dependency-based embeddings with `word2vecf`.
2. Build raw count or PPMI dependency-collocation matrices.

The main workflow is in
[`depedency_embeddings.ipynb`](depedency_embeddings.ipynb). Reusable Python
functions live in `dep_colloc/`, and the bundled C implementation of
`word2vecf` lives in `w2vf/word2vecf/`.

## Repository Layout

```text
dependency_embedding/
├── dep_colloc/
│   ├── depcolloc.py   # collocation extraction and word2vecf data preparation
│   ├── freq.py        # corpus frequency counting
│   ├── matrix.py      # raw DataFrame, CSR, and Matrix Market construction
│   ├── ppmi.py        # dense-compatible and sparse PPMI calculation
│   └── utils.py       # seven-column parsing and dependency graph helpers
├── w2vf/
│   ├── README_bach.md
│   └── word2vecf/     # C source, makefile, binaries, and upstream utilities
├── depedency_embeddings.ipynb
└── README.md
```

## Corpus Format

The corpus must contain sentence boundaries and seven tab-separated columns per
token:

```text
<s id=sentence_1>
wordform<TAB>lemma<TAB>pos<TAB>id<TAB>head<TAB>deprel<TAB>morph
...
</s>
```

Example:

```text
<s id=example_1>
Dogs	dog	NOUN	1	2	nsubj	Number=Plur
run	run	VERB	2	0	root	VerbForm=Fin
</s>
```

Only the seven-column format is supported. Token IDs and head IDs must be
integers. Every sentence must have exactly one root, valid head references, no
duplicate IDs, and no dependency cycles.

Malformed input stops collocation extraction. The source filename, sentence ID,
and validation error are written to `malformed_sentences.log`.

## Python Setup

Run the notebook from the repository root so that `dep_colloc` is importable:

```bash
cd /path/to/dependency_embedding
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy tqdm matplotlib jupyter
jupyter lab depedency_embeddings.ipynb
```

`nephosem` is optional and is needed only for the notebook's `.pac` export
helper.

## Collocation Extraction

`generate_colloc_counts(...)` traverses the undirected dependency graph around
each target token up to `max_depth`. Target and context representations are
configured independently:

- `token_only`
- `lemma_only`
- `token/pos`
- `lemma/pos`

Basic usage:

```python
from dep_colloc.depcolloc import generate_colloc_counts
from dep_colloc.utils import pattern

generate_colloc_counts(
    corpus_dir="/path/to/parsed_corpus",
    output_dir="/path/to/output",
    max_depth=1,
    pattern=pattern,
    output_filename="syn_colloc_counts.txt",
    target_format="lemma_only",
    context_format="lemma_only",
    append_deprel_path=True,
)
```

The pair-count format is:

```text
target context<TAB>count
```

When `append_deprel_path=False`, contexts contain only the selected token
representation. This is the path-collocation mode.

When `append_deprel_path=True`, the directed dependency path is appended to the
context:

```text
target context/pa_nsubj<TAB>count
target context/pa_nsubj>chi_obj<TAB>count
```

`pa_` means traversal from a dependent toward its parent. `chi_` means traversal
from a parent toward its child.

Parallelism is file-based: each worker processes one complete `.txt` corpus
file and returns a local `Counter`. For a folder containing six files, more than
six workers provide no additional file-level parallelism.

## Preparing word2vecf Data

The notebook Sections 4a-4c provide independent steps for vocabulary creation,
frequency filtering, optional pair reversal, and count expansion.

### 1. Calculate pair marginals

Build `wvocab` and `cvocab` from the pair-count file, not from raw corpus token
frequency:

```python
from dep_colloc.depcolloc import calculate_colloc_column_frequencies

wvocab_path, cvocab_path = calculate_colloc_column_frequencies(
    input_path="syn_colloc_counts.txt",
    first_col_freq_path="wvocab.txt",
    second_col_freq_path="cvocab.txt",
)
```

The first-column marginal is the frequency of the first training item. The
second-column marginal is the frequency of the second training item.

### 2. Filter vocabularies and pairs

Apply the same threshold to both vocabularies, filter the pair file, and then
recalculate marginals from the surviving pairs:

```python
from dep_colloc.depcolloc import (
    calculate_colloc_column_frequencies,
    filter_colloc_counts_by_vocabs,
    filter_vocab_by_frequency,
)

wvocab_filtered = filter_vocab_by_frequency(
    min_frequency=100,
    vocab_path="wvocab.txt",
    vocab_filtered_path="wvocab_filtered.txt",
)
cvocab_filtered = filter_vocab_by_frequency(
    min_frequency=100,
    vocab_path="cvocab.txt",
    vocab_filtered_path="cvocab_filtered.txt",
)
filtered_pairs = filter_colloc_counts_by_vocabs(
    syn_colloc_counts_path="syn_colloc_counts.txt",
    wvocab_path=wvocab_filtered,
    cvocab_path=cvocab_filtered,
    syn_colloc_counts_filtered_path="syn_colloc_counts_filtered.txt",
)
calculate_colloc_column_frequencies(
    input_path=filtered_pairs,
    first_col_freq_path="wvocab_filtered.txt",
    second_col_freq_path="cvocab_filtered.txt",
)
```

### 3. Choose pair direction

`word2vecf` treats each training line as:

```text
word context
```

It writes vectors for the first item using `wvocab`; the second item is looked
up in `cvocab`. Therefore:

- Keep `target context/deprel` to learn vectors for target words.
- Reverse to `context/deprel target` to learn vectors for dependency-conditioned
  context items.

Use `switch_colloc_columns(...)` only when the reversed model is intended.

### 4. Expand counts

This version of `word2vecf` reads one training event per line and does not accept
a third count field. Expand the filtered counts before training:

```python
from dep_colloc.depcolloc import explode_colloc_counts

explode_colloc_counts(
    input_path="syn_colloc_counts_filtered.txt",
    output_path="dep.contexts",
)
```

A count of three becomes three identical pair lines.

## Building word2vecf

The C compiler, `make`, pthreads, and the math library are required.

```bash
cd w2vf/word2vecf
make -B word2vecf count_and_filter
```

The resulting executables are:

```text
w2vf/word2vecf/word2vecf
w2vf/word2vecf/count_and_filter
```

## Training Dependency Embeddings

Create the output directory before training. This C implementation does not
create parent directories for output files.

Use short, relative paths where practical. The current C source uses fixed-size
path buffers and should not be given arbitrarily long absolute paths.

```bash
mkdir -p model

/path/to/dependency_embedding/w2vf/word2vecf/word2vecf \
  -train data_prep/dep.contexts \
  -wvocab data_prep/wvocab_filtered.txt \
  -cvocab data_prep/cvocab_filtered.txt \
  -output model/word_vectors.txt \
  -dumpcv model/context_vectors.txt \
  -size 300 \
  -negative 15 \
  -threads 10 \
  -alpha 0.025 \
  -iters 2
```

Important arguments:

| Argument | Meaning |
|---|---|
| `-train` | Expanded `word context` training events |
| `-wvocab` | Counts for first-column items |
| `-cvocab` | Counts for second-column items |
| `-output` | First-column/word vectors |
| `-dumpcv` | Optional second-column/context vectors |
| `-size` | Embedding dimensions |
| `-negative` | Negative samples per positive pair |
| `-threads` | Training threads |
| `-alpha` | Initial learning rate |
| `-iters` | Number of complete passes over the data |
| `-binary` | `1` for binary vectors, `0` for text |

Use `-iters`, plural. Unknown arguments are not rejected, so `-iter` would be
silently ignored and training would use the default of one iteration.

The learning rate decays linearly across all iterations and does not reset at
the beginning of each iteration.

## Raw Count Matrices

Notebook Section 5 converts either syntactic or path pair counts into a
target-by-context matrix:

```python
from dep_colloc.matrix import create_raw_colloc_matrix

matrix_paths = create_raw_colloc_matrix(
    input_path="syn_colloc_counts.txt",
    output_prefix="syn_colloc_raw_matrix",
    output_format="sparse_npz",
)
```

Available formats:

| Format | Result | Recommended use |
|---|---|---|
| `dataframe` | pandas sparse DataFrame | Interactive work on manageable matrices |
| `sparse_npz` | Compressed SciPy CSR `.npz` | Compact storage and fast Python reloads |
| `matrix_market` | Streamed coordinate `.mtx` | Very large sparse pair-count files |

Saved sparse formats also create:

```text
<prefix>.rows.txt
<prefix>.columns.txt
```

Label line numbers correspond to zero-based CSR row or column indices. Matrix
Market coordinate indices are one-based according to that format's standard.

`matrix_market` performs two input passes and streams entries to disk. It keeps
the row and column label mappings in memory but does not assemble the complete
sparse matrix in RAM.

## PPMI Matrices

Notebook Section 5a computes PPMI only for nonzero count cells. Marginals are
calculated directly from the raw target-by-context count matrix.

For a pandas DataFrame:

```python
from dep_colloc.ppmi import calculate_ppmi_dataframe

ppmi_df = calculate_ppmi_dataframe(raw_count_df, min_count=1)
```

For a sparse `.npz` or Matrix Market `.mtx` file:

```python
from dep_colloc.ppmi import calculate_ppmi_sparse_file

ppmi_path = calculate_ppmi_sparse_file(
    count_matrix_path="syn_colloc_raw_matrix.npz",
    output_path="syn_colloc_ppmi_matrix.npz",
    min_count=1,
)
```

The raw matrix label files also describe the PPMI matrix because its shape and
axis ordering are preserved.

## Notebook Sections

| Section | Purpose |
|---|---|
| 1 | Configure corpus, output paths, formats, and depth |
| 2 | Optional dependency-depth quality assurance |
| 3 | Raw corpus token/lemma frequency counts |
| 4 | Generate path or syntactic pair counts |
| 4a | Calculate marginals and optionally reverse pairs |
| 4b | Filter vocabularies and pair counts |
| 4c | Expand counts into `dep.contexts` |
| 5 | Build raw count matrices |
| 5a | Convert raw matrices to PPMI |
| 6 | Optional `.pac` export helper |
| 7 | Debug file-comparison helper |

## Output Contracts

```text
syn_colloc_counts.txt
    target context/directed_dependency_path<TAB>count

path_colloc_counts.txt
    target context<TAB>count

wvocab*.txt and cvocab*.txt
    item<TAB>marginal_count

dep.contexts
    first_item second_item

*.npz
    SciPy sparse matrix

*.rows.txt and *.columns.txt
    one matrix-axis label per line
```

## Notes

- Pair order is directional. Reversing columns changes the model being trained.
- `max_depth=1` uses direct dependency neighbors. Larger depths append the full
  directed relation path when syntactic labels are enabled.
- `syn_colloc_counts.txt` and `path_colloc_counts.txt` share the same pair-count
  contract; syntactic counts add the dependency path to the context item.
- Corpus token frequency and pair-column marginals are different quantities.
  Use pair marginals for `word2vecf` vocabularies.
- Full-corpus extraction and Matrix Market export can require substantial time
  and disk space even when the final matrix is sparse.

## word2vecf Attribution

The bundled `word2vecf` implementation is based on the software used for
dependency-based word embeddings by Omer Levy and Yoav Goldberg (2014), itself
derived from the original word2vec implementation. Its license is included at
[`w2vf/LICENSE`](w2vf/LICENSE), and upstream usage notes are retained under
`w2vf/word2vecf/`.
