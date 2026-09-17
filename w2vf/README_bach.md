1. Create input for w2vf

1.1. Generate syntactic collocation counts with `generate_colloc_counts(...)`.

Use `append_deprel_path=True` so the context is conditioned on the directed
dependency path:

```
target context/deprel_path<TAB>count
```

1.2. Build `wvocab` and `cvocab` from the collocation pair counts, not from raw
corpus lemma frequency.

For `syn_colloc_counts.txt`:

```
target context/deprel_path<TAB>count
```

- `wvocab` = sum counts by the first column (`target`)
- `cvocab` = sum counts by the second column (`context/deprel_path`)

Use `calculate_colloc_column_frequencies(...)` for this step.

1.3. Filter `wvocab`, `cvocab`, and `syn_colloc_counts.txt` together.

Use one shared minimum frequency threshold for targets and contexts:

```
filter_vocab_by_frequency(...)
filter_colloc_counts_by_vocabs(...)
```

This writes:

```
wvocab_filtered
cvocab_filtered
syn_colloc_counts_filtered.txt
```

After filtering the pair-count file, rerun
`calculate_colloc_column_frequencies(...)` on `syn_colloc_counts_filtered.txt`
so the training pairs and vocab counts stay consistent.

1.4. Reverse the filtered collocation columns.

Use `switch_colloc_columns(...)` to create:

```
context/deprel_path target<TAB>count
```

1.5. Explode the reversed pair-count file for this `word2vecf` version.

This `word2vecf` binary reads two whitespace-separated tokens per training
event, so the count column must be expanded into repeated pair lines:

```
context/deprel_path target
context/deprel_path target
...
```

Use `explode_colloc_counts(...)` to create `dep.contexts`.

2. Train the model (terminal)

2.1. cd to the word2vecf folder

2.2. Remember to compile the word2vecf repo (one time only) by running:

```
make
```
or 

```
make -B word2vecf
```

2.2. command

```
./word2vecf \
-train /home/volt/bach/Corpora/deu_news_1995-2025/deu_news_1995-2025_lemma_depw2v/2021-2025/data_prep/dep.contexts \
-wvocab /home/volt/bach/Corpora/deu_news_1995-2025/deu_news_1995-2025_lemma_depw2v/2021-2025/data_prep/wvocab_filtered.txt \
-cvocab /home/volt/bach/Corpora/deu_news_1995-2025/deu_news_1995-2025_lemma_depw2v/2021-2025/data_prep/cvocab_filtered.txt \
-output /home/volt/bach/Corpora/deu_news_1995-2025/deu_news_1995-2025_lemma_depw2v/2021-2025/depw2v/vocab_vecs \
-size 300 \
-negative 15 \
-threads 10 \
-iters 5 \
-dumpcv /home/volt/bach/Corpora/deu_news_1995-2025/deu_news_1995-2025_lemma_depw2v/2021-2025/depw2v/context_vecs
```
