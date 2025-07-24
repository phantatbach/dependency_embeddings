import os
from collections import Counter, deque
from multiprocessing import Pool, cpu_count
import pandas as pd
from scipy.sparse import coo_matrix
from tqdm import tqdm
from dep_colloc.utils import build_graph

# Saving the dataframe can be very time and resource consuming
def reformat_deprel(label: str) -> str:
    # Bỏ 'chi_' hoặc 'pa_' nếu có
    return label.removeprefix('chi_').removeprefix('pa_')

def process_file_for_syn(args):
    filename, corpus_dir, max_depth, pattern = args
    path = os.path.join(corpus_dir, filename)
    colloc = Counter()

    with open(path, encoding='utf-8') as f:
        sent_toks = []
        for line in f:
            line = line.strip()
            if line.startswith('<s'):
                sent_toks = []
                continue

            if line.startswith('</s>'):
                # 1) Build graph-level data
                id2lemma_pos, graph, id2deprel = build_graph(sent_toks, pattern)

                # 2) BFS qua mỗi token làm target
                for sid, tgt in id2lemma_pos.items():
                    seen = {sid}
                    queue = deque([(sid, 0)])
                    while queue:
                        curr, depth = queue.popleft()
                        if depth >= max_depth:
                            continue
                        for nid in graph.get(curr, []):
                            if nid in seen:
                                continue
                            seen.add(nid)

                            # Lấy raw_label ('chi_…' hoặc 'pa_…') 
                            raw_label = (id2deprel.get((curr, nid))   # child→parent, sẽ có 'pa_…'
                                        or id2deprel.get((nid, curr))  # parent→child, sẽ có 'chi_…'
                                        or 'UNK')

                            if raw_label.startswith('chi_'):
                                # Child→dep: strip chi_
                                deprel = reformat_deprel(raw_label)
                            else:
                                # Dep→child (pa_…): parse lại dòng gốc
                                orig_line = sent_toks[int(nid)-1]
                                m = pattern.match(orig_line)
                                raw_field = m.group(6) if m else 'UNK'
                                deprel = reformat_deprel(raw_field)

                            lemma_nb = id2lemma_pos[nid].split('/')[0]
                            filler = f"{lemma_nb}/{deprel}"

                            colloc[(tgt, filler)] += 1
                            queue.append((nid, depth + 1))

                sent_toks = []
                continue

            if line:
                sent_toks.append(line)

    return colloc



def generate_syn_colloc_df(corpus_dir, output_dir,max_depth, pattern, num_workers=None):
    """
    Multiprocessing version of syn collocate: returns a sparse DataFrame and
    also writes a \"vocab context\tabcount\" file.
    """
    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith('.txt'))
    num_workers = num_workers or cpu_count()
    args = [(f, corpus_dir, max_depth, pattern) for f in files]

    # 1) Process files in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file_for_syn, args),
                            total=len(files), desc="Syn files"))

    # 2) Merge Counters
    merged = Counter()
    for c in results:
        merged.update(c)

    # 3) Write to text file: vocab\tcontext\tcount
    out_syn = os.path.join(output_dir, 'syn_colloc_counts.txt')
    print("Writing syn counts to:", out_syn)
    with open(out_syn, 'w', encoding='utf-8') as fout:
        for (rk, ck), cnt in merged.items():
            fout.write(f"{rk} {ck}\t{cnt}\n")

    # # 4) Build sparse DataFrame
    # row_keys = sorted({rk for rk, _ in merged})
    # col_keys = sorted({ck for _, ck in merged})
    # row2i = {rk: i for i, rk in enumerate(row_keys)}
    # col2j = {ck: j for j, ck in enumerate(col_keys)}

    # rows, cols, vals = [], [], []
    # for (rk, ck), cnt in merged.items():
    #     rows.append(row2i[rk])
    #     cols.append(col2j[ck])
    #     vals.append(cnt)

    # mat = coo_matrix((vals, (rows, cols)), shape=(len(row_keys), len(col_keys)))

    # df = pd.DataFrame.sparse.from_spmatrix(mat, index=row_keys, columns=col_keys)
    # df.index.name = 'lemma_pos'
    # return df

def process_file_for_path(args):
    filename, corpus_dir, max_depth, pattern = args
    path = os.path.join(corpus_dir, filename)
    counts = Counter()
    vocab = set()

    with open(path, encoding='utf-8') as f:
        sent = []
        for line in f:
            line = line.strip()
            if line.startswith('<s'):
                sent = []
            elif line.startswith('</s>'):
                id2lemma_pos, graph, _ = build_graph(sent, pattern)
                for idx, tok in id2lemma_pos.items():
                    seen = set()
                    queue = deque([(idx, 0)])
                    while queue:
                        curr, d = queue.popleft()
                        if curr in seen or d > max_depth:
                            continue
                        seen.add(curr)
                        if d > 0 and curr in id2lemma_pos:
                            pair = tuple(sorted((tok, id2lemma_pos[curr])))
                            counts[pair] += 1
                        for nb in graph.get(curr, []):
                            queue.append((nb, d+1))
                vocab.update(id2lemma_pos.values())
            elif line:
                sent.append(line)

    return counts, vocab

def generate_path_colloc_df(corpus_dir, output_dir, max_depth, pattern, num_workers=None):
    """
    Multiprocessing version of path collocate: returns a sparse DataFrame and
    also writes a \"vocab context count\" file (here context=neighbor token).
    """
    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith('.txt'))
    num_workers = num_workers or cpu_count()
    args = [(f, corpus_dir, max_depth, pattern) for f in files]

    # 1) Process files in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file_for_path, args),
                            total=len(files), desc="Path files"))

    # 2) Merge results
    merged_counts = Counter()
    all_vocab = set()
    for cnt, vcb in results:
        merged_counts.update(cnt)
        all_vocab.update(vcb)

    # 3) Write to text file
    out_path = os.path.join(output_dir, 'path_colloc_counts.txt')
    print("Writing syn counts to:", out_path)
    with open(out_path, 'w', encoding='utf-8') as fout:
        for (t1, t2), c in merged_counts.items():
            fout.write(f"{t1} {t2}\t{c}\n")

    # # 4) Build sparse DataFrame
    # vocab = sorted(all_vocab)
    # idx = {tok: i for i, tok in enumerate(vocab)}
    # rows, cols, vals = [], [], []
    # for (t1, t2), c in merged_counts.items():
    #     i, j = idx[t1], idx[t2]
    #     rows.extend([i, j])
    #     cols.extend([j, i])
    #     vals.extend([c, c])

    # mat = coo_matrix((vals, (rows, cols)), shape=(len(vocab), len(vocab)))
    # df = pd.DataFrame.sparse.from_spmatrix(mat, index=vocab, columns=vocab)
    # return df
