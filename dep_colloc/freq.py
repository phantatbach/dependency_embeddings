import os
from collections import Counter
from multiprocessing import Pool, cpu_count
from dep_colloc.utils import (
    format_token,
    is_sentence_end,
    is_sentence_start,
    parse_token_line,
)

def count_lemma_file(path, mode):
    """
    Count token frequencies using one format mode:
    token_only, lemma_only, token/pos, or lemma/pos.
    """
    local_lemma_counts = Counter()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or is_sentence_start(line) or is_sentence_end(line):
                continue
            
            parsed = parse_token_line(line)
            if not parsed:
                continue

            key = format_token(parsed, mode)
            local_lemma_counts[key] += 1

    return local_lemma_counts

def count_lemma_parallel(corpus_path, file_ext=None, mode='lemma/pos'):
    # Get list of file
    all_files = []
    for root, _, files in os.walk(corpus_path):
        for fname in files:
            if file_ext and not fname.endswith(file_ext):
                continue
            all_files.append(os.path.join(root, fname))
    
    # Parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(count_lemma_file, [(path, mode) for path in all_files])

    # Combine counters
    total_counter = Counter()
    for counter in results:
        total_counter.update(counter)

    return dict(total_counter)


def save_freqs(freq_dict, out_folder, mode='lemma/pos'):
    """
    Write out `<mode>_freq.txt` into out_folder.
    Each line: key<TAB>frequency
    """
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{mode.replace('/', '_')}_freq.txt")
    with open(out_path, 'w', encoding='utf-8') as out:
        for key, freq in sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True):
            out.write(f"{key}\t{freq}\n")
    return out_path


def gen_lemma_freq(corpus_path, out_folder, file_ext=None, mode='lemma/pos'):
    """
    Complete pipeline: count then save.
    Returns the path to the file written.
    """
    freqs = count_lemma_parallel(corpus_path, file_ext, mode)
    return save_freqs(freqs, out_folder, mode)
