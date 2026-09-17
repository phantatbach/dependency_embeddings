import os
import re
from collections import Counter, deque
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Optional, Sequence
from tqdm import tqdm
from dep_colloc.utils import (
    build_graph,
    build_token_index,
    format_token,
    is_sentence_end,
    is_sentence_start,
    parse_token_line,
)


class MalformedSentenceError(ValueError):
    """Raised when a sentence is not a valid dependency tree."""


def format_relation_path(labels: Sequence[str]) -> str:
    """Join directed dependency-edge labels from target to context."""
    return ">".join(labels)


def _get_sentence_id(sentence_tag: str) -> str:
    match = re.search(r"\bid=(?:\"([^\"]+)\"|'([^']+)'|([^\s>]+))", sentence_tag)
    if not match:
        return "<unknown>"
    return next(value for value in match.groups() if value is not None)


def _validate_dependency_sentence(
    sent_toks: Sequence[str],
    pattern: Any,
    filename: str,
    sentence_tag: str,
) -> None:
    sentence_id = _get_sentence_id(sentence_tag)
    parsed_tokens = []
    token_ids = set()

    for row_number, token_row in enumerate(sent_toks, start=1):
        token = parse_token_line(token_row, pattern)
        if token is None:
            raise MalformedSentenceError(
                f"{filename}\t{sentence_id}\tunparseable token row {row_number}: {token_row}"
            )

        if token.idx in token_ids:
            raise MalformedSentenceError(
                f"{filename}\t{sentence_id}\tduplicate token ID {token.idx}"
            )
        token_ids.add(token.idx)
        parsed_tokens.append(token)

    if not parsed_tokens:
        raise MalformedSentenceError(
            f"{filename}\t{sentence_id}\tsentence contains no valid token rows"
        )

    roots = [token.idx for token in parsed_tokens if token.head == "0"]
    if len(roots) != 1:
        raise MalformedSentenceError(
            f"{filename}\t{sentence_id}\texpected one root, found {len(roots)}: {roots}"
        )

    heads = {token.idx: token.head for token in parsed_tokens}
    for token in parsed_tokens:
        if token.head != "0" and token.head not in token_ids:
            raise MalformedSentenceError(
                f"{filename}\t{sentence_id}\ttoken ID {token.idx} points to missing head ID {token.head}"
            )

        visited = set()
        current = token.idx
        while current != "0":
            if current in visited:
                raise MalformedSentenceError(
                    f"{filename}\t{sentence_id}\tdependency cycle involving token ID {current}"
                )
            visited.add(current)
            current = heads[current]


def process_file_for_colloc(args):
    (
        filename,
        corpus_dir,
        max_depth,
        pattern,
        target_format,
        context_format,
        append_deprel_path,
    ) = args
    path = os.path.join(corpus_dir, filename)
    colloc = Counter()

    with open(path, encoding='utf-8') as f:
        sent_toks = []
        sentence_tag = ""
        for line in f:
            line = line.strip()
            if is_sentence_start(line):
                sent_toks = []
                sentence_tag = line
                continue

            if is_sentence_end(line):
                _validate_dependency_sentence(
                    sent_toks,
                    pattern,
                    filename,
                    sentence_tag,
                )
                id2lemma_pos, graph, id2deprel = build_graph(sent_toks, pattern)
                id2token = build_token_index(sent_toks, pattern)

                for sid in id2lemma_pos:
                    if sid not in id2token:
                        continue
                    target = format_token(id2token[sid], target_format)
                    seen = {sid}
                    queue = deque([(sid, 0, [])])
                    while queue:
                        curr, depth, rel_path = queue.popleft()
                        if depth >= max_depth:
                            continue
                        for nid in graph.get(curr, []):
                            if nid in seen:
                                continue
                            seen.add(nid)

                            raw_label = id2deprel.get((curr, nid), 'UNK')
                            next_rel_path = rel_path + [raw_label]
                            context = format_token(id2token[nid], context_format)
                            if append_deprel_path:
                                context = f"{context}/{format_relation_path(next_rel_path)}"

                            colloc[(target, context)] += 1
                            queue.append((nid, depth + 1, next_rel_path))

                sent_toks = []
                sentence_tag = ""
                continue

            if line:
                sent_toks.append(line)

        if sentence_tag:
            sentence_id = _get_sentence_id(sentence_tag)
            raise MalformedSentenceError(
                f"{filename}\t{sentence_id}\tmissing closing </s> tag"
            )

    return colloc


def write_colloc_counts(counts: Counter, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as fout:
        for (target, context), count in counts.items():
            fout.write(f"{target} {context}\t{count}\n")


def calculate_colloc_column_frequencies(
    input_path: str,
    first_col_freq_path: Optional[str] = None,
    second_col_freq_path: Optional[str] = None,
) -> tuple[str, str]:
    """
    Sum pair-counts by the first and second columns of a collocation count file.

    Input lines must have the form `item1 item2<TAB>count`. The output files
    contain `item<TAB>total_count`.
    """
    input_file = Path(input_path)
    first_output = Path(first_col_freq_path) if first_col_freq_path else input_file.with_name(
        f"{input_file.stem}_first_col_freq{input_file.suffix}"
    )
    second_output = Path(second_col_freq_path) if second_col_freq_path else input_file.with_name(
        f"{input_file.stem}_second_col_freq{input_file.suffix}"
    )

    first_output.parent.mkdir(parents=True, exist_ok=True)
    second_output.parent.mkdir(parents=True, exist_ok=True)

    first_counts = Counter()
    second_counts = Counter()
    with input_file.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                pair_text, count_text = line.rsplit('\t', 1)
                first_item, second_item = pair_text.split(' ', 1)
                count = int(count_text)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed collocation count line {line_number} in {input_file}: {line!r}"
                ) from exc

            first_counts[first_item] += count
            second_counts[second_item] += count

    with first_output.open('w', encoding='utf-8') as output:
        for item, count in sorted(first_counts.items(), key=lambda kv: kv[1], reverse=True):
            output.write(f"{item}\t{count}\n")

    with second_output.open('w', encoding='utf-8') as output:
        for item, count in sorted(second_counts.items(), key=lambda kv: kv[1], reverse=True):
            output.write(f"{item}\t{count}\n")

    return str(first_output), str(second_output)


def filter_colloc_by_vocab_frequency(
    min_frequency: int,
    cvocab_path: str,
    wvocab_path: str,
    syn_colloc_counts_path: str,
    cvocab_filtered_path: Optional[str] = None,
    wvocab_filtered_path: Optional[str] = None,
    syn_colloc_counts_filtered_path: Optional[str] = None,
) -> tuple[str, str, str]:
    """
    Filter syntactic collocation counts by target and context marginal frequency.

    `wvocab_path` contains target frequencies and `cvocab_path` contains context
    frequencies. The pair-count file must have `target context<TAB>count` lines.
    Outputs are final marginals recomputed from the filtered pair-count file.
    """
    syn_colloc_counts = Path(syn_colloc_counts_path)
    cvocab_filtered = Path(cvocab_filtered_path) if cvocab_filtered_path else Path(cvocab_path).with_name(
        f"{Path(cvocab_path).stem}_filtered{Path(cvocab_path).suffix}"
    )
    wvocab_filtered = Path(wvocab_filtered_path) if wvocab_filtered_path else Path(wvocab_path).with_name(
        f"{Path(wvocab_path).stem}_filtered{Path(wvocab_path).suffix}"
    )
    syn_colloc_counts_filtered = (
        Path(syn_colloc_counts_filtered_path)
        if syn_colloc_counts_filtered_path
        else syn_colloc_counts.with_name(f"{syn_colloc_counts.stem}_filtered{syn_colloc_counts.suffix}")
    )

    for output_path in (cvocab_filtered, wvocab_filtered, syn_colloc_counts_filtered):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    allowed_contexts = _read_vocab_set(cvocab_path, min_frequency)
    allowed_targets = _read_vocab_set(wvocab_path, min_frequency)

    all_pairs = []
    with syn_colloc_counts.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                pair_text, count_text = line.rsplit('\t', 1)
                target, context = pair_text.split(' ', 1)
                count = int(count_text)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed collocation count line {line_number} in {syn_colloc_counts}: {line!r}"
                ) from exc

            all_pairs.append((target, context, count))

    while True:
        pairs = [
            (target, context, count)
            for target, context, count in all_pairs
            if target in allowed_targets and context in allowed_contexts
        ]
        target_counts = Counter()
        context_counts = Counter()
        for target, context, count in pairs:
            target_counts[target] += count
            context_counts[context] += count

        next_allowed_targets = {
            target for target, count in target_counts.items() if count >= min_frequency
        }
        next_allowed_contexts = {
            context for context, count in context_counts.items() if count >= min_frequency
        }
        if next_allowed_targets == allowed_targets and next_allowed_contexts == allowed_contexts:
            break
        allowed_targets = next_allowed_targets
        allowed_contexts = next_allowed_contexts

    with syn_colloc_counts_filtered.open('w', encoding='utf-8') as output:
        for target, context, count in pairs:
            output.write(f"{target} {context}\t{count}\n")

    _write_vocab_counts(wvocab_filtered, target_counts)
    _write_vocab_counts(cvocab_filtered, context_counts)

    return str(cvocab_filtered), str(wvocab_filtered), str(syn_colloc_counts_filtered)


def filter_vocab_by_frequency(
    min_frequency: int,
    vocab_path: str,
    vocab_filtered_path: Optional[str] = None,
) -> str:
    """Write vocab entries whose frequency is at least `min_frequency`."""
    vocab = Path(vocab_path)
    filtered = Path(vocab_filtered_path) if vocab_filtered_path else vocab.with_name(
        f"{vocab.stem}_filtered{vocab.suffix}"
    )
    filtered.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    with vocab.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item, count_text = line.rsplit('\t', 1)
                count = int(count_text)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed vocab line {line_number} in {vocab}: {line!r}"
                ) from exc
            if count >= min_frequency:
                counts[item] = count

    _write_vocab_counts(filtered, counts)
    return str(filtered)


def filter_colloc_counts_by_vocabs(
    syn_colloc_counts_path: str,
    cvocab_path: str,
    wvocab_path: str,
    syn_colloc_counts_filtered_path: Optional[str] = None,
) -> str:
    """
    Filter `target context<TAB>count` lines using target and context vocab files.

    `wvocab_path` is matched against the first column. `cvocab_path` is matched
    against the second column.
    """
    syn_colloc_counts = Path(syn_colloc_counts_path)
    filtered = (
        Path(syn_colloc_counts_filtered_path)
        if syn_colloc_counts_filtered_path
        else syn_colloc_counts.with_name(f"{syn_colloc_counts.stem}_filtered{syn_colloc_counts.suffix}")
    )
    filtered.parent.mkdir(parents=True, exist_ok=True)

    allowed_contexts = _read_vocab_items(cvocab_path)
    allowed_targets = _read_vocab_items(wvocab_path)

    with syn_colloc_counts.open(encoding='utf-8') as source, filtered.open('w', encoding='utf-8') as output:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                pair_text, count_text = line.rsplit('\t', 1)
                target, context = pair_text.split(' ', 1)
                int(count_text)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed collocation count line {line_number} in {syn_colloc_counts}: {line!r}"
                ) from exc
            if target in allowed_targets and context in allowed_contexts:
                output.write(f"{target} {context}\t{count_text}\n")

    return str(filtered)


def _read_vocab_set(vocab_path: str, min_frequency: int) -> set[str]:
    allowed = set()
    with Path(vocab_path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item, count_text = line.rsplit('\t', 1)
                count = int(count_text)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed vocab line {line_number} in {vocab_path}: {line!r}"
                ) from exc

            if count >= min_frequency:
                allowed.add(item)
    return allowed


def _read_vocab_items(vocab_path: str) -> set[str]:
    items = set()
    with Path(vocab_path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item, count_text = line.rsplit('\t', 1)
                int(count_text)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed vocab line {line_number} in {vocab_path}: {line!r}"
                ) from exc
            items.add(item)
    return items


def _write_vocab_counts(output_path: Path, counts: Counter) -> None:
    with output_path.open('w', encoding='utf-8') as output:
        for item, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            output.write(f"{item}\t{count}\n")


def switch_colloc_columns_and_count_first(
    input_path: str,
    switched_output_path: Optional[str] = None,
    first_col_freq_path: Optional[str] = None,
) -> tuple[str, str]:
    """
    Switch target/context columns in a collocation count file and count the new first column.

    Input lines must have the form `target context<TAB>count`. The switched output
    writes `context target<TAB>count`, and the frequency output sums counts by
    the switched first column.
    """
    input_file = Path(input_path)
    switched_output = Path(switched_output_path) if switched_output_path else input_file.with_name(
        f"{input_file.stem}_switched{input_file.suffix}"
    )
    first_col_freq = Path(first_col_freq_path) if first_col_freq_path else input_file.with_name(
        f"{input_file.stem}_first_col_freq{input_file.suffix}"
    )

    switched_output.parent.mkdir(parents=True, exist_ok=True)
    first_col_freq.parent.mkdir(parents=True, exist_ok=True)

    first_col_counts = Counter()
    with input_file.open(encoding='utf-8') as source, switched_output.open('w', encoding='utf-8') as switched:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                pair_text, count_text = line.rsplit('\t', 1)
                target, context = pair_text.split(' ', 1)
                count = int(count_text)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed collocation count line {line_number} in {input_file}: {line!r}"
                ) from exc

            switched.write(f"{context} {target}\t{count}\n")
            first_col_counts[context] += count

    with first_col_freq.open('w', encoding='utf-8') as output:
        for item, count in sorted(first_col_counts.items(), key=lambda kv: kv[1], reverse=True):
            output.write(f"{item}\t{count}\n")

    return str(switched_output), str(first_col_freq)


def switch_colloc_columns(
    input_path: str,
    switched_output_path: Optional[str] = None,
) -> str:
    """
    Switch target/context columns in a collocation count file.

    Input lines must have the form `target context<TAB>count`. The switched
    output writes `context target<TAB>count`.
    """
    input_file = Path(input_path)
    switched_output = Path(switched_output_path) if switched_output_path else input_file.with_name(
        f"{input_file.stem}_switched{input_file.suffix}"
    )
    switched_output.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open(encoding='utf-8') as source, switched_output.open('w', encoding='utf-8') as switched:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                pair_text, count_text = line.rsplit('\t', 1)
                target, context = pair_text.split(' ', 1)
                count = int(count_text)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed collocation count line {line_number} in {input_file}: {line!r}"
                ) from exc

            switched.write(f"{context} {target}\t{count}\n")

    return str(switched_output)


def explode_colloc_counts(
    input_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Expand `item1 item2<TAB>count` pair-count lines into repeated `item1 item2` lines.
    """
    input_file = Path(input_path)
    output_file = Path(output_path) if output_path else input_file.with_name(
        f"{input_file.stem}_exploded{input_file.suffix}"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open(encoding='utf-8') as source, output_file.open('w', encoding='utf-8') as output:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                pair_text, count_text = line.rsplit('\t', 1)
                first_item, second_item = pair_text.split(' ', 1)
                count = int(count_text)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed collocation count line {line_number} in {input_file}: {line!r}"
                ) from exc

            for _ in range(count):
                output.write(f"{first_item} {second_item}\n")

    return str(output_file)


def generate_colloc_counts(
    corpus_dir,
    output_dir,
    max_depth,
    pattern,
    output_filename,
    num_workers=None,
    target_format="lemma/pos",
    context_format="lemma/pos",
    append_deprel_path=False,
    progress_desc="Colloc files",
    malformed_log_path: Optional[str] = None,
):
    """
    Write ordered target/context collocation counts from dependency paths.

    Target and context items can be formatted independently with one of:
    token_only, lemma_only, token/pos, or lemma/pos. If append_deprel_path is
    true, contexts are written as formatted_context/directed_relation_path.
    Processing stops at the first malformed sentence and writes its source file,
    sentence ID, and validation error to malformed_log_path.
    """
    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith('.txt'))
    num_workers = num_workers or cpu_count()
    log_path = (
        Path(malformed_log_path)
        if malformed_log_path
        else Path(output_dir) / "malformed_sentences.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("source_file\tsentence_id\terror\n", encoding="utf-8")
    args = [
        (
            f,
            corpus_dir,
            max_depth,
            pattern,
            target_format,
            context_format,
            append_deprel_path,
        )
        for f in files
    ]

    try:
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_file_for_colloc, args),
                                total=len(files), desc=progress_desc))
    except MalformedSentenceError as exc:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{exc}\n")
        raise MalformedSentenceError(
            f"Malformed dependency sentence detected. See {log_path}"
        ) from exc

    merged = Counter()
    for counts in results:
        merged.update(counts)

    output_path = os.path.join(output_dir, output_filename)
    print("Writing colloc counts to:", output_path)
    write_colloc_counts(merged, output_path)
    return output_path


def process_file_for_syn(args):
    return process_file_for_colloc((*args, True))


def generate_syn_colloc_df(
    corpus_dir,
    output_dir,
    max_depth,
    pattern,
    num_workers=None,
    target_format="lemma/pos",
    context_format="lemma_only",
):
    """
    Write collocation counts with directed dependency paths appended to contexts.
    """
    return generate_colloc_counts(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        max_depth=max_depth,
        pattern=pattern,
        output_filename='syn_colloc_counts.txt',
        num_workers=num_workers,
        target_format=target_format,
        context_format=context_format,
        append_deprel_path=True,
        progress_desc="Syn files",
    )

def process_file_for_path(args):
    return process_file_for_colloc((*args, False))

def generate_path_colloc_df(
    corpus_dir,
    output_dir,
    max_depth,
    pattern,
    num_workers=None,
    target_format="lemma/pos",
    context_format="lemma/pos",
):
    """
    Write collocation counts without relation labels.
    """
    return generate_colloc_counts(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        max_depth=max_depth,
        pattern=pattern,
        output_filename='path_colloc_counts.txt',
        num_workers=num_workers,
        target_format=target_format,
        context_format=context_format,
        append_deprel_path=False,
        progress_desc="Path files",
    )
