from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import re

VALID_TOKEN_FORMATS = {"token_only", "lemma_only", "token/pos", "lemma/pos"}


pattern = re.compile(
    r"^([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)$"
)


def is_sentence_start(line: str) -> bool:
    """Return whether a stripped corpus line is an opening sentence tag."""
    return line == "<s>" or (line.startswith("<s ") and line.endswith(">"))


def is_sentence_end(line: str) -> bool:
    """Return whether a stripped corpus line is a closing sentence tag."""
    return line == "</s>"


@dataclass(frozen=True)
class TokenInfo:
    """Parsed token fields used by the dependency-collocation pipeline."""

    wordform: str
    lemma: str
    pos: str
    idx: str
    head: str
    deprel: str
    morph: str = "_"


def parse_token_line(token: str, token_pattern=pattern) -> Optional[TokenInfo]:
    """Parse a seven-column token row with morphology/FEATS as the final column."""
    parts = token.rstrip("\n").split("\t")
    if len(parts) == 7:
        wordform, lemma, pos, idx, head, deprel, morph = parts
        if not idx.isdigit() or not head.isdigit():
            return None
        return TokenInfo(wordform, lemma, pos, idx, head, deprel, morph)

    match = token_pattern.match(token)
    if not match:
        return None

    groups = match.groups()
    if len(groups) != 7:
        return None

    wordform, lemma, pos, idx, head, deprel, morph = groups
    if not idx.isdigit() or not head.isdigit():
        return None
    return TokenInfo(wordform, lemma, pos, idx, head, deprel, morph)


def format_token(token: TokenInfo, mode: str) -> str:
    """Format a parsed token for target or context output."""
    if mode == "token_only":
        return token.wordform
    if mode == "lemma_only":
        return token.lemma
    if mode == "token/pos":
        return f"{token.wordform}/{token.pos}"
    if mode == "lemma/pos":
        return f"{token.lemma}/{token.pos}"
    raise ValueError(
        f"Mode must be one of {sorted(VALID_TOKEN_FORMATS)}, got {mode}"
    )


def build_token_index(tokens, pattern):
    """Return parsed token info keyed by token id."""
    id2token = {}
    for tok in tokens:
        parsed = parse_token_line(tok, pattern)
        if parsed:
            id2token[parsed.idx] = parsed
    return id2token


def build_graph(tokens, pattern):
    """
    Given a list of parsed-token rows and a regex pattern, build a dependency graph,
    a mapping of id to lemma/pos, and a mapping of edge to deprel.

    Returns a tuple of (id2lemma_pos, graph, id2deprel).
    """
    
    id2lemma_pos = {}
    graph        = defaultdict(list)
    id2deprel    = {}

    for tok in tokens:
        parsed = parse_token_line(tok, pattern)
        if not parsed:
            continue

        # Create a dictionary of {id: lemma/pos}
        id2lemma_pos[parsed.idx] = f"{parsed.lemma}/{parsed.pos}"
        # If the current word is not the root
        if parsed.head != "0":
            # Create an un-directional graph of with 2 dictionary entries {children: [parents]} and {parents: [children]}
            graph[parsed.idx].append(parsed.head)
            graph[parsed.head].append(parsed.idx)

            # Create a un-directional graph with 2 dictionary entries {edge: deprel}
            # chi: child-ward, pa: parent-ward
            id2deprel[(parsed.idx, parsed.head)] = f"pa_{parsed.deprel}"
            id2deprel[(parsed.head, parsed.idx)] = f"chi_{parsed.deprel}"
    return id2lemma_pos, graph, id2deprel

# def build_graph(tokens, pattern):
#     """
#     Given a list of conllu-style token lines and a compiled regex pattern,
#     returns three things:
#       1) id2lemma_pos:   {token_id: "lemma/POS"}
#       2) graph:          undirected adjacency list {id: [neighbor_id, ...], ...}
#       3) id2deprel:      {(a, b): deprel_from_line_of_a, ...}
#     """
#     id2lemma_pos = {}
#     graph = defaultdict(list)
#     id2deprel = {}
#     row_info = {}

#     # First pass: collect lemma/pos and own deprel/head
#     for tok in tokens:
#         m = pattern.match(tok)
#         if not m:
#             continue
#         _, lemma, pos, idx, head, deprel = m.groups()
#         id2lemma_pos[idx] = f"{lemma}/{pos}"
#         row_info[idx] = {'head': head, 'deprel': deprel}

#     # Second pass: build undirected graph and assign deprel labels correctly
#     for idx, info in row_info.items():
#         head = info['head']
#         if head != '0' and head in id2lemma_pos:
#             # add both directions for traversal
#             graph[idx].append(head)
#             graph[head].append(idx)
#             # deprel from dependent->head
#             id2deprel[(idx, head)] = info['deprel']
#             # deprel from head->dependent using head's own rel
#             id2deprel[(head, idx)] = row_info[head]['deprel']

#     return id2lemma_pos, graph, id2deprel
