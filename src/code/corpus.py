"""Corpus statistics over lowercased text (Hugging Face parquet shards): an exact top-k vocabulary,
the corpus as a stream of word ids, and the full vocabulary x vocabulary window co-occurrence matrix
on the GPU; plus a reference window counter and the infini-gram client for exact n-gram counts."""
import itertools
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")                            # a token is a run of letters or digits


def _count_chunk(args):
    """Co-occurrence and frequency counters over one chunk of texts (a worker's share)."""
    texts, concepts, window = args
    counts, freq = {c: Counter() for c in concepts}, Counter()
    for text in texts:
        tokens = _TOKEN.findall(text.lower())
        freq.update(tokens)
        for i, t in enumerate(tokens):
            if t in concepts:
                counts[t].update(tokens[max(0, i - window):i]); counts[t].update(tokens[i + 1:i + window + 1])
    return counts, freq


def window_counts(texts, concepts, window=16, workers=1, chunk=2000):
    """``{concept: Counter}`` of the tokens within ``window`` positions of each occurrence of the
    concept, and the token frequencies over all ``texts``. Text is lowercased; a token is a run of
    letters or digits; concepts are lowercase strings matched exactly."""
    concepts = frozenset(concepts)
    texts = iter(texts)
    chunks = iter(lambda: list(itertools.islice(texts, chunk)), [])
    counts, freq = {c: Counter() for c in concepts}, Counter()
    with ProcessPoolExecutor(workers) as pool:
        for part, f in pool.map(_count_chunk, ((ch, concepts, window) for ch in chunks)):
            freq.update(f)
            for c in concepts:
                counts[c].update(part[c])
    return counts, freq


def _shard_freq(args):
    read, shard = args
    return Counter(t for text in read(shard) for t in _TOKEN.findall(text.lower()))


def vocabulary(shards, read, size, workers=40):
    """``(words, counts)``: the ``size`` most frequent tokens over all texts (``read(shard)`` yields
    one shard's texts) with their exact counts, in frequency order, ties alphabetical. Shards are
    counted in parallel."""
    freq = Counter()
    with ProcessPoolExecutor(workers) as pool:
        for f in pool.map(_shard_freq, ((read, s) for s in shards)):
            freq.update(f)
    words = sorted(freq, key=lambda w: (-freq[w], w))[:size]
    return np.array(words), np.array([freq[w] for w in words], dtype=np.int64)


def word_tokenizer(words):
    """A word-level tokenizer (Rust) over ``words`` (id = position): lowercase, split on anything that
    is not a letter or digit; any other token maps to ``[UNK]``."""
    from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers
    vocab = {w: i for i, w in enumerate(words)} | {"[UNK]": len(words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("[^a-z0-9]+"), behavior="removed")
    return tok


def _batches(texts, chars):
    """Consecutive texts grouped until they hold ``chars`` characters; a longer text goes alone."""
    batch, size = [], 0
    for t in texts:
        if batch and size + len(t) > chars:
            yield batch
            batch, size = [], 0
        batch.append(t); size += len(t)
    if batch:
        yield batch


def _encode_shard(args):
    read, shard, tok_json, chars = args
    os.environ["TOKENIZERS_PARALLELISM"] = "false"                # one Rust thread per worker process
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(tok_json); unk = tok.token_to_id("[UNK]")
    parts, lengths = [], []
    for batch in _batches(read(shard), chars):
        for enc in tok.encode_batch_fast(batch):                 # ids only, no offsets or token strings
            a = np.asarray(enc.ids, dtype=np.int64); a = a[a != unk]
            parts.append(a.astype(np.uint16)); lengths.append(len(a))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16), np.array(lengths, dtype=np.int64)


def encode_corpus(shards, read, tokenizer, workers=40, chars=50_000_000):
    """``(ids, lengths)``: every text as its in-vocabulary token ids, concatenated into one uint16 stream
    in shard order, and the number of kept tokens per text. Out-of-vocabulary tokens are dropped.
    Shards are encoded in parallel, each in batches of about ``chars`` characters."""
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_encode_shard, ((read, s, tokenizer.to_str(), chars) for s in shards)))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def cooccurrence_matrix(ids, docs, V, window=16, weighting="1/d", device="cuda", chunk=200_000_000):
    """The full V x V co-occurrence matrix over an id stream: for every pair of tokens within
    ``window`` positions in the same document, ``C[i, j]`` (and ``C[j, i]``) gains the weight of
    their distance d: ``"flat"`` 1, ``"linear"`` window+1-d, ``"1/d"`` 1/d. Accumulated on ``device``."""
    import torch
    weight = {"flat": lambda d: 1.0, "linear": lambda d: float(window + 1 - d), "1/d": lambda d: 1.0 / d}[weighting]
    C = torch.zeros(V * V, dtype=torch.float32, device=device)
    for start in range(0, len(ids), chunk):
        n = min(chunk, len(ids) - start)
        x = torch.as_tensor(np.asarray(ids[start:start + n + window], dtype=np.int64), device=device)
        doc = torch.as_tensor(np.asarray(docs[start:start + n + window], dtype=np.int64), device=device)
        for d in range(1, window + 1):
            m = min(n, len(x) - d)
            same = doc[:m] == doc[d:d + m]
            pair = (x[:m] * V + x[d:d + m])[same]
            C.index_add_(0, pair, torch.full((len(pair),), weight(d), device=device))
    C = C.view(V, V)
    return (C + C.T).cpu().numpy()


def load_corpus(path, window=16, weighting="1/d", chunk=5000):
    """``(C, vocab, freq, r)``: a built corpus directory's V x V window matrix (memory-mapped), its words
    in id order, their counts, and the matrix's row sums, computed once in chunks of rows and cached
    beside the matrix."""
    path, name = Path(path), f"C_w{window}_{weighting.replace('/', '')}"
    C = np.load(path / f"{name}.npy", mmap_mode="r")
    rows_file = path / f"{name}_rows.npy"
    if not rows_file.exists():
        np.save(rows_file, np.concatenate([np.asarray(C[i:i + chunk], dtype=np.float64).sum(1) for i in range(0, len(C), chunk)]))
    vocab = np.array(json.loads((path / "vocab.json").read_text()))
    return C, vocab, np.load(path / "freq.npy"), np.load(rows_file)


def log_conditional(C_ij, r_i, floor=1.0):
    """log P(j | i) = log(C_ij / r_i) for a (k, m) block of the matrix, with r_i the rows' full totals.
    A cell below ``floor`` co-occurrence weight is scored as ``floor`` (one co-occurrence), so an empty
    cell is finite."""
    return np.log(np.maximum(np.asarray(C_ij, dtype=np.float64), floor)) - np.log(np.asarray(r_i, dtype=np.float64))[:, None]


def collapse_phrases(ids, docs, phrases, V):
    """The id stream with every in-document occurrence of ``phrases[p]`` (a list of ids) replaced by the single
    id ``V + p``; longer phrases take precedence over their prefixes, occurrences never overlap. Returns
    ``(ids, docs)`` of the shorter stream."""
    ids = np.asarray(ids); docs = np.asarray(docs)
    out = ids.astype(np.uint16 if V + len(phrases) < 2 ** 16 else np.int32)
    taken = np.zeros(len(ids), dtype=bool)
    for p in sorted(range(len(phrases)), key=lambda p: -len(phrases[p])):
        phrase, k = phrases[p], len(phrases[p])
        n = len(ids) - k + 1
        match = np.ones(n, dtype=bool)
        for t, tok in enumerate(phrase):
            match &= ids[t:t + n] == tok
        match &= docs[:n] == docs[k - 1:k - 1 + n]
        last = -k
        for s in np.flatnonzero(match):
            if s < last + k or taken[s:s + k].any():
                continue
            out[s] = V + p; taken[s + 1:s + k] = True; last = s
    return out[~taken], docs[~taken]


def concept_rows(ids, docs, concepts, n_ids, window=16):
    """(len(concepts), n_ids): flat window counts of every id within ``window`` positions, same document, of
    each occurrence of each concept id; the rows of the co-occurrence matrix for those concepts only."""
    ids = np.asarray(ids); docs = np.asarray(docs)
    which = np.full(n_ids, -1, dtype=np.int64); which[list(concepts)] = np.arange(len(concepts))
    pos = np.flatnonzero(which[ids] >= 0)
    row = which[ids[pos]]
    R = np.zeros(len(concepts) * n_ids, dtype=np.float64)
    for d in range(1, window + 1):
        for q in (pos - d, pos + d):
            ok = (q >= 0) & (q < len(ids))
            ok[ok] &= docs[q[ok]] == docs[pos[ok]]
            R += np.bincount(row[ok] * n_ids + ids[q[ok]].astype(np.int64), minlength=len(R))
    return R.reshape(len(concepts), n_ids)


def tree_splits(tree):
    """(2^depth - 1, nodes) split vectors of a perfect binary tree given as levels of words: one row per
    internal node, root first, +1 on the nodes under its first child, -1 under its second, 0 elsewhere."""
    nodes = [w for level in tree for w in level]
    pos = {w: i for i, w in enumerate(nodes)}
    depth = len(tree) - 1

    def under(level, i):                                      # every node in the subtree headed by tree[level][i]
        return [w for l in range(level, depth + 1) for w in tree[l][i * 2 ** (l - level):(i + 1) * 2 ** (l - level)]]

    S = np.zeros((2 ** depth - 1, len(nodes)))
    row = 0
    for level in range(depth):
        for i in range(len(tree[level])):
            for sign, child in ((1, 2 * i), (-1, 2 * i + 1)):
                S[row, [pos[w] for w in under(level + 1, child)]] = sign
            row += 1
    return S


def coarse_to_fine(E, tree, k):
    """Alignment (1/k) ||U_k^T V_k||_F^2 in [0, 1] between the top-k PCs of the tree's node code ``E`` (nodes in
    level order) and the k coarsest split vectors: 1 when the code's leading PCs are exactly the tree's splits,
    root split first."""
    E = np.asarray(E, dtype=np.float64); E = E - E.mean(0)
    U = np.linalg.svd(E, full_matrices=False)[0][:, :k]
    S = tree_splits(tree)[:k]
    V = np.linalg.qr(S.T)[0]
    return float(np.linalg.norm(U.T @ V) ** 2 / k)


def orthogonality(E, tree):
    """Mean |cos| between each child's offset from its parent and the parent's own vector, over every parent
    below the root, in the node-centred code ``E``: 0 when the hierarchy is encoded orthogonally."""
    E = np.asarray(E, dtype=np.float64); E = E - E.mean(0)
    nodes = [w for level in tree for w in level]
    pos = {w: i for i, w in enumerate(nodes)}
    cos = []
    for level in range(1, len(tree) - 1):
        for i, parent in enumerate(tree[level]):
            p = E[pos[parent]]
            for child in tree[level + 1][2 * i:2 * i + 2]:
                d = E[pos[child]] - p
                cos.append(abs(d @ p) / (np.linalg.norm(d) * np.linalg.norm(p)))
    return float(np.mean(cos))


def binary_subtrees(children, root, depth, n, seed=0):
    """Up to ``n`` distinct perfect binary subtrees of ``depth`` levels below ``root`` in the tree ``children(node)``,
    as levels of nodes (each node's children in sorted order), drawn uniformly over the valid structures: with
    N(u, d) the number of depth-d trees at u, a child pair (a, b) is chosen with weight N(a, d-1) N(b, d-1)."""
    from functools import cache
    rng = np.random.default_rng(seed)

    @cache
    def count(node, d):
        return 1 if d == 0 else sum(w for _, w in pairs(node, d))

    def pairs(node, d):
        kids = sorted(children(node) or [], key=str)
        return [((a, b), count(a, d - 1) * count(b, d - 1)) for i, a in enumerate(kids) for b in kids[i + 1:]
                if count(a, d - 1) * count(b, d - 1) > 0]

    def grow(node, d):
        if d == 0:
            return [[node]]
        opts, weights = zip(*pairs(node, d))
        a, b = opts[rng.choice(len(opts), p=np.array(weights, float) / sum(weights))]
        return [[node]] + [l + r for l, r in zip(grow(a, d - 1), grow(b, d - 1))]

    trees, seen = [], set()
    for _ in range(50 * n if count(root, depth) else 0):
        t = grow(root, depth)
        nodes = tuple(w for level in t for w in level)
        if nodes not in seen:
            seen.add(nodes); trees.append(t)
        if len(trees) == n:
            break
    return trees


def wordnet_hierarchy(vocab):
    """WordNet's noun hyponymy contracted to the synsets named by ``vocab`` (words, most frequent first):
    ({synset: [child synsets]}, {synset: word}). A word names its most-tagged noun sense when that sense holds
    at least half of the word's WordNet tags (so at least one); a synset takes its first eligible word; a synset's
    parent is its deepest hypernym, contracted to the nearest named ancestor. Lemmas must be spelled exactly as
    the word, so capitalised proper nouns and acronyms ("Amon", "IT") never name a synset."""
    from nltk.corpus import wordnet as wn
    word = {}
    for w in vocab:
        if not (w.isalpha() and w.islower()):
            continue
        lemmas = [l for l in wn.lemmas(w) if l.name() == w]
        nouns = [l for l in lemmas if l.synset().pos() == "n"]
        if not nouns:
            continue
        top = max(nouns, key=lambda l: l.count())
        if top.count() and top.count() / sum(l.count() for l in lemmas) >= 0.5:
            word.setdefault(top.synset(), w)

    def parent(s):                                          # deepest hypernym, ties by name
        hyper = s.hypernyms()
        return min(hyper, key=lambda h: (-h.max_depth(), h.name())) if hyper else None

    children = {s: [] for s in word}
    for s in word:
        p = parent(s)
        while p is not None and p not in word:
            p = parent(p)
        if p is not None:
            children[p].append(s)
    return children, word


def read_analogies(path):
    """Mikolov's questions-words file as {family: [(a, b, c, d), ...]}, words in the file's own case."""
    out, family = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith(":"):
            family = line[1:].strip(); out[family] = []
        elif line.strip():
            out[family].append(tuple(line.split()))
    return out


def analogy_accuracy(E, index, questions, exclude_queries=True):
    """Top-1 3CosAdd accuracy of the code ``E`` (rows indexed by ``index`` {word: row}) on ``questions``
    [(a, b, c, d)]: the fraction whose nearest row by cosine to E_b - E_a + E_c, among all rows but a, b, c, is d."""
    U = np.asarray(E, dtype=np.float64)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    q = np.array([[index[w] for w in question] for question in questions])
    target = U[q[:, 1]] - U[q[:, 0]] + U[q[:, 2]]
    sims = target @ U.T
    if exclude_queries:
        np.put_along_axis(sims, q[:, :3], -np.inf, axis=1)
    return float(np.mean(sims.argmax(1) == q[:, 3]))


def parquet_texts(paths, column="text", batch=4096):
    """The ``column`` strings of parquet files, streamed file by file in row batches."""
    import pyarrow.parquet as pq
    for path in paths:
        for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=[column]):
            yield from b.column(0).to_pylist()


def at_least(texts, min_words):
    """Only the texts with at least ``min_words`` whitespace-separated words."""
    return (t for t in texts if len(t.split()) >= min_words)


def joined(texts, n):
    """Consecutive texts joined ``n`` at a time into one passage (for corpora stored one line per row,
    so that windows run across lines)."""
    texts = iter(texts)
    return iter(lambda: " ".join(itertools.islice(texts, n)), "")


def hub_shards(repo, prefix):
    """Local paths of every parquet shard under ``prefix`` in a Hugging Face dataset repo (e.g.
    ``"wikimedia/wikipedia"``, ``"20231101.en/"``), downloaded into the HF cache when missing."""
    from huggingface_hub import HfApi, hf_hub_download
    shards = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                    if f.startswith(prefix) and f.endswith(".parquet"))
    return [hf_hub_download(repo, f, repo_type="dataset") for f in shards]


API = "https://api.infini-gram.io/"


def query(payload, timeout=60, attempts=8, backoff=2.0):
    """POST one infini-gram query and return its response dict. Rate limiting (403/429) and network
    failures are retried with doubling waits from ``backoff`` seconds, then raised; an error
    reported by the API is raised at once."""
    data = json.dumps(payload).encode()
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(backoff * 2 ** attempt)
            continue
        if "error" in out:
            raise RuntimeError(f"infini-gram rejected {payload['query']!r}: {out['error']}")
        return out
