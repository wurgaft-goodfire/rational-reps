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


def _encode_shard(args):
    read, shard, tok_json, batch = args
    os.environ["TOKENIZERS_PARALLELISM"] = "false"                # one Rust thread per worker process
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(tok_json); unk = tok.token_to_id("[UNK]")
    parts, lengths = [], []
    texts = iter(read(shard))
    for chunk in iter(lambda: list(itertools.islice(texts, batch)), []):
        for enc in tok.encode_batch(chunk):
            a = np.asarray(enc.ids, dtype=np.int64); a = a[a != unk]
            parts.append(a.astype(np.uint16)); lengths.append(len(a))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16), np.array(lengths, dtype=np.int64)


def encode_corpus(shards, read, tokenizer, workers=40, batch=1000):
    """``(ids, lengths)``: every text as its in-vocabulary token ids, concatenated into one uint16 stream
    in shard order, and the number of kept tokens per text. Out-of-vocabulary tokens are dropped.
    Shards are encoded in parallel."""
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_encode_shard, ((read, s, tokenizer.to_str(), batch) for s in shards)))
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
