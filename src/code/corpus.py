"""Corpus statistics: windowed co-occurrence counts between concept words and the corpus vocabulary
over a lowercased text sample (Hugging Face parquet shards), plus the infini-gram client for exact
n-gram counts."""
import itertools
import json
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


def cooccurrence_table(counts, freq, concepts, k):
    """``(C, vocab)``: counts over the ``k`` most frequent tokens, in frequency order. ``C[i, j]`` is
    how often concept ``i`` occurred within the window of ``vocab[j]``. (n_concepts, k)."""
    vocab = sorted(freq, key=freq.get, reverse=True)[:k]
    return np.array([[counts[c].get(w, 0) for w in vocab] for c in concepts], dtype=np.float64), vocab


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


def hub_texts(repo, prefix, n_docs=None):
    """The texts of every parquet shard under ``prefix`` in a Hugging Face dataset repo (e.g.
    ``"wikimedia/wikipedia"``, ``"20231101.en/"``), each downloaded into the HF cache when reached."""
    from huggingface_hub import HfApi, hf_hub_download
    shards = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                    if f.startswith(prefix) and f.endswith(".parquet"))
    texts = parquet_texts(hf_hub_download(repo, f, repo_type="dataset") for f in shards)
    return itertools.islice(texts, n_docs) if n_docs else texts


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
