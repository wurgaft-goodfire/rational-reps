"""Corpus statistics: windowed co-occurrence counts between concept words and a context vocabulary
over a text sample (FineWeb-Edu parquet shards), plus the infini-gram client for exact n-gram counts."""
import itertools
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_WORD = re.compile(r"[A-Za-z]+")


def _count_chunk(args):
    """Co-occurrence and frequency counters over one chunk of texts (a worker's share)."""
    texts, lookup, window = args
    counts, freq = {c: Counter() for c in set(lookup.values())}, Counter()
    for text in texts:
        tokens = _WORD.findall(text)
        lower = [t.lower() for t in tokens]
        freq.update(lower)
        for i, t in enumerate(tokens):
            c = lookup.get(t)
            if c is not None:
                counts[c].update(lower[max(0, i - window):i]); counts[c].update(lower[i + 1:i + window + 1])
    return counts, freq


def window_counts(texts, concepts, window=16, workers=1, chunk=2000):
    """``{concept: Counter}`` of the lowercase words within ``window`` tokens of each occurrence of the
    concept, and the lowercase word frequencies over all ``texts``. A capitalised concept ("May")
    matches its own case only; a lowercase one matches either case."""
    lookup = {c: c for c in concepts}
    lookup.update({c.capitalize(): c for c in concepts if c.islower()})
    texts = iter(texts)
    chunks = iter(lambda: list(itertools.islice(texts, chunk)), [])
    counts, freq = {c: Counter() for c in concepts}, Counter()
    with ProcessPoolExecutor(workers) as pool:
        for part, f in pool.map(_count_chunk, ((ch, lookup, window) for ch in chunks)):
            freq.update(f)
            for c in concepts:
                counts[c].update(part[c])
    return counts, freq


def cooccurrence_table(counts, freq, concepts, k):
    """``(C, vocab)``: counts over the ``k`` most frequent words that are not concepts, in frequency
    order. ``C[i, j]`` is how often concept ``i`` co-occurred with ``vocab[j]``. (n_concepts, k)."""
    skip = {c.lower() for c in concepts}
    vocab = [w for w in sorted(freq, key=freq.get, reverse=True) if w not in skip][:k]
    return np.array([[counts[c].get(w, 0) for w in vocab] for c in concepts], dtype=np.float64), vocab


def parquet_texts(paths, column="text", batch=4096):
    """The ``column`` strings of parquet files, streamed file by file in row batches."""
    import pyarrow.parquet as pq
    for path in paths:
        for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=[column]):
            yield from b.column(0).to_pylist()


def hub_texts(repo, prefix, n_docs=None):
    """The texts of every parquet shard under ``prefix`` in a Hugging Face dataset repo (e.g.
    ``"HuggingFaceFW/fineweb"``, ``"sample/10BT/"``), each downloaded into the HF cache when reached."""
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
