"""Corpus statistics through the infini-gram API (Liu et al. 2024): windowed co-occurrence counts
between concept words and a context vocabulary over trillion-token indexes, cached per query."""
import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

API = "https://api.infini-gram.io/"


def query(payload, timeout=60, attempts=3):
    """POST one infini-gram query and return its response dict. Network failures are retried a few
    times, then raised; an error reported by the API is raised at once."""
    data = json.dumps(payload).encode()
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == attempts - 1:
                raise
            continue
        if "error" in out:
            raise RuntimeError(f"infini-gram rejected {payload['query']!r}: {out['error']}")
        return out


def _cached(cache_dir, index, payload, query_fn):
    """The response for ``payload``, read from ``cache_dir/index/<hash>.json`` or queried and written."""
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / index / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = query_fn(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def cooccurrence(concepts, context, index, window=16, max_clause_freq=500_000, cache_dir="outputs/corpus",
                 query_fn=query, workers=4):
    """Windowed co-occurrence counts ``C[i, j]``: how often concept ``i`` and context word ``j`` fall
    within ``window`` tokens of each other in the index. One CNF count query per pair; counts are
    exact for words below ``max_clause_freq`` occurrences and sampled above. (n_concepts, n_context)."""
    payloads = [{"index": index, "query_type": "count", "query": f" {c} AND {w}",
                 "max_diff_tokens": window, "max_clause_freq": max_clause_freq}
                for c in concepts for w in context]
    with ThreadPoolExecutor(workers) as pool:
        counts = list(pool.map(lambda p: _cached(cache_dir, index, p, query_fn)["count"], payloads))
    return np.array(counts, dtype=np.float64).reshape(len(concepts), len(context))


def context_vocabulary(tokenizer, k, exclude=()):
    """The first ``k`` vocabulary entries, in id order (BPE merge order stands in for frequency), that
    decode to a lowercase alphabetic word with a leading space, minus ``exclude``."""
    out, skip = [], set(exclude)
    for i in range(len(tokenizer)):
        s = tokenizer.decode([i])
        if s.startswith(" ") and s[1:].isalpha() and s[1:].islower() and s[1:] not in skip:
            out.append(s[1:])
            if len(out) == k:
                break
    return out
