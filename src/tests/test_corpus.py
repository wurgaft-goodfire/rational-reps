"""Corpus statistics through the infini-gram API: windowed co-occurrence tables and the context
vocabulary they are counted over."""
import json

import numpy as np
import pytest

from src.code.corpus import context_vocabulary, cooccurrence, query


class _Tok:
    """A tokenizer stub: ids decode to the listed strings."""
    def __init__(self, pieces):
        self.pieces = pieces

    def __len__(self):
        return len(self.pieces)

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def test_context_vocabulary_keeps_early_lowercase_words_with_a_leading_space():
    """The first ``k`` vocabulary entries that decode to a lowercase alphabetic word with a leading
    space, in id order (BPE merge order stands in for frequency); the concept words are excluded."""
    tok = _Tok(["<s>", " the", "ing", " Paris", " red", " and", " 42", " snow", "er", " cold", " hot"])
    assert context_vocabulary(tok, 3, exclude=["red"]) == ["the", "and", "snow"]


def test_cooccurrence_fills_the_table_from_one_query_per_pair_and_caches_it(tmp_path):
    """``C[i, j]`` is the windowed count of concept ``i`` with context word ``j``; every response is
    written to the cache, and a second call reads it back without querying."""
    calls = []

    def fake_query(payload):
        calls.append(payload["query"])
        a, b = payload["query"].split(" AND ")                      # " red", "car"
        return {"count": len(a.strip()) * len(b), "approx": False}

    C = cooccurrence(["red", "blue"], ["car", "sky"], index="idx", window=16, cache_dir=tmp_path, query_fn=fake_query)
    assert np.array_equal(C, [[3 * 3, 3 * 3], [4 * 3, 4 * 3]]) and len(calls) == 4
    assert " red AND car" in calls
    assert len(list((tmp_path / "idx").glob("*.json"))) == 4
    again = cooccurrence(["red", "blue"], ["car", "sky"], index="idx", window=16, cache_dir=tmp_path, query_fn=fake_query)
    assert np.array_equal(again, C) and len(calls) == 4


def test_query_backs_off_on_rate_limiting_then_succeeds(monkeypatch):
    """A 403/429 from the API is a burst limit: sleep with doubling waits and retry; the successful
    response comes back once the API answers."""
    import io, urllib.error
    from src.code import corpus
    replies = [urllib.error.HTTPError("u", 403, "Forbidden", {}, None),
               urllib.error.HTTPError("u", 429, "Too Many", {}, None),
               io.BytesIO(b'{"count": 7}')]
    sleeps = []
    monkeypatch.setattr(corpus.urllib.request, "urlopen", lambda req, timeout: (lambda r: (_ for _ in ()).throw(r) if isinstance(r, Exception) else r)(replies.pop(0)))
    monkeypatch.setattr(corpus.time, "sleep", sleeps.append)
    assert corpus.query({"query": "x"}, backoff=1.0)["count"] == 7
    assert sleeps == [1.0, 2.0]


@pytest.mark.slow
def test_query_reaches_the_api():
    r = query({"index": "v4_dolma-v1_7_llama", "query_type": "count", "query": " January"})
    assert r["count"] > 1_000_000
