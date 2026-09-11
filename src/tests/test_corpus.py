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


@pytest.mark.slow
def test_query_reaches_the_api():
    r = query({"index": "v4_dolma-v1_7_llama", "query_type": "count", "query": " January"})
    assert r["count"] > 1_000_000
