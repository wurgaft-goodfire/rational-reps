"""Corpus statistics: windowed co-occurrence counts from lowercased text, and the infini-gram client
for exact n-gram counts."""
import numpy as np
import pytest

from src.code.corpus import (at_least, cooccurrence_matrix, encode_corpus, joined, parquet_texts, query, vocabulary,
                             window_counts, word_tokenizer)


def test_window_counts_lowercase_the_text_and_count_every_token_in_the_window():
    """Both "May" and "may" are occurrences of the concept "may"; numbers are tokens like words;
    punctuation only separates. Frequencies are counted over the whole text."""
    texts = ["In May the garden may bloom.", "On March 5, 2010 it snowed."]
    counts, freq = window_counts(texts, ["may", "march"], window=2)
    assert counts["may"] == {"in": 1, "the": 2, "garden": 2, "bloom": 1}
    assert counts["march"] == {"on": 1, "5": 1, "2010": 1}
    assert freq["may"] == 2 and freq["2010"] == 1


def test_at_least_filters_short_texts_by_word_count():
    assert list(at_least(["one two three", "four five", "six seven eight nine"], 3)) == ["one two three", "six seven eight nine"]


def test_joined_concatenates_consecutive_lines_into_passages():
    assert list(joined(["a", "b", "c", "d", "e"], 2)) == ["a b", "c d", "e"]


def test_parquet_texts_streams_the_text_column_across_files(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    for i, rows in enumerate((["a", "b", "c"], ["d"])):
        pq.write_table(pa.table({"text": rows, "id": list(range(len(rows)))}), tmp_path / f"{i}.parquet")
    assert list(parquet_texts([tmp_path / "0.parquet", tmp_path / "1.parquet"], batch=2)) == ["a", "b", "c", "d"]


TEXTS = ["The cat sat on the mat. The cat sat, 2 cats sat!", "A dog sat; the dog ran to the cat."]   # 11 distinct tokens
SHARDS = [TEXTS[:1], TEXTS[1:]]                                          # two shards, read with ``list``


def test_vocabulary_keeps_the_most_frequent_lowercased_tokens_in_frequency_order():
    """Text is lowercased and split on anything that is not a letter or digit; the ``size`` most
    frequent tokens are kept, counted exactly over all shards."""
    words, counts = vocabulary(SHARDS, list, 3, workers=2)
    assert words.tolist() == ["the", "sat", "cat"] and counts.tolist() == [5, 4, 3]


def test_encode_corpus_streams_in_vocabulary_ids_per_document():
    """Ids index the vocabulary; out-of-vocabulary tokens are dropped; ``lengths`` counts the kept
    tokens of each text, in shard order."""
    words, _ = vocabulary(SHARDS, list, 3, workers=2)
    ids, lengths = encode_corpus(SHARDS, list, word_tokenizer(words), workers=2)
    assert ids.dtype == np.uint16 and ids.tolist() == [0, 2, 1, 0, 0, 2, 1, 1, 1, 0, 0, 2]
    assert lengths.tolist() == [8, 4]


def test_cooccurrence_matrix_matches_the_window_counter_and_weights_by_distance():
    """Flat weighting reproduces ``window_counts`` over the same tokens (both orders, same document
    only); ``1/d`` divides each pair's count by its distance."""
    words, _ = vocabulary(SHARDS, list, 11, workers=2)
    ids, lengths = encode_corpus(SHARDS, list, word_tokenizer(words), workers=2)
    docs = np.repeat(np.arange(len(lengths)), lengths)
    counts, _ = window_counts(TEXTS, list(words), window=3)
    expected = np.array([[counts[a].get(b, 0) for b in words] for a in words])
    C = cooccurrence_matrix(ids, docs, len(words), window=3, weighting="flat", device="cpu")
    assert np.array_equal(C, expected)
    C1 = cooccurrence_matrix(ids, docs, len(words), window=3, weighting="1/d", device="cpu")
    i, j = list(words).index("cat"), list(words).index("mat")             # "mat the cat" once, distance 2; nothing else within 3
    assert np.isclose(C[i, j], 1) and np.isclose(C1[i, j], 0.5)


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
