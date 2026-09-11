"""Corpus statistics: windowed co-occurrence counts from lowercased text, and the infini-gram client
for exact n-gram counts."""
import numpy as np
import pytest

from src.code.corpus import at_least, cooccurrence_table, parquet_texts, query, window_counts


def test_window_counts_lowercase_the_text_and_count_every_token_in_the_window():
    """Both "May" and "may" are occurrences of the concept "may"; numbers are tokens like words;
    punctuation only separates. Frequencies are counted over the whole text."""
    texts = ["In May the garden may bloom.", "On March 5, 2010 it snowed."]
    counts, freq = window_counts(texts, ["may", "march"], window=2)
    assert counts["may"] == {"in": 1, "the": 2, "garden": 2, "bloom": 1}
    assert counts["march"] == {"on": 1, "5": 1, "2010": 1}
    assert freq["may"] == 2 and freq["2010"] == 1


def test_cooccurrence_table_uses_the_most_frequent_tokens_concepts_included():
    counts = {"a": {"x": 3, "y": 1, "b": 2}, "b": {"y": 2, "a": 2}}
    freq = {"x": 10, "y": 20, "z": 30, "a": 100, "b": 5}
    C, vocab = cooccurrence_table(counts, freq, ["a", "b"], k=3)
    assert vocab == ["a", "z", "y"] and np.array_equal(C, [[0, 0, 1], [2, 0, 2]])


def test_at_least_filters_short_texts_by_word_count():
    assert list(at_least(["one two three", "four five", "six seven eight nine"], 3)) == ["one two three", "six seven eight nine"]


def test_parquet_texts_streams_the_text_column_across_files(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    for i, rows in enumerate((["a", "b", "c"], ["d"])):
        pq.write_table(pa.table({"text": rows, "id": list(range(len(rows)))}), tmp_path / f"{i}.parquet")
    assert list(parquet_texts([tmp_path / "0.parquet", tmp_path / "1.parquet"], batch=2)) == ["a", "b", "c", "d"]


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
