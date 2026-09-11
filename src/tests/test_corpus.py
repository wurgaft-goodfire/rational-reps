"""Corpus statistics: windowed co-occurrence counts from text, and the infini-gram client for exact
n-gram counts."""
import numpy as np
import pytest

from src.code.corpus import cooccurrence_table, parquet_texts, query, window_counts


def test_window_counts_add_the_words_around_each_concept_occurrence():
    """Every word within ``window`` tokens of a concept occurrence is counted for that concept (the
    occurrence itself excluded); word frequencies are counted over the whole text. A capitalised
    concept ("May") matches only its own case, a lowercase one matches either case."""
    texts = ["In May the garden may bloom.", "Red roses and red wine.", "Nothing here."]
    counts, freq = window_counts(texts, ["May", "red"], window=2)
    assert counts["May"] == {"in": 1, "the": 1, "garden": 1}          # "may" (modal) is not an occurrence
    assert counts["red"] == {"roses": 2, "and": 2, "wine": 1}         # "Red" and "red" both occur, three apart
    assert freq["may"] == 2 and freq["nothing"] == 1


def test_window_counts_skip_sentence_initial_capitalised_concepts_and_stop_at_no_punctuation():
    """A capitalised concept at a sentence start is mostly its other sense ("May" the modal), so it
    is not an occurrence; punctuation is not a context word but the window runs across it."""
    counts, _ = window_counts(["Rain. May showers came in May. Cold!"], ["May"], window=2)
    assert counts["May"] == {"came": 1, "in": 1, "cold": 1}


def test_parquet_texts_streams_the_text_column_across_files(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    for i, rows in enumerate((["a", "b", "c"], ["d"])):
        pq.write_table(pa.table({"text": rows, "id": list(range(len(rows)))}), tmp_path / f"{i}.parquet")
    assert list(parquet_texts([tmp_path / "0.parquet", tmp_path / "1.parquet"], batch=2)) == ["a", "b", "c", "d"]


def test_cooccurrence_table_orders_the_top_vocabulary_by_frequency_without_the_concepts():
    counts = {"a": {"x": 3, "y": 1}, "b": {"y": 2}}
    freq = {"x": 10, "y": 20, "z": 30, "a": 100, "b": 5}
    C, vocab = cooccurrence_table(counts, freq, ["a", "b"], k=2)
    assert vocab == ["z", "y"] and np.array_equal(C, [[0, 1], [0, 2]])


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
