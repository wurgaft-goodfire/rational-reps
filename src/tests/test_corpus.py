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


def test_log_conditional_is_the_skip_gram_log_probability_with_an_empty_cell_as_one_cooccurrence():
    """log P(j | i) = log(C_ij / r_i) with r_i the row's total weight: each row exponentiates to a
    distribution over the vocabulary; an empty cell is scored as one co-occurrence of weight 1, so
    nothing is -inf; a block keeps the full row total, not the block's."""
    from src.code.corpus import log_conditional
    C = np.random.default_rng(0).integers(2, 50, size=(4, 6)).astype(float)
    r = C.sum(1)
    L = log_conditional(C, r)
    assert np.allclose(np.exp(L).sum(1), 1) and np.allclose(L, np.log(C) - np.log(r)[:, None])
    C[0, 2] = 0
    L = log_conditional(C[:1], r[:1])
    assert np.isclose(L[0, 2], -np.log(r[0])) and np.isfinite(L).all()
    block = log_conditional(C[:2, :3], r[:2])                       # a block: same r, so exp sums to the block's share
    assert np.allclose(np.exp(block).sum(1), np.maximum(C[:2, :3], 1).sum(1) / r[:2])


def test_load_corpus_memory_maps_the_matrix_and_caches_its_row_sums(tmp_path):
    """``load_corpus`` returns the matrix as a memmap, the words, the counts and the row sums; the
    row sums are computed in chunks and written beside the matrix so later loads skip the pass."""
    import json
    from src.code.corpus import load_corpus
    C = np.random.default_rng(1).random((7, 7)).astype(np.float32); C = C + C.T
    np.save(tmp_path / "C_w16_1d.npy", C)
    (tmp_path / "vocab.json").write_text(json.dumps(list("abcdefg"))); np.save(tmp_path / "freq.npy", np.arange(7, 0, -1))
    Cm, vocab, freq, r = load_corpus(tmp_path, chunk=3)
    assert isinstance(Cm, np.memmap) and list(vocab) == list("abcdefg") and freq.tolist() == list(range(7, 0, -1))
    assert np.allclose(r, C.sum(1)) and np.allclose(np.load(tmp_path / "C_w16_1d_rows.npy"), r)


def test_encode_corpus_batches_by_characters_without_changing_the_stream():
    """Texts are encoded in batches of at most ``chars`` characters, a longer text alone, so a shard
    of whole books never holds thousands of encodings at once; the id stream and the per-text lengths
    are the same for any budget."""
    words, _ = vocabulary(SHARDS, list, 11, workers=2)
    tok = word_tokenizer(words)
    ids, lengths = encode_corpus(SHARDS, list, tok, workers=2)
    for chars in (1, 40, 10**9):                                   # every text alone; some pairs; everything at once
        ids_c, lengths_c = encode_corpus(SHARDS, list, tok, workers=2, chars=chars)
        assert np.array_equal(ids_c, ids) and np.array_equal(lengths_c, lengths)


def test_collapse_phrases_replaces_each_in_document_occurrence_with_one_new_id():
    """A phrase's tokens, consecutive and in the same document, become the single id V + p; a phrase
    straddling a document boundary is left alone; longer phrases win over their prefixes."""
    from src.code.corpus import collapse_phrases
    ids = np.array([3, 1, 2, 5, 1, 2, 9, 1, 2, 9, 7], dtype=np.uint16)
    docs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    ids2, docs2 = collapse_phrases(ids, docs, [[1, 2], [1, 2, 9]], V=10)
    assert ids2.tolist() == [3, 10, 5, 1, 2, 9, 11, 7] and docs2.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_concept_rows_match_the_window_counter_around_each_concept_occurrence():
    """Flat counts of every id within ``window`` positions of each occurrence of a concept, same document
    only, both sides; agrees with ``window_counts`` on the same tokens."""
    from src.code.corpus import concept_rows
    words, _ = vocabulary(SHARDS, list, 11, workers=2)
    ids, lengths = encode_corpus(SHARDS, list, word_tokenizer(words), workers=2)
    docs = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
    concepts = [list(words).index("cat"), list(words).index("dog")]
    R = concept_rows(ids, docs, concepts, n_ids=len(words), window=3)
    counts, _ = window_counts(TEXTS, ["cat", "dog"], window=3)
    expected = np.array([[counts[c].get(w, 0) for w in words] for c in ("cat", "dog")])
    assert R.shape == (2, len(words)) and np.array_equal(R, expected)
