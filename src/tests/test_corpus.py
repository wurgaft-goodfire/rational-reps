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


TREE = [["organism"], ["plant", "animal"], ["flower", "tree", "bird", "fish"],
        ["rose", "tulip", "oak", "pine", "robin", "crow", "trout", "shark"]]        # levels of a perfect binary tree


def test_tree_splits_are_the_haar_vectors_of_the_tree_coarse_to_fine():
    """One vector per internal node: +1 on the nodes under its first child, -1 under its second, 0 elsewhere,
    ordered root first; they are orthogonal and there are 2^depth - 1 of them over 2^(depth+1) - 1 nodes."""
    from src.code.corpus import tree_splits
    S = tree_splits(TREE)
    nodes = [w for level in TREE for w in level]
    assert S.shape == (7, 15) and np.allclose(S @ S.T, np.diag((S ** 2).sum(1)))
    root = dict(zip(nodes, S[0]))
    assert root["organism"] == 0 and root["plant"] == root["rose"] == 1 and root["animal"] == root["shark"] == -1


def test_coarse_to_fine_and_orthogonality_are_perfect_for_a_code_built_from_the_splits():
    """A code whose columns are the tree's split vectors with decreasing scale has top-k PCs spanning the
    k coarsest splits (alignment 1 at k = 1, 3, 7) and every child's offset from its parent orthogonal to
    the parent (mean |cos| 0); a shuffled code does neither."""
    from src.code.corpus import coarse_to_fine, orthogonality, tree_splits
    S = tree_splits(TREE)
    E = (S * (2.0 ** -np.arange(7))[:, None]).T                          # (15 nodes, 7), coarse splits largest
    assert np.allclose([coarse_to_fine(E, TREE, k) for k in (1, 3, 7)], 1)
    assert abs(orthogonality(E, TREE)) < 1e-9
    rng = np.random.default_rng(0)
    Es = E[np.r_[0, 1 + rng.permutation(14)]]                         # root fixed: its vector is 0 in this code
    assert coarse_to_fine(Es, TREE, 3) < 0.9 and orthogonality(Es, TREE) > 0.1


KIDS = {"r": ["a", "b", "c"], "a": ["a1", "a2", "a3"], "b": ["b1", "b2"], "c": ["c1", "c2"]}   # 3+3+1 = 7 depth-2 trees


def test_binary_subtrees_enumerates_every_distinct_full_tree():
    """Depth-2 trees under a toy children map: each internal node has two distinct children of its own,
    children in sorted order, and asking for more trees than exist returns exactly the 7 structures."""
    from src.code.corpus import binary_subtrees
    trees = binary_subtrees(KIDS.get, "r", depth=2, n=20, seed=0)
    assert len(trees) == 7 and len({tuple(map(tuple, t)) for t in trees}) == 7
    for t in trees:
        assert [len(l) for l in t] == [1, 2, 4]
        for level, nxt in zip(t[:-1], t[1:]):
            for i, parent in enumerate(level):
                pair = nxt[2 * i:2 * i + 2]
                assert set(pair) <= set(KIDS[parent]) and pair[0] < pair[1]


def test_binary_subtrees_samples_uniformly_over_structures_not_child_pairs():
    """The pair {b, c} heads 1 of the 7 depth-2 trees, so uniform sampling draws it 1/7 of the time (100
    of 700 seeds); drawing child pairs uniformly would give 1/3 (233)."""
    from src.code.corpus import binary_subtrees
    hits = sum(binary_subtrees(KIDS.get, "r", depth=2, n=1, seed=s)[0][1] == ["b", "c"] for s in range(700))
    assert 70 <= hits <= 130, hits


def test_wordnet_hierarchy_names_synsets_by_vocabulary_words_and_contracts_unnamed_nodes():
    """A word names its most-counted noun sense when that sense holds most of the word's WordNet counts
    ("young", mostly an adjective, names nothing); a synset takes its first eligible word in vocabulary
    order (person before someone); a synset's parent is its deepest hypernym (person under organism, not
    causal agent) contracted to the nearest named ancestor (bird under animal, skipping chordate and
    vertebrate; owl under bird, skipping bird of prey)."""
    from nltk.corpus import wordnet as wn
    from src.code.corpus import wordnet_hierarchy
    children, word = wordnet_hierarchy(["organism", "person", "someone", "animal", "young", "bird", "owl"])
    assert sorted(word.values()) == ["animal", "bird", "organism", "owl", "person"]
    assert word[wn.synset("bird.n.01")] == "bird" and wn.synset("bird.n.02") not in word
    assert {wn.synset("animal.n.01"), wn.synset("person.n.01")} <= set(children[wn.synset("organism.n.01")])
    assert children[wn.synset("animal.n.01")] == [wn.synset("bird.n.01")]
    assert children[wn.synset("bird.n.01")] == [wn.synset("owl.n.01")]


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


def test_read_analogies_groups_questions_by_family_keeping_the_case(tmp_path):
    from src.code.corpus import read_analogies
    p = tmp_path / "q.txt"
    p.write_text(": capital-common-countries\nAthens Greece Baghdad Iraq\n: gram8-plural\nbanana bananas bird birds\ncar cars cat cats\n")
    q = read_analogies(p)
    assert q == {"capital-common-countries": [("Athens", "Greece", "Baghdad", "Iraq")],
                 "gram8-plural": [("banana", "bananas", "bird", "birds"), ("car", "cars", "cat", "cats")]}


def _cube_analogies():
    """16 words = every sign pattern of 4 attributes; a:b::c:d flips the same attribute in both pairs."""
    import itertools
    words = ["".join("+-"[b] for b in bits) for bits in itertools.product((0, 1), repeat=4)]
    E = np.array([[1.0 if ch == "+" else -1.0 for ch in w] for w in words])
    flip = lambda w, k: w[:k] + "+-"[w[k] == "+"] + w[k + 1:]
    questions = [(a, flip(a, k), c, flip(c, k)) for k in range(4) for a in words for c in words if a != c and a[k] == c[k]]
    return words, E, questions


def test_analogy_accuracy_is_exact_on_the_attribute_cube_and_chance_when_rows_are_shuffled():
    """Cosine 3CosAdd (nearest word to b - a + c, excluding a, b, c) recovers every parallelogram of a +-1
    attribute code; shuffling the code's rows leaves accuracy near chance (1 of 13 candidates)."""
    from src.code.corpus import analogy_accuracy
    words, E, questions = _cube_analogies()
    index = {w: i for i, w in enumerate(words)}
    assert analogy_accuracy(E, index, questions) == 1.0
    shuffled = analogy_accuracy(E[np.random.default_rng(0).permutation(len(words))], index, questions)
    assert shuffled < 0.2, shuffled


def test_analogy_accuracy_excludes_the_query_words_from_the_candidates():
    """With a duplicated row (E_b = E_a) the target b - a + c is c itself; c is excluded, so d is found only when
    it is the next nearest, and never when queries may answer."""
    from src.code.corpus import analogy_accuracy
    E = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.1, 1.0]])
    index = {"a": 0, "b": 1, "c": 2, "d": 3}
    assert analogy_accuracy(E, index, [("a", "b", "c", "d")]) == 1.0
    assert analogy_accuracy(E, index, [("a", "b", "c", "d")], exclude_queries=False) == 0.0
