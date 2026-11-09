import numpy as np

from src.code.treebank import parse_conllu, tree_distances

CONLLU = """# sent_id = a
# text = The cat chases the mouse.
1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\t_
2\tcat\tcat\tNOUN\t_\t_\t3\tnsubj\t_\t_
3\tchases\tchase\tVERB\t_\t_\t0\troot\t_\t_
4\tthe\tthe\tDET\t_\t_\t5\tdet\t_\t_
5\tmouse\tmouse\tNOUN\t_\t_\t3\tobj\t_\t_
6\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_

# sent_id = b
# text = Mail me at a@b.com
1-2\tgonna\t_\t_\t_\t_\t_\t_\t_\t_
1\tMail\tmail\tVERB\t_\t_\t0\troot\t_\t_
2\tme\tI\tPRON\t_\t_\t1\tobj\t_\t_
3\tat\tat\tADP\t_\t_\t4\tcase\t_\t_
4\ta@b.com\ta@b.com\tX\t_\t_\t1\tobl:tmod\t_\t_

"""


def test_parse_keeps_words_heads_and_base_relations_and_drops_web_sentences(tmp_path):
    p = tmp_path / "x.conllu"; p.write_text(CONLLU)
    sents = parse_conllu(p)
    assert len(sents) == 1                                              # the e-mail sentence is filtered out
    s = sents[0]
    assert s["words"] == ["The", "cat", "chases", "the", "mouse", "."]
    assert s["heads"] == [1, 2, -1, 4, 2, 2]                            # 0-based head index, -1 for the root
    assert s["rels"] == ["det", "nsubj", "root", "det", "obj", "punct"]


def test_tree_distances_count_edges():
    D = tree_distances([1, 2, -1, 4, 2, 2])
    assert D[0, 4] == 3 and D[0, 3] == 4 and D[2, 5] == 1 and D[0, 0] == 0 and np.array_equal(D, D.T)


def test_syntax_probe_recovers_a_planted_typed_tree_code():
    """Words coded as path sums of tied per-type steps from the root (a polar code) plus a per-edge part: with
    tied steps alone, same-type siblings and paths in a different order (a then b, b then a) coincide, in a random 8-dim
    subspace of 64 dims plus noise: the probe recovers the trees, the types and the directions."""
    from src.code.syntax_probe import evaluate, prototypes, train_probe
    rng = np.random.default_rng(0)
    steps, A = rng.normal(size=(3, 8)), rng.normal(size=(8, 64))
    def sample(k):
        sents, X = [], []
        for _ in range(k):
            m = int(rng.integers(5, 12)); heads, rels, used = [-1], ["root"], {0: set()}
            for j in range(1, m):                     # siblings take distinct types (else tied steps make them coincide)
                h = int(rng.choice([k for k in used if len(used[k]) < 3]))
                r = str(rng.choice(sorted(set("abc") - used[h]))); used[h].add(r); used[j] = set()
                heads.append(h); rels.append(r)
            code = np.zeros((m, 8))
            for j in range(1, m):
                code[j] = code[heads[j]] + steps["abc".index(rels[j])] + 0.5 * rng.normal(size=8)   # per-edge part keeps codes distinct
            sents.append({"words": ["w"] * m, "heads": heads, "rels": rels}); X.append(code @ A + 0.05 * rng.normal(size=(m, 64)))
        return sents, np.concatenate(X).astype(np.float32)
    types = {"a": 0, "b": 1, "c": 2}
    tr, Xtr = sample(400); te, Xte = sample(100)
    probe = train_probe(Xtr, tr, types, rank=8, epochs=40, lr=1e-2, batch_size=50, seed=0)
    r = evaluate(probe, prototypes(probe, Xtr, tr), Xte, te)
    exact = {"B": np.linalg.pinv(A).T, "mu": np.zeros(64, np.float32), "sd": np.ones(64, np.float32), "types": types}
    r0 = evaluate(exact, prototypes(exact, Xtr, tr), Xte, te)           # the planted code read exactly: the ceiling
    assert r["uuas"] > r0["uuas"] - 0.05 and r["type"] > 0.9 and r["direction"] > 0.9, (r, r0)


def test_parse_keeps_universal_pos(tmp_path):
    p = tmp_path / "x.conllu"; p.write_text(CONLLU)
    assert parse_conllu(p)[0]["upos"] == ["DET", "NOUN", "VERB", "DET", "NOUN", "PUNCT"]


def test_root_paths_are_relation_sequences_from_the_root():
    from src.code.treebank import root_paths
    s = {"heads": [1, 2, -1, 4, 2, 2], "rels": ["det", "nsubj", "root", "det", "obj", "punct"]}
    assert root_paths(s) == [("nsubj", "det"), ("nsubj",), (), ("obj", "det"), ("obj",), ("punct",)]


def test_context_kernel_counts_classes_at_relative_paths():
    """Rows: root paths; columns: (relative path, class). The subject's parent is the verb; the subject's
    det child is a determiner; a missing child counts as the class 'none'."""
    from src.code.treebank import context_counts
    s = {"heads": [1, 2, -1, 4, 2, 2], "rels": ["det", "nsubj", "root", "det", "obj", "punct"],
         "upos": ["DET", "NOUN", "VERB", "DET", "NOUN", "PUNCT"]}
    C, rows, cols = context_counts([s], child_types=["det"], classes=["DET", "NOUN", "VERB", "PUNCT"])
    r = rows.index(("nsubj",))
    assert C[r, cols.index(("up", "VERB"))] == 1
    assert C[r, cols.index((("down", "det"), "DET"))] == 1
    assert C[rows.index(("nsubj", "det")), cols.index((("down", "det"), "none"))] == 1


def test_typed_path_counts_follow_every_path_up_to_length():
    """From the subject 'cat': up its nsubj edge to the verb; down its det edge to 'The'; up then down obj to
    'mouse'. Steps are labelled by the relation crossed."""
    from src.code.treebank import typed_path_counts
    s = {"heads": [1, 2, -1, 4, 2, 2], "rels": ["det", "nsubj", "root", "det", "obj", "punct"],
         "upos": ["DET", "NOUN", "VERB", "DET", "NOUN", "PUNCT"]}
    counts = typed_path_counts([s], max_len=2)
    cat = counts[("nsubj",)]
    assert cat[(("u", "nsubj"),), "VERB"] == 1
    assert cat[(("d", "det"),), "DET"] == 1
    assert cat[(("u", "nsubj"), ("d", "obj")), "NOUN"] == 1
    assert (("u", "nsubj"), ("d", "nsubj")) not in {p for p, _ in cat}           # no stepping back down to itself
