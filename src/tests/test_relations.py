import numpy as np
import pytest

from src.code.relations import DOMAINS, comparison_table, describe, next_table, sample_order, within_table


def test_chain_description_states_each_adjacent_fact_once_and_ends_with_the_entity_list():
    """A chain of 5 letters: 4 adjacent facts, each said once in either direction, then 'Who are ...?'
    listing every letter once in the fixed list order."""
    rng = np.random.default_rng(0)
    order = ["c", "a", "b", "d", "e"]                                   # c < a < b < d < e
    text = describe("chain", order, listing=["a", "b", "c", "d", "e"], rng=rng)
    lines = text.split("\n")
    assert lines[0] == DOMAINS["chain"]["intro"]
    facts = [f for f in lines[1].split(". ") if f]
    assert len(facts) == 4
    said = {frozenset(f.rstrip(".").split(" is ")[0:1] + [f.rstrip(".").split(" than ")[-1]]) for f in facts}
    assert said == {frozenset(p) for p in [("c", "a"), ("a", "b"), ("b", "d"), ("d", "e")]}
    for f in facts:                                                     # each fact is true
        f = f.rstrip(".")
        x, rest = f.split(" is ", 1)
        y = rest.split(" than ")[-1]
        assert (order.index(x) > order.index(y)) == rest.startswith("greater")
    assert lines[2] == "Who are a, b, c, d, e?"


def test_cycle_description_closes_the_loop():
    rng = np.random.default_rng(1)
    order = ["lake", "pub", "zoo", "park", "mall"]
    facts = [f for f in describe("cycle", order, listing=sorted(order), rng=rng).split("\n")[1].split(". ") if f]
    pairs = {(f.split(" after ")[1].split(" is ")[0], f.rstrip(".").split(" is ")[1]) for f in facts}
    assert pairs == {(order[i], order[(i + 1) % 5]) for i in range(5)}


def test_sample_order_draws_distinct_names_from_the_domain_pool():
    order = sample_order("cycle", 6, np.random.default_rng(2))
    assert len(set(order)) == 6 and set(order) <= set(DOMAINS["cycle"]["names"])


def test_tables_follow_positions():
    """Rows and columns are positions 0..n-1 along the structure."""
    C = comparison_table(4)
    assert C[2, 0] == 1 and C[0, 2] == -1 and C[1, 1] == 0               # is i greater than j?
    W = within_table(5, 2)
    assert W[4, 0] == 1 and W[4, 1] == 1 and W[4, 2] == -1 and W[0, 4] == -1   # is j within 2 stops after i?
    N = next_table(5, cyclic=True)
    assert N[4, 0] == 1 and N.sum() == 5 and next_table(5, cyclic=False)[4].sum() == 0


@pytest.mark.slow
def test_entity_positions_point_at_the_listed_entities_in_the_llama_tokenization():
    from transformers import AutoTokenizer
    from src.code.relations import encode
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    text = describe("chain", ["c", "a", "b"], listing=["a", "b", "c"], rng=np.random.default_rng(0))
    ids, at = encode(tok, text, ["a", "b", "c"])
    assert ids[0] == tok.bos_token_id
    assert [tok.decode([ids[i]]) for i in at] == [" a", " b", " c"]
    assert tok.decode(ids[1:]) == text                                  # the pieces reproduce the prompt
