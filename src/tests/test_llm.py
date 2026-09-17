"""Reading a pretrained LM on graph walks: per-node mean residuals and the walk's Bayes floor."""
import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig

from src.code.llm import node_means, node_tokens, pair_means, row_entropy, triple_stream, word_means
from src.code.models import TransformerModel


def _tiny():
    torch.manual_seed(0)
    return TransformerModel(GPTNeoXConfig(vocab_size=13, hidden_size=16, num_hidden_layers=2,
                                          num_attention_heads=2, intermediate_size=32,
                                          max_position_embeddings=32)).eval()


def test_node_means_and_nll_match_a_brute_force_pass():
    """The prompt is ``prefix, word, sep, word, sep, ...``. Per layer, each node's representation
    is the mean residual at the separator after it, over its appearances in the last ``window``
    steps pooled over sequences; ``nll`` scores the next word at that separator under the softmax
    over the node tokens."""
    model = _tiny()
    tokens = torch.tensor([2, 4, 6, 8, 10, 12]); B, L, window, sep = 5, 12, 5, 1
    walks = torch.randint(0, 6, (B, L), generator=torch.Generator().manual_seed(1))
    out = node_means(model, walks, tokens, sep=sep, window=window, prefix=(0,), batch_size=2)

    ids = torch.zeros(B, 1 + 2 * L, dtype=torch.long)
    ids[:, 1::2] = tokens[walks]; ids[:, 2::2] = sep
    with torch.no_grad():
        res = model(ids, output_hidden_states=True)
    H = torch.stack(res["hidden_states"])[:, :, 2::2]                      # (layers, B, L, d) at separators
    sums = torch.zeros(H.shape[0], 6, H.shape[-1]); counts = torch.zeros(6)
    for b in range(B):
        for t in range(L - window, L):
            sums[:, walks[b, t]] += H[:, b, t]; counts[walks[b, t]] += 1
    assert np.allclose(out["means"], (sums / counts[:, None]).numpy(), atol=1e-5)
    lp = torch.log_softmax(res["logits"][:, 2::2][:, :-1][:, :, tokens], -1)  # (B, L-1, n)
    nll = -lp.gather(-1, walks[:, 1:, None]).squeeze(-1).mean(0)
    assert np.allclose(out["nll"], nll.numpy(), atol=1e-5)


def test_pair_means_nll_and_seen_match_a_brute_force_pass():
    """Triples ``a b : c ,``: a pair's representation is the residual at its colon, averaged over
    its appearances in the last ``window`` triples pooled over sequences; ``nll`` (B, K) scores c
    at that colon over the outcome tokens; ``seen`` marks a pair that already appeared in its sequence."""
    model = _tiny()
    items, outs, colon, sep = torch.tensor([2, 4, 6, 8]), torch.tensor([3, 5, 7]), 1, 12
    B, K, window = 4, 6, 3
    g = torch.Generator().manual_seed(2)
    triples = torch.stack([torch.randint(0, 4, (B, K), generator=g), torch.randint(0, 4, (B, K), generator=g),
                           torch.randint(0, 3, (B, K), generator=g)], -1)
    out = pair_means(model, triples, items, outs, colon=colon, sep=sep, window=window, prefix=(0,), batch_size=3)

    ids = torch.full((B, 1 + 5 * K), sep, dtype=torch.long); ids[:, 0] = 0
    ids[:, 1::5] = items[triples[..., 0]]; ids[:, 2::5] = items[triples[..., 1]]
    ids[:, 3::5] = colon; ids[:, 4::5] = outs[triples[..., 2]]
    with torch.no_grad():
        res = model(ids, output_hidden_states=True)
    H = torch.stack(res["hidden_states"])[:, :, 3::5]                      # (layers, B, K, d) at the colons
    key = triples[..., 0] * 4 + triples[..., 1]
    sums = torch.zeros(H.shape[0], 16, H.shape[-1]); counts = torch.zeros(16)
    for b in range(B):
        for t in range(K - window, K):
            sums[:, key[b, t]] += H[:, b, t]; counts[key[b, t]] += 1
    assert np.allclose(np.nan_to_num(out["means"]), np.nan_to_num((sums / counts[:, None]).numpy()), atol=1e-5)
    lp = torch.log_softmax(res["logits"][:, 3::5][:, :, outs], -1)          # (B, K, 3)
    assert np.allclose(out["nll"], -lp.gather(-1, triples[..., 2:]).squeeze(-1).numpy(), atol=1e-5)
    seen = np.array([[bool((key[b, :t] == key[b, t]).any()) for t in range(K)] for b in range(B)])
    assert (out["seen"] == seen).all()


def _relational(relation, B, K, **kwargs):
    from omegaconf import OmegaConf
    from src.code.config import Config
    from src.code.data import RelationalDataset
    return RelationalDataset(Config(OmegaConf.create({
        "setting": "relational", "relation": relation, "relation_kwargs": kwargs, "lapse": 0.0,
        "random_seed": 0, "sample_mode": "pointwise", "training_kwargs": {"batch_size": B * K}})))


def test_triple_stream_draws_each_outcome_from_its_pair_row():
    """The stream is the dataset's pointwise rows (every pair enumerated, an outcome drawn per
    row), permuted and reshaped to (B, K, 3). On exact XOR every c equals a ^ b, and two full
    passes over the 64 pairs give each pair exactly twice."""
    triples = triple_stream(_relational("symmetric_difference", 4, 32, graded=False), 4, 32)
    a, b, c = triples.reshape(-1, 3).unbind(-1)
    assert triples.shape == (4, 32, 3) and (c == (a ^ b)).all()
    assert torch.bincount(a * 8 + b, minlength=64).eq(2).all()


def test_triple_stream_indexes_a_separate_outcome_alphabet_from_zero():
    """A bipartite relation's outcome tokens sit after the item tokens in the dataset; the stream
    returns the outcome index itself (here one of six (feature, yes/no) answers)."""
    triples = triple_stream(_relational("set_difference", 2, 32), 2, 32)
    assert triples[..., 2].min() >= 0 and triples[..., 2].max() < 6


def test_word_means_read_the_last_token_of_each_word_in_each_template():
    """A word's representation is the residual at its last token inside a template, averaged over
    templates; tokens are given as id lists, so a multi-token word is read at its final piece."""
    model = _tiny()
    templates = [([0, 1], [5]), ([2], [])]                     # (prefix ids, suffix ids) around the word
    words = {"a": [3], "bc": [4, 6]}
    out = word_means(model, words, templates)
    with torch.no_grad():
        for j, (w, ids) in enumerate(words.items()):
            reps = []
            for pre, post in templates:
                res = model(torch.tensor([pre + ids + post]), output_hidden_states=True)
                reps.append(torch.stack(res["hidden_states"])[:, 0, len(pre) + len(ids) - 1])
            assert np.allclose(out[:, j], torch.stack(reps).mean(0).numpy(), atol=1e-5)
    assert out.shape[1] == 2


def test_word_means_can_read_the_prompts_last_token_instead():
    """``at_end=True`` reads the residual at the final token of each template (after the suffix),
    the Bhalla et al. style probe, still averaged over templates."""
    model = _tiny()
    templates = [([0, 1], [5, 7]), ([2], [9])]
    words = {"a": [3], "bc": [4, 6]}
    out = word_means(model, words, templates, at_end=True)
    with torch.no_grad():
        for j, ids in enumerate(words.values()):
            reps = [torch.stack(model(torch.tensor([pre + ids + post]), output_hidden_states=True)["hidden_states"])[:, 0, -1]
                    for pre, post in templates]
            assert np.allclose(out[:, j], torch.stack(reps).mean(0).numpy(), atol=1e-5)


def test_row_entropy_indexes_the_flattened_kernel_by_key():
    """A walk's kernel (n, n) indexed by node, a relation's (n, n, C) by pair ``a*n + b``: the
    entropy of the row each key predicts from."""
    P = np.array([[0.5, 0.5], [1.0, 0.0]])
    assert np.allclose(row_entropy(P, torch.tensor([[0, 1, 0]])), [[np.log(2), 0.0, np.log(2)]])
    T = np.array([[[1.0, 0.0], [0.5, 0.5]], [[0.5, 0.5], [0.0, 1.0]]])   # pairs (0,0),(1,1) certain
    assert np.allclose(row_entropy(T, torch.tensor([[0, 1], [3, 2]])), [[0.0, np.log(2)], [0.0, np.log(2)]])


@pytest.mark.slow
def test_node_tokens_are_distinct_single_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    ids, words = node_tokens(tok, 25, seed=0)
    assert len(set(ids)) == 25 and len(words) == 25
    assert all(tok.encode(" " + w, add_special_tokens=False) == [i] for i, w in zip(ids, words))
    assert len(tok.encode(",", add_special_tokens=False)) == 1
