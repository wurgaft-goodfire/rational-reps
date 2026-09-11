"""Reading a pretrained LM on graph walks: per-node mean residuals and the walk's Bayes floor."""
import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig

from src.code.data import ring
from src.code.llm import bayes_floor, node_means, node_tokens
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


def test_bayes_floor_of_a_ring_walk():
    """Every ring node has two neighbours: log 2 per step, and log 1 = 0 once the step back is
    forbidden (the first step still has two options)."""
    walks = torch.tensor([[0, 1, 2, 3], [5, 4, 3, 2]])
    assert np.allclose(bayes_floor(ring(6), walks, backtrack=True), np.log(2))
    assert np.allclose(bayes_floor(ring(6), walks, backtrack=False), [np.log(2), 0.0, 0.0])


@pytest.mark.slow
def test_node_tokens_are_distinct_single_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    ids, words = node_tokens(tok, 25, seed=0)
    assert len(set(ids)) == 25 and len(words) == 25
    assert all(tok.encode(" " + w, add_special_tokens=False) == [i] for i, w in zip(ids, words))
    assert len(tok.encode(",", add_special_tokens=False)) == 1
