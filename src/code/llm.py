"""Read a pretrained LM on graph walks: nodes become single-token words, the walk becomes a
prompt, and each node's representation is its mean residual over recent appearances."""
import numpy as np
import torch
import torch.nn.functional as F

# Concrete, unordered nouns for the graph nodes (Park et al. 2025 style); filtered per tokenizer
# to those that are one token with a leading space.
NOUNS = """apple sand math river chair cloud bread stone piano tiger lemon window garden pencil
ocean castle silver forest candle mirror jacket rocket hammer violin planet basket bottle camera
carpet desert engine guitar island kitten ladder magnet needle pepper puzzle rabbit saddle shadow
spider sugar table ticket tunnel valley wagon whistle anchor bridge button cactus dragon feather
flower helmet jungle kettle lantern meadow mountain napkin parrot pillow ribbon salmon temple
turtle umbrella velvet walnut zebra coffee doctor horse house money music paper train water wheel
honey glass metal night fire snow tree book door fish king moon star salt wine gold iron milk rain
road rock ship wolf bear corn rice hat ball cat dog box""".split()


def node_tokens(tokenizer, n, seed=0):
    """``n`` node words drawn from NOUNS, each a single token with a leading space.
    Returns ``(token_ids, words)``."""
    single = [(ids[0], w) for w in NOUNS
              if len(ids := tokenizer.encode(" " + w, add_special_tokens=False)) == 1]
    if len(single) < n:
        raise ValueError(f"only {len(single)} single-token nouns for {n} nodes")
    order = np.random.default_rng(seed).permutation(len(single))[:n]
    return [single[i][0] for i in order], [single[i][1] for i in order]


@torch.no_grad()
def node_means(model, walks, tokens, sep, window, prefix=(), batch_size=16):
    """Prompt ``prefix, word, sep, word, sep, ...``; a node's representation is the residual at the
    separator after it. ``means`` (layers, n, d): the mean over its appearances in the last
    ``window`` steps, pooled over walks. ``nll`` (L-1,): mean negative log-probability of the next
    word at that separator under the softmax over the node tokens, by step. ``walks`` (B, L) node
    ids, ``tokens`` (n,) token ids, ``prefix`` token ids prepended to every walk (e.g. BOS)."""
    device = next(model.parameters()).device
    walks = torch.as_tensor(walks, device=device); tokens = torch.as_tensor(tokens, device=device)
    B, L = walks.shape; n, k = len(tokens), len(prefix)
    prefix = torch.tensor(prefix, dtype=torch.long, device=device)
    sums, counts, nll = 0.0, torch.zeros(n, device=device), torch.zeros(L - 1, device=device)
    for b in range(0, B, batch_size):
        w = walks[b:b + batch_size]
        ids = torch.full((len(w), k + 2 * L), sep, dtype=torch.long, device=device)
        ids[:, :k] = prefix; ids[:, k::2] = tokens[w]
        out = model(ids, output_hidden_states=True)
        at_sep = torch.stack(out["hidden_states"])[:, :, k + 1::2]                  # (layers, b, L, d)
        onehot = F.one_hot(w[:, -window:], n).float()                               # (b, window, n)
        sums = sums + torch.einsum("lbtd,btn->lnd", at_sep[:, :, -window:].float(), onehot)
        counts += onehot.sum((0, 1))
        logp = torch.log_softmax(out["logits"][:, k + 1::2][:, :-1][:, :, tokens].float(), -1)  # (b, L-1, n)
        nll += -logp.gather(-1, w[:, 1:, None]).squeeze(-1).sum(0)
    return {"means": (sums / counts[:, None]).cpu().numpy(), "nll": (nll / B).cpu().numpy()}


def bayes_floor(A, walks, backtrack=True):
    """Entropy of the walk's exact next-node distribution by position, (L-1,): ``log deg`` of the
    current node, minus one option once the step back is forbidden."""
    deg = np.asarray(A).sum(1)[np.asarray(walks)[:, :-1]]                          # (B, L-1)
    allowed = deg - (np.arange(deg.shape[1]) >= 1) * (not backtrack)
    return np.log(allowed).mean(0)
