"""Read a pretrained LM on in-context tasks: items become single-token words, a stream of steps
becomes a prompt, and each key's representation is its mean residual over recent appearances."""
import numpy as np
import torch
import torch.nn.functional as F

# Concrete, unordered nouns for the items (Park et al. 2025 style); filtered per tokenizer to
# those that are one token with a leading space.
NOUNS = """apple sand math river chair cloud bread stone piano tiger lemon window garden pencil
ocean castle silver forest candle mirror jacket rocket hammer violin planet basket bottle camera
carpet desert engine guitar island kitten ladder magnet needle pepper puzzle rabbit saddle shadow
spider sugar table ticket tunnel valley wagon whistle anchor bridge button cactus dragon feather
flower helmet jungle kettle lantern meadow mountain napkin parrot pillow ribbon salmon temple
turtle umbrella velvet walnut zebra coffee doctor horse house money music paper train water wheel
honey glass metal night fire snow tree book door fish king moon star salt wine gold iron milk rain
road rock ship wolf bear corn rice hat ball cat dog box""".split()


def node_tokens(tokenizer, n, seed=0):
    """``n`` item words drawn from NOUNS, each a single token with a leading space.
    Returns ``(token_ids, words)``."""
    single = [(ids[0], w) for w in NOUNS
              if len(ids := tokenizer.encode(" " + w, add_special_tokens=False)) == 1]
    if len(single) < n:
        raise ValueError(f"only {len(single)} single-token nouns for {n} items")
    order = np.random.default_rng(seed).permutation(len(single))[:n]
    return [single[i][0] for i in order], [single[i][1] for i in order]


@torch.no_grad()
def _read(model, ids, at, keys, n_keys, window, tokens, targets, batch_size):
    """Forward ``ids`` (B, T) in batches. ``at`` (S,): the position read at each step; ``keys``
    (B, S): the key each step's residual is filed under; the last ``window`` steps of every
    sequence are pooled into per-key sums. ``targets`` (B, S'): index into ``tokens`` of the token
    predicted at the first S' read positions, scored under the softmax over ``tokens``.
    Returns ``means`` (layers, n_keys, d) and ``nll`` (B, S')."""
    device = next(model.parameters()).device
    ids, keys, targets, tokens, at = (torch.as_tensor(x).to(device) for x in (ids, keys, targets, tokens, at))
    sums, counts, nll = 0.0, torch.zeros(n_keys, device=device), []
    for b in range(0, len(ids), batch_size):
        out = model(ids[b:b + batch_size], output_hidden_states=True)
        H = torch.stack(out["hidden_states"])[:, :, at]                          # (layers, b, S, d)
        onehot = F.one_hot(keys[b:b + batch_size, -window:], n_keys).float()     # (b, window, n_keys)
        sums = sums + torch.einsum("lbtd,btn->lnd", H[:, :, -window:].float(), onehot)
        counts += onehot.sum((0, 1))
        logp = torch.log_softmax(out["logits"][:, at[:targets.shape[1]]][:, :, tokens].float(), -1)
        nll.append(-logp.gather(-1, targets[b:b + batch_size, :, None]).squeeze(-1))
    return {"means": (sums / counts[:, None]).cpu().numpy(), "nll": torch.cat(nll).cpu().numpy()}


def node_means(model, walks, tokens, sep, window, prefix=(), batch_size=16):
    """Prompt ``prefix, word, sep, word, sep, ...``; a node's representation is the residual at the
    separator after it. ``means`` (layers, n, d): the mean over its appearances in the last
    ``window`` steps, pooled over walks. ``nll`` (L-1,): mean negative log-probability of the next
    word at that separator under the softmax over the node tokens, by step. ``walks`` (B, L) node
    ids, ``tokens`` (n,) token ids, ``prefix`` token ids prepended to every walk (e.g. BOS)."""
    walks = torch.as_tensor(walks).cpu(); B, L = walks.shape; k = len(prefix)
    ids = torch.full((B, k + 2 * L), sep, dtype=torch.long)
    ids[:, :k] = torch.tensor(prefix, dtype=torch.long); ids[:, k::2] = torch.as_tensor(tokens)[walks]
    out = _read(model, ids, k + 1 + 2 * torch.arange(L), walks, len(tokens), window, tokens, walks[:, 1:], batch_size)
    return {"means": out["means"], "nll": out["nll"].mean(0)}


def pair_means(model, triples, tokens, out_tokens, colon, sep, window, prefix=(), batch_size=16):
    """Prompt ``prefix, a, b, colon, c, sep, ...``; a pair's representation is the residual at its
    colon. ``means`` (layers, n*n, d): the mean over its appearances in the last ``window`` triples,
    pooled over sequences, keyed ``a*n + b``. ``nll`` (B, K): negative log-probability of ``c`` at
    that colon under the softmax over ``out_tokens``. ``seen`` (B, K): the pair already appeared
    earlier in its sequence. ``triples`` (B, K, 3) of ``(a, b, c)`` with ``c`` an outcome index."""
    triples = torch.as_tensor(triples).cpu(); B, K, _ = triples.shape; k, n = len(prefix), len(tokens)
    ids = torch.full((B, k + 5 * K), sep, dtype=torch.long)
    ids[:, :k] = torch.tensor(prefix, dtype=torch.long)
    ids[:, k::5] = torch.as_tensor(tokens)[triples[..., 0]]; ids[:, k + 1::5] = torch.as_tensor(tokens)[triples[..., 1]]
    ids[:, k + 2::5] = colon; ids[:, k + 3::5] = torch.as_tensor(out_tokens)[triples[..., 2]]
    keys = triples[..., 0] * n + triples[..., 1]
    out = _read(model, ids, k + 2 + 5 * torch.arange(K), keys, n * n, window, out_tokens, triples[..., 2], batch_size)
    earlier = torch.tril(torch.ones(K, K, dtype=torch.bool), -1)                  # [t, t']: t' < t
    seen = ((keys[:, :, None] == keys[:, None, :]) & earlier).any(-1)
    return {**out, "seen": seen.numpy()}


def triple_stream(dataset, B, K):
    """(B, K, 3) triples ``(a, b, c)`` from a relational dataset's pointwise rows (every pair
    enumerated, an outcome drawn per row), permuted into ``B`` streams; ``c`` indexes the outcome
    alphabet from zero. The dataset's batch must hold at least ``B * K`` rows."""
    rows = dataset[0]["input_ids"]
    if len(rows) < B * K:
        raise ValueError(f"{len(rows)} rows for {B} x {K} triples; raise training_kwargs.batch_size")
    perm = torch.randperm(len(rows), generator=dataset.rng, device=rows.device)[:B * K]
    rows = rows[perm].clone(); rows[:, 2] -= dataset.n_items * dataset.bipartite
    return rows.reshape(B, K, 3).cpu()


def row_entropy(P, keys):
    """Entropy of each key's outcome row of ``P`` (..., C), flattened over the key axes: the Bayes
    floor for predicting the next token from that key. ``keys`` (B, S) -> (B, S). A walk's key is
    the current node; a relation's is the pair ``a*n + b``."""
    rows = np.asarray(P, dtype=np.float64).reshape(-1, np.shape(P)[-1])[np.asarray(keys)]
    return -(rows * np.log(np.where(rows > 0, rows, 1.0))).sum(-1)
