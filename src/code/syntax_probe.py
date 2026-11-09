"""The syntactic Polar Probe (Diego-Simon et al., NeurIPS 2024): one linear map B, trained so that squared probe
distances match tree distances (Hewitt & Manning's L1 structural loss) and the probed edge vectors
s = B(h_head - h_dependent) of the same relation type are collinear and of different types orthogonal
(angular loss on their cosines). Types and directions are read from per-type prototypes, the mean probed edge."""
import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree

from .treebank import tree_distances


def sentence_index(sentences):
    """Start offset of each sentence in the concatenated word array."""
    return np.cumsum([0] + [len(s["words"]) for s in sentences])[:-1]


def _batch(X, sentences, dists, starts, idx, types, device):
    """Padded tensors for the sentences ``idx``: H (b, n, d), D (b, n, n), word mask (b, n), and the edges as
    (sentence, dependent, head, type) index tensors (root excluded)."""
    n = max(len(sentences[i]["words"]) for i in idx)
    H = np.zeros((len(idx), n, X.shape[1]), dtype=np.float32); D = np.zeros((len(idx), n, n), dtype=np.float32)
    mask = np.zeros((len(idx), n), dtype=bool); edges = []
    for b, i in enumerate(idx):
        s = sentences[i]; m = len(s["words"])
        H[b, :m] = X[starts[i]:starts[i] + m]; D[b, :m, :m] = dists[i]; mask[b, :m] = True
        edges += [(b, j, h, types[r]) for j, (h, r) in enumerate(zip(s["heads"], s["rels"])) if h >= 0 and r in types]
    e = torch.tensor(edges, device=device).T
    return torch.tensor(H, device=device), torch.tensor(D, device=device), torch.tensor(mask, device=device), e


def train_probe(X, sentences, types, rank=128, epochs=30, lr=5e-3, lam=10.0, batch_size=200, max_edges=2000, seed=0, device=None):
    """``X`` (total words, d) one layer's word codes for ``sentences``; ``types`` {relation: index}. Features are used raw,
    as in Hewitt & Manning's implementation. Returns {"B", "mu", "sd", "types"} (numpy B)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(seed)
    mu, sd = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)         # raw features, as Hewitt & Manning
    Xs = ((X - mu) / sd).astype(np.float32); starts = sentence_index(sentences)
    dists = [tree_distances(s["heads"]) for s in sentences]
    B = (torch.rand(rank, X.shape[1], generator=g) * 0.1 - 0.05).to(device).requires_grad_()   # Hewitt & Manning: uniform(-0.05, 0.05)
    opt = torch.optim.Adam([B], lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(len(sentences), generator=g).split(batch_size):
            H, D, mask, e = _batch(Xs, sentences, dists, starts, idx.tolist(), types, device)
            Z = H @ B.T
            sq = ((Z[:, :, None] - Z[:, None, :]) ** 2).sum(-1)
            pair = mask[:, :, None] & mask[:, None, :]
            n = mask.sum(1).float()
            structural = (((sq - D).abs() * pair).sum((1, 2)) / n ** 2).mean()   # Hewitt & Manning: per sentence, / n^2
            s = Z[e[0], e[2]] - Z[e[0], e[1]]                                      # head minus dependent
            if len(s) > max_edges:
                keep = torch.randperm(len(s), generator=g)[:max_edges].to(device); s, t = s[keep], e[3][keep]
            else:
                t = e[3]
            u = s / (s.norm(dim=-1, keepdim=True) + 1e-12)
            angular = ((u @ u.T - (t[:, None] == t[None, :]).float()) ** 2).mean()
            opt.zero_grad(); (structural + lam * angular).backward(); opt.step()
    return {"B": B.detach().cpu().numpy(), "mu": mu, "sd": sd, "types": types}


def _probed(probe, X):
    return ((X - probe["mu"]) / probe["sd"]) @ probe["B"].T


def prototypes(probe, X, sentences):
    """Mean unit probed edge (head minus dependent) per type over ``sentences``: (n_types, rank)."""
    Z = _probed(probe, X); starts = sentence_index(sentences); T = probe["types"]
    sums, counts = np.zeros((len(T), Z.shape[1])), np.zeros(len(T))
    for i, s in enumerate(sentences):
        for j, (h, r) in enumerate(zip(s["heads"], s["rels"])):
            if h >= 0 and r in T:
                v = Z[starts[i] + h] - Z[starts[i] + j]
                sums[T[r]] += v / (np.linalg.norm(v) + 1e-12); counts[T[r]] += 1
    P = sums / np.maximum(counts, 1)[:, None]
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)


def evaluate(probe, P, X, sentences, skip=("punct",)):
    """UUAS (gold undirected edges recovered by the minimum spanning tree of squared probe distances), type
    accuracy and direction accuracy on gold edges (nearest prototype by |cosine|, head by its sign), and LAS
    (gold edges in the tree with type and head right). Edges of relations in ``skip`` are not scored."""
    Z = _probed(probe, X); starts = sentence_index(sentences); T = probe["types"]
    hit = typ = dirn = las = total = 0
    for i, s in enumerate(sentences):
        m = len(s["words"]); z = Z[starts[i]:starts[i] + m]
        if m < 2:
            continue
        sq = ((z[:, None] - z[None]) ** 2).sum(-1) + 1e-9
        mst = minimum_spanning_tree(np.triu(sq, 1)).toarray() > 0
        mst = mst | mst.T
        for j, (h, r) in enumerate(zip(s["heads"], s["rels"])):
            if h < 0 or r in skip or r not in T:
                continue
            v = z[h] - z[j]; c = P @ (v / (np.linalg.norm(v) + 1e-12))
            k = int(np.argmax(np.abs(c))); right_type = k == T[r]; right_dir = c[k] > 0
            total += 1; hit += mst[j, h]; typ += right_type; dirn += right_dir; las += mst[j, h] and right_type and right_dir
    return {"uuas": hit / total, "type": typ / total, "direction": dirn / total, "las": las / total}
