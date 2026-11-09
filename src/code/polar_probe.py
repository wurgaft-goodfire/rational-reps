"""The Polar Probe (Diego-Simon et al. 2024, 2026): a linear map B to a low-rank space and one prototype per
relation type, trained so that probe distances rank like graph distances (structural loss, 1 - soft
Spearman) and the probed difference between linked entities points along its relation's prototype
(angular loss, squared error of the cosines against the incidence tensor)."""
import numpy as np
import torch
from scipy.stats import spearmanr


def _soft_rank(x, tau):
    """Differentiable ranks of the last axis: 1/2 + sum_j sigmoid((x_i - x_j) / tau)."""
    tau = torch.as_tensor(tau, device=x.device)
    tau = tau[..., None] if tau.ndim else tau                                  # a per-row scale broadcasts over pairs
    return 0.5 + torch.sigmoid((x[..., :, None] - x[..., None, :]) / tau).sum(-1)


def _pearson(a, b):
    a = a - a.mean(-1, keepdim=True); b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)


def _losses(B, P, H, D, M, tau):
    """Per-layer structural and angular losses, summed over layers. ``B`` (L, rank, d), ``P`` (L, T, rank),
    ``H`` (L, S, n, d); ``D`` (S, n, n) and ``M`` (S, n, n, T) are shared by the layers."""
    n = H.shape[2]; iu = torch.triu_indices(n, n, 1)
    Z = torch.einsum("lsnd,lrd->lsnr", H, B)
    delta = Z[:, :, None, :, :] - Z[:, :, :, None, :]                           # delta_ij = B(h_j - h_i): (L, S, n, n, rank)
    dist = delta.norm(dim=-1)[:, :, iu[0], iu[1]]                               # (L, S, pairs)
    gold = _soft_rank(D[:, iu[0], iu[1]], 1e-3)                                 # tiny tau: exact gold ranks, ties shared
    structural = (1 - _pearson(_soft_rank(dist, tau * dist.detach().std(-1, keepdim=True) + 1e-6), gold).mean(-1)).sum()
    cos = torch.einsum("lsijr,ltr->lsijt", delta / (delta.norm(dim=-1, keepdim=True) + 1e-12), P / P.norm(dim=-1, keepdim=True))
    linked = (M != 0).any(-1)                                                   # (S, n, n): both orientations of each edge
    angular = ((cos - M) ** 2).sum(-1)[:, linked].mean(-1).sum()
    return structural, angular


def train_probes(H, D, M, rank=512, epochs=100, lr=1e-5, lam=5.0, batch_size=32, tau=0.1, seed=0, device=None):
    """One probe per layer, trained together (independent parameters, so each trains exactly as alone).
    ``H`` (L, S, n, d), one description per sample; ``D`` (S, n, n); ``M`` (S, n, n, T). Features are
    standardised per layer over the training set. Returns [{"B", "P", "mu", "sd"}] (numpy), one per layer."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(seed)
    H = torch.as_tensor(np.asarray(H), dtype=torch.float32, device=device)
    mu, sd = H.mean((1, 2), keepdim=True), H.std((1, 2), keepdim=True) + 1e-6
    H = (H - mu) / sd
    D = torch.as_tensor(np.asarray(D), dtype=torch.float32, device=device)
    M = torch.as_tensor(np.asarray(M), dtype=torch.float32, device=device)
    L, d, T = H.shape[0], H.shape[-1], M.shape[-1]
    B = (torch.randn(L, rank, d, generator=g) / d ** 0.5).to(device).requires_grad_()
    P = torch.randn(L, T, rank, generator=g).to(device).requires_grad_()
    opt = torch.optim.Adam([B, P], lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(H.shape[1], generator=g).split(batch_size):
            s, a = _losses(B, P, H[:, idx.to(device)], D[idx.to(device)], M[idx.to(device)], tau)
            opt.zero_grad(); (s + lam * a).backward(); opt.step()
    B, P, mu, sd = (x.detach().cpu().numpy() for x in (B, P, mu, sd))
    return [{"B": B[l], "P": P[l], "mu": mu[l, 0, 0], "sd": sd[l, 0, 0]} for l in range(L)]


def train_probe(samples, n_types=None, **kw):
    """One layer: ``samples`` [(H (n, d), D (n, n), M (n, n, T))] -> {"B", "P", "mu", "sd"}."""
    H, D, M = (np.stack([s[k] for s in samples]) for k in range(3))
    return train_probes(H[None], D, M, **kw)[0]


def evaluate(probe, samples):
    """Mean Spearman over descriptions of (existence) probe distances vs graph distances over pairs, and
    (type) the probed incidence (cosine of delta_ij with each prototype) vs the gold incidence over i != j.
    The projection runs batched on the GPU when there is one."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = torch.as_tensor(np.stack([np.asarray(s[0], dtype=np.float32) for s in samples]), device=device)
    B, mu, sd = (torch.as_tensor(np.asarray(probe[k]), dtype=torch.float32, device=device) for k in ("B", "mu", "sd"))
    Z = ((H - mu) / sd) @ B.T                                                   # (S, n, rank)
    delta = Z[:, None, :, :] - Z[:, :, None, :]
    P = torch.as_tensor(probe["P"], dtype=torch.float32, device=device); P = P / P.norm(dim=-1, keepdim=True)
    dist = delta.norm(dim=-1).cpu().numpy()
    cos = (delta / (delta.norm(dim=-1, keepdim=True) + 1e-12) @ P.T).cpu().numpy()
    n = H.shape[1]; iu = np.triu_indices(n, 1); off = ~np.eye(n, dtype=bool)
    ex = [spearmanr(dist[s][iu], D[iu])[0] for s, (_, D, _) in enumerate(samples)]
    ty = [spearmanr(cos[s][off].ravel(), M[off].ravel())[0] for s, (_, _, M) in enumerate(samples)]
    return float(np.mean(ex)), float(np.mean(ty))
