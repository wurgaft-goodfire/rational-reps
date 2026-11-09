"""Uncertainty over weekdays in a pretrained LM. Prompts state a distribution over days in a
sentence (Spokes, Not a Ring's frequency adverbs; stated percentages; sets of equally likely days),
then an answer slot asks for the day. We read the residual at the sentence's final period, where
Spokes read, and at the answer slot, where the model predicts a day.

The prediction (tenets 1 and 4): at the answer slot the representation is a linear image of the
centred log-probabilities over the days, so a map fitted on certain prompts places every uncertain
one. Coordinates are the ring's harmonics: the seven day centroids' discrete Fourier vectors."""
import itertools

import numpy as np
import torch

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CONTS = ["her flight leaves", "his class meets", "the market opens", "their team practices",
         "the night bus runs", "her book club meets", "the shop gets deliveries", "his shift starts"]
ADVERBS = [("Always", "never", 1.0), ("Usually", "sometimes", .75), ("Sometimes", "sometimes", .5),
           ("Sometimes", "usually", .25), ("Never", "always", 0.0)]     # (A's, B's, weight on A)
PERCENTS = [100, 90, 75, 50, 25, 10, 0]                                 # percent on A
PAIRS = [(a, (a + s) % 7, s) for a in range(7) for s in (1, 2, 3)]
PAIRS += [(b, a, s) for a, b, s in PAIRS]                               # both naming orders: 42


def _two(A, B, wa):
    w = np.zeros(7); w[A] += wa; w[B] += 1 - wa
    return w


def _listing(days):
    names = [DAYS[d] for d in days]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", or " + names[-1]


def all_prompts():
    """[dict] with the sentence ``pre`` (ends at the period read as ``mark``), the answer slot
    ``post``, the stated ``weights`` (7,) and labels: family, level, A, B, sep (pairs), start, size
    (sets), cont (continuation index)."""
    out = []
    for c, cont in enumerate(CONTS):
        post, Cont = f" This week, {cont} on", cont[0].upper() + cont[1:]
        for (A, B, s), (l, (a, b, wa)) in itertools.product(PAIRS, enumerate(ADVERBS)):
            out.append(dict(family="adverb", level=l, A=A, B=B, sep=s, cont=c, weights=_two(A, B, wa),
                            pre=f"{a} on {DAYS[A]}, {b} on {DAYS[B]}, {cont}.", post=post))
        for (A, B, s), (l, pa) in itertools.product(PAIRS, enumerate(PERCENTS)):
            out.append(dict(family="percent", level=l, A=A, B=B, sep=s, cont=c, weights=_two(A, B, pa / 100),
                            pre=f"{Cont} on {DAYS[A]} with {pa} percent probability and on {DAYS[B]} "
                                f"with {100 - pa} percent probability.", post=post))
        for size, start in itertools.product(range(1, 8), range(7)):
            days = [(start + i) % 7 for i in range(size)]
            w = np.zeros(7); w[days] = 1 / size
            pre = f"{Cont} on {_listing(days)}" + (", each equally likely." if size > 1 else ".")
            out.append(dict(family="set", level=size - 1, size=size, start=start, cont=c, weights=w, pre=pre, post=post))
    return out


def day_token_ids(tok):
    """Per day, the ids of its single-token spellings (" Monday", "Monday", " monday")."""
    out = []
    for d in DAYS:
        ids = [tok.encode(w, add_special_tokens=False) for w in (" " + d, d, " " + d.lower(), d.lower())]
        out.append([i[0] for i in ids if len(i) == 1])
    return out


def day_distribution(logits, days):
    """Next-token distribution (N, V) logits -> (N, 8): each day's spellings summed, then 'other'."""
    p = torch.softmax(logits.float(), -1)
    q = torch.stack([p[:, ids].sum(-1) for ids in days], -1)
    return torch.cat([q, 1 - q.sum(-1, keepdim=True)], -1).cpu().numpy()


@torch.no_grad()
def read_prompts(model, tok, prompts, batch_size=32):
    """Residuals at every layer at the period (``mark``) and the last token (``answer``), each
    (layers, N, d) float16, and the day distributions there, ``q_mark`` and ``q_answer`` (N, 8)."""
    device, days = next(model.parameters()).device, day_token_ids(tok)
    seqs = [(tok(p["pre"]).input_ids, tok(p["post"], add_special_tokens=False).input_ids) for p in prompts]
    out = {k: [] for k in ("mark", "answer", "q_mark", "q_answer")}
    for b in range(0, len(seqs), batch_size):
        chunk = [pre + post for pre, post in seqs[b:b + batch_size]]
        T = max(map(len, chunk))
        ids = torch.full((len(chunk), T), tok.eos_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(chunk):
            ids[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
        res = model(ids.to(device), attention_mask=mask.to(device), output_hidden_states=True)
        H = torch.stack(res["hidden_states"])
        rows = torch.arange(len(chunk), device=device)
        for key, at in (("mark", [len(pre) - 1 for pre, _ in seqs[b:b + batch_size]]),
                        ("answer", [len(s) - 1 for s in chunk])):
            at = torch.tensor(at, device=device)
            out[key].append(H[:, rows, at].half().cpu().numpy())
            out["q_" + key].append(day_distribution(res["logits"][rows, at], days))
    return {k: np.concatenate(v, 1 if k in ("mark", "answer") else 0) for k, v in out.items()}


def harmonic_basis(C):
    """Seven day centroids (7, d) -> their mean and discrete Fourier vectors (d, 6), harmonics
    1-3 as (cos, sin) pairs."""
    mu = C.mean(0); th = 2 * np.pi * np.arange(7) / 7
    B = [f(m * th) @ (C - mu) for m in (1, 2, 3) for f in (np.cos, np.sin)]
    return mu, np.stack(B, 1)


def ring_coords(H, mu, B):
    """Angle (in days, 0-7) and radius in the first harmonic, scaled so the centroids sit at
    radius 1 and angle d: least-squares coefficients on the Fourier vectors, times 7/2."""
    c = (H - mu) @ np.linalg.pinv(B).T * 3.5
    z = c[:, 0] + 1j * c[:, 1]
    return np.mod(np.angle(z) * 7 / (2 * np.pi), 7), np.abs(z)


def day_span(H, mu, B):
    """Orthonormal coordinates (N, 6) in the centroids' span, harmonic 1 in the first two."""
    return (H - mu) @ np.linalg.qr(B)[0]


def off_plane(P):
    """Share of each displacement (N, 6) in ``day_span`` coordinates outside the ring plane."""
    return (P[:, 2:] ** 2).sum(1) / (P ** 2).sum(1)


def _features(Q, kind):
    if kind == "log":
        L = np.log(Q); return L - L.mean(1, keepdims=True)
    return Q


def fit_predict(Q_fit, H_fit, Q, kind, ridge=1e-6):
    """Fit H ~ features(Q) + b by ridge on (Q_fit, H_fit) and predict at Q. ``kind``: 'log'
    (centred log-probabilities, ours) or 'prob' (probabilities: centroids averaged by belief)."""
    X = _features(Q_fit, kind); xm, hm = X.mean(0), H_fit.mean(0)
    Xc = X - xm
    W = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(X.shape[1]), Xc.T @ (H_fit - hm))
    return (_features(Q, kind) - xm) @ W + hm


def spoke_predict(H, Q, C, mu):
    """Spokes' picture, given its best radius: each point on the spoke of its most likely day,
    at the radius that fits it best."""
    V = C[Q[:, :7].argmax(1)] - mu
    r = ((H - mu) * V).sum(1) / (V * V).sum(1)
    return mu + r[:, None] * V


def r2(H, Hhat):
    return 1 - ((H - Hhat) ** 2).sum() / ((H - H.mean(0)) ** 2).sum()


def pair_position(angle, A, B, sep):
    """Where an angle (in days) sits between a pair's spokes: 0 on A's, 1 on B's, measured from A
    along the short arc toward B, in units of their separation ``sep``."""
    toward = np.where((B - A) % 7 == sep, 1, -1)
    return toward * (np.mod(angle - A + 3.5, 7) - 3.5) / sep
