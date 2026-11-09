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
    if len(names) <= 2:
        return " or ".join(names)
    return ", ".join(names[:-1]) + ", or " + names[-1]


HEADER = ("Each line says how likely each day is. The day that actually happens this week is drawn "
          "from those odds.\n\n")


def all_prompts(shots=0, seed=0):
    """[dict] with the sentence ``pre`` (ends at the period read as ``mark``), the answer slot
    ``post``, the stated ``weights`` (7,) and labels: family, level, A, B, sep (pairs), start, size
    (sets), cont (continuation index). With ``shots``, ``pre`` starts with HEADER and that many
    completed lines from the prompt's family (other continuations), each ending in a day drawn from
    its own stated weights: the prompt then describes a sampling process, so the model's prediction
    at the answer slot should follow the stated odds."""
    out = _sentences()
    if not shots:
        return out
    rng = np.random.default_rng(seed)
    for p in out:
        pool = [x for x in out if x["family"] == p["family"] and x["cont"] != p["cont"]]
        lines = [x["pre"] + x["post"] + " " + DAYS[rng.choice(7, p=x["weights"])] + "."
                 for x in (pool[i] for i in rng.choice(len(pool), shots, replace=False))]
        p["context"] = HEADER + "".join(l + "\n" for l in lines)
    return [{**{k: v for k, v in p.items() if k != "context"}, "pre": p["context"] + p["pre"]} for p in out]


def _sentences():
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


def harmonic_coords(H, mu, B):
    """(N, 3) complex: each point's place in harmonics 1-3, the least-squares coefficients on the
    Fourier vectors times 7/2, so that centroid d sits at exp(2 pi i m d / 7) in harmonic m."""
    c = (H - mu) @ np.linalg.pinv(B).T * 3.5
    return c[:, 0::2] + 1j * c[:, 1::2]


def ring_coords(H, mu, B):
    """Angle (in days, 0-7) and radius in the first harmonic (centroids at radius 1, angle d)."""
    z = harmonic_coords(H, mu, B)[:, 0]
    return np.mod(np.angle(z) * 7 / (2 * np.pi), 7), np.abs(z)


def canonical(z, A, B, sep):
    """Harmonic coordinates (N, 3) of pair prompts turned so that A sits at Monday's place and B
    ``sep`` days clockwise of it: harmonic m rotated by -m*A, mirrored when B was named the other way."""
    m = np.arange(1, 4)
    rot = z * np.exp(-2j * np.pi * m * np.asarray(A)[:, None] / 7)
    return np.where(((np.asarray(B) - A) % 7 == sep)[:, None], rot, np.conj(rot))


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


NUMBERS = ["one", "two", "three", "four", "five", "six", "seven"]


def _circ(a, b):
    return min((a - b) % 7, (b - a) % 7)


def arithmetic_prompts():
    """The manifold-steering paper's "Q: What day is {k} days after {day}?\\nA:", certain or with
    an either/or in the offset ("four or two days after Tuesday"), the start day ("four days after
    Tuesday or Wednesday"), or three consecutive offsets. [dict] with ``pre`` the whole prompt
    (``post`` empty, so it tokenizes as one text), uniform ``weights`` over the answers, A and B the
    first and second option's answers and ``sep`` their circular distance."""
    out, q = [], "Q: What day is {} days after {}?\nA:"
    def add(family, weights, text, **kw):
        out.append(dict(family=family, weights=weights, pre=text, post="", **kw))
    for d, k in itertools.product(range(7), range(1, 8)):
        A = (d + k) % 7
        add("certain", np.eye(7)[A], q.format(NUMBERS[k - 1], DAYS[d]), day=d, k=k, A=A, B=A, sep=0)
    for d, k, k2 in itertools.product(range(7), range(1, 8), range(1, 8)):
        if k != k2:
            A, B = (d + k) % 7, (d + k2) % 7
            add("two_k", _two(A, B, .5), q.format(f"{NUMBERS[k - 1]} or {NUMBERS[k2 - 1]}", DAYS[d]),
                day=d, k=k, k2=k2, A=A, B=B, sep=_circ(A, B))
    for d, d2, k in itertools.product(range(7), range(7), range(1, 8)):
        if d != d2:
            A, B = (d + k) % 7, (d2 + k) % 7
            add("two_day", _two(A, B, .5), q.format(NUMBERS[k - 1], f"{DAYS[d]} or {DAYS[d2]}"),
                day=d, day2=d2, k=k, A=A, B=B, sep=_circ(A, B))
    for d, k in itertools.product(range(7), range(1, 6)):
        w = np.zeros(7); w[[(d + k + i) % 7 for i in range(3)]] = 1 / 3
        add("three_k", w, q.format(f"{NUMBERS[k - 1]}, {NUMBERS[k]}, or {NUMBERS[k + 1]}", DAYS[d]),
            day=d, k=k, A=(d + k + 1) % 7, B=(d + k + 1) % 7, sep=0)
    return out


def nearest_centroid_accuracy(X_fit, g_fit, X, g):
    """Share of the points ``X`` whose nearest centroid of the ``X_fit`` groups is their group ``g``."""
    labels = np.unique(g_fit)
    C = np.stack([X_fit[g_fit == l].mean(0) for l in labels])
    nearest = labels[((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)]
    return (nearest == g).mean()
