"""Relational structures described in text, King et al.'s format (Diego-Simon et al. 2026): a one-line
intro, every adjacent fact once in random order and random direction, then 'Who are ...?' listing the
entities in a fixed order. An entity's representation is the residual at its token in that list. The
predictions are the answer tables the structure fixes, rows and columns indexed by position along it."""
import re

import numpy as np
import torch

# Diego-Simon et al. (COLM 2026), Table 1: entity names (in-distribution), and the domain header with its
# vocabulary slot ``[]``; ``fact(i, j, k)`` states that j sits one positive step along axis k from i, in a
# random direction (either phrasing).
KING = {
    "ordinality": {
        "names": list("dfcyuzogrejln"), "dims": 1,
        "header": "**Mathematical Variables** You will receive a description of set of mathematical constants, each "
                  "associated with an unknown real number. **Vocabulary** The constants are named according to []. "
                  "**Relations** The following relations are used to describe the relative position of a constant with "
                  "respect to another one on the real axis: (greater than, less than) **Goal** Infer the position of each "
                  "constant on the real axis so that **all** constraints are satisfied. **Scene description:**",
        "fact": [lambda i, j: f"{j} is greater than {i}.", lambda i, j: f"{i} is less than {j}."]},
    "spatial": {
        "names": "bag ball clock lamp ring key train plane boat bus bike shirt coin".split(), "dims": 2,
        "header": "**Geometrical Layout** You will receive a description of an abstract scene laid out on a regular square "
                  "grid. **Vocabulary** The scene contains objects, named: [] **Relations** The following spatial "
                  "relations are used to describe the relative position of the objects on the grid: (above, below, left "
                  "of, right of) **Goal** Infer the grid coordinates of every object so that **all** spatial constraints "
                  "are satisfied. **Scene description:**",
        "fact": [lambda i, j: f"The {j} is to the right of the {i}.", lambda i, j: f"The {i} is to the left of the {j}.",
                 lambda i, j: f"The {j} is on top of the {i}.", lambda i, j: f"The {i} is below the {j}."]},
}


def grid_graph(n, d, rng):
    """Algorithm 1: ``n`` distinct points of Z^d grown from the origin by random signed unit steps, and the
    edges ``(i, j, k)`` whenever point j is one positive unit step along axis k from point i.
    Returns ``(pos (n, d) int, edges)``."""
    pos = [np.zeros(d, dtype=int)]
    while len(pos) < n:
        p = pos[rng.integers(len(pos))] + rng.choice([-1, 1]) * np.eye(d, dtype=int)[rng.integers(d)]
        if not any(np.array_equal(p, q) for q in pos):
            pos.append(p)
    pos = np.array(pos)
    edges = [(i, j, k) for i in range(n) for j in range(n) for k in range(d)
             if np.array_equal(pos[j] - pos[i], np.eye(d, dtype=int)[k])]
    return pos, edges


def graph_distances(n, edges):
    """Shortest-path distances of the undirected graph (Floyd-Warshall)."""
    D = np.full((n, n), np.inf); np.fill_diagonal(D, 0)
    for i, j, _ in edges:
        D[i, j] = D[j, i] = 1
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D.astype(int)


def incidence(n, edges, n_types):
    """``M[i, j, k]`` = +1 for an edge (i, j, k), -1 for its reverse (j, i, k), 0 otherwise."""
    M = np.zeros((n, n, n_types))
    for i, j, k in edges:
        M[i, j, k], M[j, i, k] = 1, -1
    return M


def king_graph(domain, n, rng):
    """A graph of their domain: ``n`` names from the pool (kept in pool order), placed on the lattice at random.
    Returns ``(names, pos, edges)`` with node i the i-th name."""
    pool = KING[domain]["names"]
    names = [pool[i] for i in sorted(rng.permutation(len(pool))[:n])]
    pos, _ = grid_graph(n, KING[domain]["dims"], rng)
    pos = pos[rng.permutation(n)]                                     # the bijection f: positions -> names
    d = KING[domain]["dims"]
    edges = [(i, j, k) for i in range(n) for j in range(n) for k in range(d)
             if np.array_equal(pos[j] - pos[i], np.eye(d, dtype=int)[k])]
    return names, pos, edges


def king_description(domain, names, edges, rng):
    """``(text, listing)``: the header with the vocabulary, every edge once in random order and phrasing, then
    'Who are a, b, and c?' in vocabulary order (the listing their reads are taken at)."""
    fact = KING[domain]["fact"]
    facts = [fact[2 * k + rng.integers(2)](names[i], names[j]) for i, j, k in (edges[e] for e in rng.permutation(len(edges)))]
    ask = "Who are " + ", ".join(names[:-1]) + ", and " + names[-1] + "?"
    return "\n".join([KING[domain]["header"].replace("[]", ", ".join(names)), " ".join(facts), ask]), list(names)


DOMAINS = {
    "chain": {"intro": "I am going to describe how some variables compare.",
              "names": list("abcdefghjkmnpqrstuvwxyz")},
    "cycle": {"intro": "I am going to describe a circular metro line.",
              "names": "lake pub zoo park mall bank school museum church market bridge station tower garden "
                       "library harbor castle beach airport stadium".split()},
}


def sample_order(domain, n, rng):
    """``n`` distinct names of the domain in a random order: position 0 first along the structure."""
    names = DOMAINS[domain]["names"]
    return [names[i] for i in rng.permutation(len(names))[:n]]


def _fact(domain, x, y, rng):
    """The fact that ``y`` follows ``x`` (one step up the chain, or the next stop), in a random direction."""
    if domain == "chain":
        return f"{y} is greater than {x}." if rng.random() < 0.5 else f"{x} is less than {y}."
    return f"The stop after {x} is {y}."


def describe(domain, order, listing, rng):
    """The prompt for one structure: intro, the adjacent facts shuffled, then the entity list in ``listing`` order."""
    n = len(order)
    steps = [(order[i], order[i + 1]) for i in range(n - 1)] + ([(order[-1], order[0])] if domain == "cycle" else [])
    facts = [_fact(domain, x, y, rng) for x, y in (steps[i] for i in rng.permutation(len(steps)))]
    return "\n".join([DOMAINS[domain]["intro"], " ".join(facts), "Who are " + ", ".join(listing) + "?"])


def encode(tok, text, listing):
    """``(ids, at)``: BOS plus the prompt's token ids, and the index of each listed entity's token in the final line."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, spans = [tok.bos_token_id] + enc["input_ids"], [(0, 0)] + enc["offset_mapping"]
    start = text.rindex("\n") + len("Who are ")
    at = []
    for e in listing:
        s = start + re.search(rf"\b{re.escape(e)}\b", text[start:]).start()   # the entity's word in the list
        at.append(next(i for i, (a, b) in enumerate(spans) if a <= s < b))
        start = s + len(e)
    return ids, at


@torch.no_grad()
def read_listed(model, tok, prompts, batch_size=32):
    """The residual at every listed entity of every prompt, at every layer. ``prompts``: [(text, listing)]
    with listings of equal length -> (layers, n_prompts, n, d) float16, entities in listing order."""
    device = next(model.parameters()).device
    enc = [encode(tok, text, listing) for text, listing in prompts]
    out = None
    for b in range(0, len(enc), batch_size):
        chunk = enc[b:b + batch_size]
        L = max(len(ids) for ids, _ in chunk)
        x = torch.full((len(chunk), L), tok.eos_token_id, device=device); mask = torch.zeros_like(x)
        for r, (ids, _) in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids); mask[r, :len(ids)] = 1
        H = torch.stack(model(x, attention_mask=mask, output_hidden_states=True)["hidden_states"])       # (layers, b, T, d)
        at = torch.tensor([a for _, a in chunk], device=device)                                          # (b, n)
        picked = H[:, torch.arange(len(chunk), device=device)[:, None], at].half().cpu().numpy()       # (layers, b, n, d)
        if out is None:
            out = np.zeros((len(H), len(enc), at.shape[1], H.shape[-1]), dtype=np.float16)
        out[:, b:b + len(chunk)] = picked
    return out


def read_entities(model, tok, graphs, batch_size=32):
    """Mean residual per entity, by position along its structure, over each graph's descriptions.
    ``graphs``: [(order, [(text, listing), ...])] -> (layers, n_graphs, n, d) float32."""
    H = read_listed(model, tok, [d for _, descs in graphs for d in descs], batch_size).astype(np.float32)
    out, r = [], 0
    for order, descs in graphs:
        k = len(descs)
        perm = [[listing.index(e) for e in order] for _, listing in descs]                              # listing -> position
        out.append(np.mean([H[:, r + i, p] for i, p in enumerate(perm)], 0)); r += k
    return np.stack(out, 1)


def comparison_table(n):
    """Is the entity at position i greater than the one at j: +1 yes, -1 no, 0 on the diagonal."""
    p = np.arange(n)
    return np.sign(p[:, None] - p[None, :]).astype(float)


def within_table(n, k):
    """On a loop of ``n`` stops: is stop j within ``k`` stops after stop i (+1) or not (-1)."""
    gap = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
    return np.where((gap >= 1) & (gap <= k), 1.0, -1.0)


def next_table(n, cyclic):
    """Which entity comes right after i: 1 at it, 0 elsewhere (the chain's last entity has none)."""
    N = np.zeros((n, n))
    for i in range(n if cyclic else n - 1):
        N[i, (i + 1) % n] = 1.0
    return N
