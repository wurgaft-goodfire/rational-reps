"""Relational structures described in text, King et al.'s format (Diego-Simon et al. 2026): a one-line
intro, every adjacent fact once in random order and random direction, then 'Who are ...?' listing the
entities in a fixed order. An entity's representation is the residual at its token in that list. The
predictions are the answer tables the structure fixes, rows and columns indexed by position along it."""
import numpy as np
import torch

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
        s = text.index(e, start)                                       # the entity's characters in the list
        at.append(next(i for i, (a, b) in enumerate(spans) if a <= s < b))
        start = s + len(e)
    return ids, at


@torch.no_grad()
def read_entities(model, tok, graphs, batch_size=32):
    """Mean residual per entity, by position along its structure, over each graph's descriptions.
    ``graphs``: [(order, [(text, listing), ...])] -> (layers, n_graphs, n, d) float32."""
    device = next(model.parameters()).device
    rows = [(g, order.index(e), ids, i) for g, (order, descs) in enumerate(graphs) for text, listing in descs
            for ids, at in [encode(tok, text, listing)] for e, i in zip(listing, at)]
    prompts = {}
    for g, p, ids, i in rows:
        prompts.setdefault(tuple(ids), []).append((g, p, i))
    n, G = len(graphs[0][0]), len(graphs)
    sums, counts = None, np.zeros((G, n))
    keys = list(prompts)
    for b in range(0, len(keys), batch_size):
        chunk = keys[b:b + batch_size]
        L = max(map(len, chunk))
        x = torch.full((len(chunk), L), tok.eos_token_id, device=device); mask = torch.zeros_like(x)
        for r, ids in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids); mask[r, :len(ids)] = 1
        H = torch.stack(model(x, attention_mask=mask, output_hidden_states=True)["hidden_states"]).float()  # (layers, b, T, d)
        if sums is None:
            sums = torch.zeros(H.shape[0], G, n, H.shape[-1], device=device)
        for r, ids in enumerate(chunk):
            for g, p, i in prompts[ids]:
                sums[:, g, p] += H[:, r, i]; counts[g, p] += 1
    return (sums / torch.tensor(counts, device=device, dtype=torch.float32)[None, :, :, None]).cpu().numpy()


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
