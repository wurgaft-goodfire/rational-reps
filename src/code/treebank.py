"""Universal Dependencies treebanks for the syntactic Polar Probe (Diego-Simon et al. 2024): sentences as
words, 0-based heads and base relation labels; tree distances; and each word's residual at chosen layers,
averaged over its sub-tokens."""
import re

import numpy as np
import torch

WEB = re.compile(r"\S+@\S+|https?://|www\.", re.I)                    # sentences with e-mail or web addresses are dropped


def parse_conllu(path):
    """[{"words", "heads", "rels"}] for every sentence of a CoNLL-U file without an e-mail or web address.
    Multi-word token lines and empty nodes are skipped; relation subtypes are dropped (obl:tmod -> obl)."""
    out, rows = [], []
    for line in open(path, encoding="utf-8").read().split("\n") + [""]:
        if not line:
            if rows:
                words = [r[1] for r in rows]
                if not WEB.search(" ".join(words)):
                    out.append({"words": words, "heads": [int(r[6]) - 1 for r in rows],
                                "rels": [r[7].split(":")[0] for r in rows], "upos": [r[3] for r in rows]})
            rows = []
        elif not line.startswith("#"):
            r = line.split("\t")
            if r[0].isdigit():
                rows.append(r)
    return out


def root_paths(s):
    """Each word's tree position: the relations on the path from the root down to it (the root's is ())."""
    out = [None] * len(s["heads"])
    def path(i):
        if out[i] is None:
            h = s["heads"][i]
            out[i] = () if h < 0 else path(h) + (s["rels"][i],)
        return out[i]
    return [path(i) for i in range(len(out))]


def context_counts(sentences, child_types, classes):
    """``(C, rows, cols)``: how often a word at each tree position (row: its root path) has each class at each
    relative path (column: (path, class)). Paths: its parent ("up"), grandparent ("up2"), and its first child
    by each relation in ``child_types`` (("down", rel)); a missing word, or one of another class, counts as
    'none'. Rows in order of first appearance."""
    paths = ["up", "up2"] + [("down", t) for t in child_types]
    labels = list(classes) + ["none"]
    cols = [(p, c) for p in paths for c in labels]; col = {c: i for i, c in enumerate(cols)}
    rows, row, counts = [], {}, []
    for s in sentences:
        rp = root_paths(s); heads, rels, upos = s["heads"], s["rels"], s["upos"]
        children = {}
        for j, h in enumerate(heads):
            if h >= 0:
                children.setdefault((h, rels[j]), j)
        for i in range(len(heads)):
            if rp[i] not in row:
                row[rp[i]] = len(rows); rows.append(rp[i]); counts.append(np.zeros(len(cols)))
            h = heads[i]; g = heads[h] if h >= 0 else -1
            at = {"up": h, "up2": g, **{("down", t): children.get((i, t), -1) for t in child_types}}
            for p, j in at.items():
                c = upos[j] if j >= 0 and upos[j] in classes else "none"
                counts[row[rp[i]]][col[(p, c)]] += 1
    return np.array(counts), rows, cols


def tree_distances(heads):
    """Path lengths between every pair of words in the dependency tree."""
    n = len(heads); D = np.full((n, n), n, dtype=int); np.fill_diagonal(D, 0)
    for i, h in enumerate(heads):
        if h >= 0:
            D[i, h] = D[h, i] = 1
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


@torch.no_grad()
def read_words(model, tok, sentences, layers, batch_size=32):
    """Each word's residual at ``layers``, averaged over its sub-tokens (each word encoded with a leading
    space after BOS). Returns (len(layers), total words, d) float16, words of all sentences concatenated."""
    device = next(model.parameters()).device
    enc = []
    for s in sentences:
        ids, spans = [tok.bos_token_id], []
        for w in s["words"]:
            t = tok.encode(" " + w, add_special_tokens=False)
            spans.append((len(ids), len(ids) + len(t))); ids += t
        enc.append((ids, spans))
    out = np.zeros((len(layers), sum(len(s["words"]) for s in sentences), model.config.hidden_size), dtype=np.float16)
    w = 0
    for b in range(0, len(enc), batch_size):
        chunk = enc[b:b + batch_size]
        L = max(len(ids) for ids, _ in chunk)
        x = torch.full((len(chunk), L), tok.eos_token_id, device=device); mask = torch.zeros_like(x)
        for r, (ids, _) in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids); mask[r, :len(ids)] = 1
        hs = model(x, attention_mask=mask, output_hidden_states=True)["hidden_states"]
        H = torch.stack([hs[l] for l in layers])                              # (layers, b, T, d)
        for r, (_, spans) in enumerate(chunk):
            for a, e in spans:
                out[:, w] = H[:, r, a:e].float().mean(1).half().cpu().numpy(); w += 1
    return out
