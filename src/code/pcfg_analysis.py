"""Analysis helpers for the PCFG geometry experiments.

Three groups of pure helpers used by the ``pcfg_geometry`` notebook:
  * interpretable prefix-state labels per grammar (the hand sufficient statistic we colour by),
  * canonical constituent spans (gold parse structure, for the unsupervised-tree check),
  * unsupervised constituent recovery from a per-token representation (top-down syntactic-distance
    bracketing, à la Shen et al. 2018) plus the untrained-model CCA control.
"""
import numpy as np


# ── interpretable sufficient-statistic labels ────────────────────────────────
def anbn_count(prefix):
    """Unmatched a's after ``prefix`` (a = 0, b = 1) — the a^n b^n counter."""
    return sum(t == 0 for t in prefix) - sum(t == 1 for t in prefix)


def dyck_stack(prefix, k=2):
    """Stack of open bracket types after ``prefix`` (open i is ``i < k``; a close pops)."""
    st = []
    for t in prefix:
        st.append(t) if t < k else st.pop()
    return tuple(st)


def pl_stack_phase(prefix):
    """``(stack, phase)`` for the control-flow language: constructs pushed on ``if``/``while`` and
    popped on ``end``; phase = the last token seen (which keyword the parser is between)."""
    iftok, then, whiletok, do, end = 1, 2, 3, 4, 5
    st = []
    for t in prefix:
        if t == iftok:
            st.append("if")
        elif t == whiletok:
            st.append("while")
        elif t == end:
            st.pop()
    return tuple(st), (prefix[-1] if len(prefix) else -1)


def pl_nesting(prefix):
    """Control-construct nesting depth — the scalar view of :func:`pl_stack_phase`."""
    return len(pl_stack_phase(prefix)[0])


# ── canonical constituent spans (gold parse structure) ───────────────────────
def dyck_constituents(terminals, k=2):
    """Matched-bracket spans ``(open_pos, close_pos)`` — the unambiguous Dyck constituents."""
    stack, spans = [], set()
    for i, t in enumerate(terminals):
        if t < k:
            stack.append(i)
        else:
            spans.add((stack.pop(), i))
    return spans


def nl_constituents(terminals):
    """Constituent spans of a mini_nl sentence (det adj* noun / verb / prep) by deterministic parse:
    subject NP, object NP, any PP and its NP, and the VP. Terminals: det 0, adj 1, noun 2, verb 3,
    prep 4."""
    det, adj, noun, verb, prep = range(5)
    spans = set()

    def parse_np(i):                                             # det adj* noun -> (span, next index)
        start = i; i += 1                                        # det
        while terminals[i] == adj:
            i += 1
        i += 1                                                   # noun
        spans.add((start, i - 1)); return i
    i = parse_np(0)                                              # subject NP
    vp_start = i; i += 1                                         # verb
    if i < len(terminals) and terminals[i] == det:              # object NP
        i = parse_np(i)
    if i < len(terminals) and terminals[i] == prep:             # PP = prep + NP
        pp_start = i; i += 1
        i = parse_np(i); spans.add((pp_start, i - 1))
    spans.add((vp_start, len(terminals) - 1))                   # VP spans verb..end
    return spans


# ── unsupervised constituent recovery from a representation ──────────────────
def syntactic_distance_brackets(reps):
    """Top-down greedy bracketing of a token sequence from its per-token representation: split each
    span at the adjacent pair with the largest representation gap (Shen et al. 2018). Returns the set
    of induced constituent spans ``(start, end)`` inclusive, length >= 2, root excluded."""
    reps = np.asarray(reps, float)
    L = len(reps)
    gap = np.linalg.norm(np.diff(reps, axis=0), axis=1)          # gap[i] between token i and i+1
    spans = set()

    def split(a, b):                                             # inclusive [a, b], b > a
        i = a + int(np.argmax(gap[a:b]))                         # boundary between i and i+1
        for lo, hi in ((a, i), (i + 1, b)):
            if hi > lo:
                spans.add((lo, hi)); split(lo, hi)
    if L >= 2:
        split(0, L - 1)
    return spans


def constituent_recall(reps, gold):
    """Fraction of gold constituent spans recovered by the unsupervised bracketing of ``reps``."""
    if not gold:
        return 0.0
    pred = syntactic_distance_brackets(reps)
    return len(pred & set(gold)) / len(gold)


# ── held-out linear decoding (the capacity-robust discriminator) ─────────────
def _ridge_fit_r2(Xtr, Ytr, Xte, Yte, alpha):
    """Ridge-regress ``Y`` on standardized ``X`` (bias unregularized); return R² on the test split."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    Xte = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]); A[-1, -1] -= alpha
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    ss_res = ((Yte - Xte @ W) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def heldout_decode_r2(H, F, groups, train_frac=0.6, seed=0, alpha=1.0, train_mask=None):
    """R² of a linear map ``H -> F`` fit on a train split of *distinct prefix groups* and evaluated on
    held-out groups. Because a causal hidden is identical for every occurrence of a prefix, splitting
    by group keeps test prefixes wholly unseen. A wide hidden that merely *spans* F on the training
    prefixes (random features, an untrained net) does not yield a single linear map that generalizes to
    unseen prefixes; a genuine linear representation of the predictive state does. Pass ``train_mask``
    to use a custom split (e.g. train on short prefixes, test on long — length extrapolation)."""
    H = np.asarray(H, float); F = np.asarray(F, float); groups = np.asarray(groups)
    if train_mask is None:
        ids = np.unique(groups)
        rng = np.random.default_rng(seed); rng.shuffle(ids)
        train_ids = set(ids[:int(round(train_frac * len(ids)))].tolist())
        train_mask = np.array([g in train_ids for g in groups])
    train_mask = np.asarray(train_mask, bool)
    return _ridge_fit_r2(H[train_mask], F[train_mask], H[~train_mask], F[~train_mask], alpha)


# ── untrained-model CCA control ──────────────────────────────────────────────
def untrained_twin(config, dataset):
    """A fresh model with the run's architecture but random (pre-training) weights. If the predictive
    state were merely 'linearly fittable' by a wide hidden, this random model would match the trained
    one under CCA; it does not — that gap is what makes CCA≈1 a real result, not a capacity artefact."""
    from omegaconf import open_dict
    from .models import init_model
    mc = config.cfg.model_kwargs.model_config
    with open_dict(mc):
        mc.vocab_size = dataset.vocab_size
        mc.max_position_embeddings = dataset.seq_len
    return init_model(config)
