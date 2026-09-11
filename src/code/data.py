import hashlib
import inspect
import itertools
from collections import namedtuple
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .config import Config
from .geometry import log_conditional


def kernel_hash(dataset):
    """Fingerprint of the kernel ``P``, the outcome distribution a run is fit to. Run dirs are
    keyed by the override string, so a code change to a builder moves the task while leaving the
    config identical; this catches that. Shapes and values both enter the hash.

    Deliberately not ``conditional``, which is the log-space object the geometry oracle factors:
    changing how the oracle reads a task must not invalidate weights trained on the same kernel."""
    target = getattr(dataset, "P", None)
    if target is None:
        target = dataset.conditional
    T = np.ascontiguousarray(np.asarray(getattr(target, "cpu", lambda: target)(), dtype=np.float64))
    return hashlib.sha1(repr(T.shape).encode() + T.tobytes()).hexdigest()[:16]


def apply_lapse(P, lapse=0.05):
    """Mix a conditional with the uniform distribution: ``(1 - lapse) p + lapse / C``. A target with
    zeros is not identified by the loss — KL weights an outcome by its true probability, so an
    impossible one constrains nothing and its logit is free. A lapse rate ("with probability
    ``lapse`` the response is random") makes every outcome possible, hence every logit scored.

    A target that already has full support is returned untouched: there is nothing to identify, and
    mixing in uniform would only flatten its tails onto the floor, which raises the rank spuriously
    (approx_add goes from rank 1 to rank 21). So this is safe to apply to every conditional."""
    P = np.asarray(P, dtype=np.float64)
    if not lapse or (P > 0).all():
        return P
    return (1 - lapse) * P + lapse / P.shape[-1]


def _spec(config, name_key, kwargs_key):
    """Merge a setting's name and its ``*_kwargs`` block into one spec dict."""
    from omegaconf import OmegaConf
    kw = config.get(kwargs_key)
    return {name_key: config.get(name_key),
            **(OmegaConf.to_container(kw, resolve=True) if kw else {})}


##### Init dataset #####
def init_dataset(config, overwrite_data: bool = False):
    """Initialize a dataset from the config's `setting`."""
    if config.setting == "graph-cooccurrence":
        return KernelDataset(config)
    if config.setting == "relational":
        return RelationalDataset(config)
    if config.setting == "hmm":
        return HMMDataset(config)
    raise ValueError(f"Unsupported setting: {config.setting}")


##### Kernel dataset — every task is a kernel; the base all of them specialize #####

class KernelDataset(Dataset):
    """A task is a **kernel**: a conditional over a table ``P(outcome | inputs)``. The settings
    differ only in what the slots are —

    * co-occurrence ``z -> x``: one input slot — that is the pointwise reading; written as
      tokens the same kernel becomes a walk, since a graph's outcomes are its own items,
      while a bipartite structure's (words -> features) are a separate alphabet;
    * relational ``(a, b) -> c``: two or more input slots;
    * hidden Markov ``x_{1:t} -> z -> x_{t+1}``: the outcome is reached through a latent walk,
      and the target is either the next token or the latent itself.

    ``sample_mode`` picks between the two things you can do with a kernel.

    **pointwise** — one input, one outcome, the input observed. Every input tuple is written
    as ``[z_1 ... z_arity, x]`` with a sampled outcome, tiled to ``batch_size`` with fresh
    draws per copy, and the batch also carries the exact row as ``distribution_labels``: a
    causal LM predicts the outcome token (a large batch averages the sampling noise), a readout
    model trains on the exact distribution, and both see the same kernel. Bayes is the kernel row.

    **sequence** — a stream of observations, which a kernel produces in exactly one of two
    ways, decided by the kernel rather than by config:

    * it **chains**, its outcomes being its own inputs, so the stream is a Markov walk
      ``[start, x_1 ... x_L]`` and each observation conditions the next;
    * it does **not** chain, so a stream can only come from one hidden input generating many
      outcomes: ``[start, x_1 ... x_L]`` with z absent. Held fixed here (``z -> x_{1:t}``),
      walked by a subclass with dynamics (the HMM).

    Prefixing an observed z to its own outcomes is not a third case — the draws are i.i.d.
    given z, so a stream states nothing the pointwise kernel does not.

    ``infer_latent`` then picks the target within a sequence: the next observation, or the
    hidden input behind it. It needs a hidden input, so a walk rejects it. Bayes with a hidden
    input is the posterior predictive from the prior over z, not a lookup: with no dynamics the
    posterior is the prior times the accumulated likelihood; with dynamics a subclass filters.
    Subclasses override ``_build`` and only what they genuinely differ on."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(config.random_seed)
        self.sample_mode = config.get("sample_mode", "pointwise")
        self.arity = 1

        self.infer_latent = bool(config.get("infer_latent", False))
        self.prior = self.transition = None                       # a subclass may supply dynamics
        self._cond, P = self._build(config)
        self.P = P / P.sum(-1, keepdim=True)                      # the kernel p(outcome | z)
        self.n_outcomes = self.P.shape[-1]

        self.sequential = self.sample_mode == "sequence"
        # a structure over one node set has its nodes as outcomes, so a stream chains; a structure
        # with a separate outcome set (declared by the structure) hides its input
        self.connected = not self.bipartite
        if self.infer_latent and not (self.sequential and self.bipartite):
            raise ValueError("infer_latent needs a hidden input: a connected stream has none, "
                             "and a pointwise task states its input outright")
        if self.sequential and self.bipartite and self.prior is None:   # z stands still
            self.prior = torch.full((self.n_items,), 1.0 / self.n_items,
                                    dtype=torch.float64, device=self.device)
        self._layout()

    def _build(self, config):
        """``(conditional, P)``: the log-conditional the oracle factors and the row-wise
        outcome distribution. Also sets ``n_items`` and any task-specific tensors."""
        spec = _spec(config, "structure", "structure_kwargs")
        L, self.info = graph_build(spec)
        self.n_items = len(L)
        self.bipartite = self.info["bipartite"]
        self.features = self.info.get("Z")                        # item feature codes, if any
        if "prior" in self.info:                                  # the structure's prior over its hidden inputs
            self.prior = torch.tensor(self.info["prior"], dtype=torch.float64, device=self.device)
        P = apply_lapse(np.exp(L), config.get("lapse", 0.05))
        return np.log(P), torch.tensor(P, dtype=torch.float32, device=self.device)

    def _layout(self):
        """Token layout, which follows from the sampler. A bipartite stream hides its input, so
        the tokens are the outcomes plus a start, and the inputs get a block of their own only
        when they are the target. A connected stream reuses the input tokens for its outcomes
        and needs a start token. A pointwise batch writes the input tuple, then its outcome."""
        if self.sequential and self.bipartite:
            self.item_tokens = (self.n_outcomes + 1 + torch.arange(self.n_items)
                                if self.infer_latent else torch.arange(self.n_outcomes))
            # the scored head slice: the latent block when z is the target, else the outcomes
            self.label_tokens = (self.item_tokens if self.infer_latent
                                 else torch.arange(self.n_outcomes))
            self.start_token = self.n_outcomes
            self.vocab_size = self.n_outcomes + 1 + self.n_items * self.infer_latent
            return
        self.item_tokens = torch.arange(self.n_items)
        # an outcome needs a token of its own only when it is not already an input
        self.label_tokens = (torch.arange(self.n_items, self.n_items + self.n_outcomes)
                             if self.bipartite else torch.arange(self.n_outcomes))
        self.start_token = self.n_items if self.sequential else None
        self.vocab_size = (self.n_items + self.n_outcomes * self.bipartite
                           + (self.start_token is not None))

    # ── interface read by the oracle / trainer / evaluator ──────────────────
    @property
    def conditional(self):
        """The conditional the geometry oracle factors."""
        return self._cond

    @property
    def num_dims(self):
        """Target classes: the outcomes, or the inputs when they are what we infer."""
        return self.n_items if self.infer_latent else self.n_outcomes

    @property
    def seq_len(self):
        if self.sequential:
            return self.context_length + 1                        # start + L observations
        return self.arity + 1                                     # the input tuple, then its outcome

    @property
    def context_length(self):
        return self.config.training_kwargs["context_length"]

    def __len__(self):
        return 1

    def _batch_size(self):
        return self.config.training_kwargs["batch_size"]

    def _markov_walk(self, P, init, length, backtrack=True):
        """Sample (B, length) walks under transition ``P``, starting from states ``init``.
        ``backtrack=False`` forbids the step back to the previous state (the row is renormalised)."""
        walk = torch.empty(len(init), length, dtype=torch.long, device=P.device)
        walk[:, 0] = init
        rows = torch.arange(len(init), device=P.device)
        for t in range(1, length):
            p = P[walk[:, t - 1]]
            if not backtrack and t > 1:
                p = p.clone(); p[rows, walk[:, t - 2]] = 0.0
            walk[:, t] = torch.multinomial(p, 1, generator=self.rng).squeeze(1)
        return walk

    def _start_column(self, B):
        return torch.full((B, 1), self.start_token, dtype=torch.long, device=self.device)

    def __getitem__(self, i=None):
        B = self._batch_size()
        if not self.sequential:
            return self._pointwise_batch()
        L = self.context_length
        if self.connected:                                        # keep sampling from the chain
            init = torch.randint(0, self.n_items, (B,), generator=self.rng, device=self.device)
            walk = self._markov_walk(self.P, init, L, backtrack=self.config.get("backtrack", True))
            seq = torch.cat([self._start_column(B), walk], dim=1)
            return {"input_ids": seq, "labels": seq.clone()}
        init = torch.multinomial(self.prior, B, replacement=True, generator=self.rng)
        z = (self._markov_walk(self.transition, init, L) if self.transition is not None
             else init[:, None].expand(-1, L))                    # no dynamics: one z per episode
        x = torch.multinomial(self.P[z.reshape(-1)], 1, generator=self.rng).reshape(B, L)
        seq = torch.cat([self._start_column(B), x], dim=1)
        labels = seq.clone()
        if self.infer_latent:
            labels[:, 1:] = z + self.n_outcomes + 1
        return {"input_ids": seq, "labels": labels}

    def _pointwise_batch(self):
        """Every input tuple as ``[z..., x]`` with a sampled outcome, plus the exact row: a
        causal LM predicts the outcome token, a readout model trains on the distribution.
        Only the outcome is scored — with several input slots the stream also contains
        transitions between inputs, which are independent and so unlearnable."""
        inputs = self._input_tuples()
        rows = self.P[self._row_index(inputs)]
        x = torch.multinomial(rows, 1, generator=self.rng) + self.n_items * self.bipartite
        seq = torch.cat([inputs, x], dim=1)
        labels = torch.full_like(seq, -100)
        labels[:, -1] = seq[:, -1]
        return {"input_ids": seq, "labels": labels, "distribution_labels": rows}

    def _tile(self, tuples):
        """Repeat the enumeration to reach ``batch_size`` (at least one full copy): downstream
        outcome draws are fresh per copy, so a large batch averages the sampling noise."""
        return tuples.repeat(max(1, -(-self._batch_size() // len(tuples))), 1)

    def input_tuples(self):
        """Every input the task defines, once, ``(n, 1)``. A stated input is its item token; a
        stream's inputs are its observation tokens (the length-1 histories), whatever the target.
        Training tiles this to fill a batch; analysis wants it untiled."""
        tokens = torch.arange(self.n_outcomes) if self.sequential and self.bipartite else self.item_tokens
        return tokens.to(self.device).unsqueeze(1)

    def _input_tuples(self):
        """``input_tuples`` tiled to ``batch_size``. The sampler appends the outcome column
        itself, so this is the tuple alone — never ``model_inputs``."""
        return self._tile(self.input_tuples())

    def model_inputs(self):
        """What the model reads for each input the task defines, one row per input: the tuple,
        with the start token in front when the layout has one. ``(n_items ** arity, arity [+1])``."""
        tuples = self.input_tuples()
        if self.start_token is None:
            return tuples
        start = torch.full_like(tuples[:, :1], int(self.start_token))
        return torch.cat([start, tuples], dim=1)

    def _row_index(self, inputs):
        """Row of ``P`` addressed by each input tuple."""
        return inputs[:, 0]

    def optimal_predictions(self, input_ids):
        """Bayes targets aligned with the causal shift, i.e. ``logits[:, :-1]``. Pointwise
        (B, 1) -> (B, C); sequence (B, T) -> (B, T-1, C)."""
        if not self.sequential:                                   # z is stated: Bayes is the row
            return self.P[self._row_index(input_ids)]
        if self.connected:
            marginal = self.P.mean(0, keepdim=True)               # dist after the start token
            return torch.cat([self.P, marginal], 0)[input_ids[:, :-1]]
        return self._posterior_predictive(input_ids[:, 1:])

    def _posterior_predictive(self, obs):
        """Bayes for a hidden input: the posterior over z hypotheses after each prefix, pushed
        back through the kernel. z is held fixed, so the posterior is just the prior times the
        accumulated likelihood. obs: (B, L) -> (B, L, num_dims)."""
        logp = self.P.double().log()[:, obs]                      # (n_items, B, L)
        beliefs = torch.log_softmax(self.prior.log()[:, None, None] + logp.cumsum(-1), dim=0)
        beliefs = beliefs.permute(1, 2, 0).exp()                  # (B, L, n_items) = b_1 .. b_L
        return self._predict_from(beliefs, len(obs))

    def _predict_from(self, beliefs, B):
        """Next-position targets from the per-prefix posteriors, shifted so position t predicts
        what follows it: the prior (or its predictive) leads, then each belief in turn."""
        if self.infer_latent:
            return torch.cat([self.prior.expand(B, 1, -1), beliefs[:, :-1]], dim=1).float()
        first = (self.prior @ self.P.double()).expand(B, 1, -1)   # x_1 from the prior alone
        return torch.cat([first, beliefs[:, :-1] @ self.P.double()], dim=1).float()

    def compute_metrics(self, eval_pred, return_all=True):
        """KL(model || true) over the outcome distribution. Handles pointwise (logits (B,C)) and
        sequence (logits (B,T,C)) shapes; for causal sequences drop the final unmatched prediction."""
        logits, labels = eval_pred
        logits = torch.as_tensor(logits, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        if logits.ndim == 3 and labels.ndim == 3:
            logits = logits[:, :-1, :]                                # causal shift
        elif logits.ndim == 3 and labels.ndim == 2:
            logits = logits[:, -2, :]                                 # the slot that predicts the outcome

        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            loss = nn.KLDivLoss(reduction="none")(log_probs, labels).sum(dim=-1)
            # An all-zero target row marks an unscored position (the bipartite sampler scores only the
            # outcome); average the KL over scored predictions so the rest don't dilute the Bayes gap.
            valid = labels.sum(dim=-1) > 0 if labels.ndim == logits.ndim else None
            if valid is not None and not bool(valid.all()):
                loss = loss[valid]
            loss = loss.reshape(loss.shape[0] if loss.ndim else 1, -1)
        results = {"loss": loss.mean().item()}
        if return_all:
            results["all_loss"] = loss.detach().cpu()
        return results


##### Relational dataset (order >= 2, combination) #####

class RelationalDataset(KernelDataset):
    """Operation task ``c = a * b``: the relation's tensor ``T[a, b, ..., c]``, with arity
    inferred from its shape. Pointwise by construction — extra input slots are the only thing
    that differs from any other kernel. ``pair_inputs`` folds the tuple into a single token."""

    def _build(self, config):
        T = relation_tensor(_spec(config, "relation", "relation_kwargs"))
        self.arity = T.ndim - 1                                   # (n_items,)*arity + (n_outcomes,)
        self.n_items = T.shape[0]
        self.features = None
        # per-input-tuple outcome distribution, flattened over the tuple for fast lookup
        M = apply_lapse(T.reshape(-1, T.shape[-1]), config.get("lapse", 0.05))
        flat = torch.tensor(M, dtype=torch.float32, device=self.device)
        if config.get("pair_inputs", False):
            self.n_items, self.arity = flat.shape[0], 1           # each input tuple is one token
        self.bipartite = T.shape[-1] != self.n_items              # the outcome alphabet is its own
        return log_conditional(M), flat

    def input_tuples(self):
        items = self.item_tokens.to(self.device)
        tuples = torch.cartesian_prod(*([items] * self.arity))    # (n ** arity, arity)
        return tuples.unsqueeze(1) if self.arity == 1 else tuples

    def _row_index(self, inputs):
        flat = torch.zeros(len(inputs), dtype=torch.long, device=self.device)
        for j in range(self.arity):
            flat = flat * self.n_items + inputs[:, j]             # row-major over the item axes
        return flat


##### Hidden-Markov dataset (latent graph walk + emissions) #####

class HMMDataset(KernelDataset):
    """The hidden-input sequence with real dynamics: the state walks the structure's graph
    instead of standing still, and the kernel is an emission matrix. ``emission``: "bump"
    (Gaussian in hop distance, symbols = states) or "permuted" (a peaked template permuted per
    state); ``emission_sigma`` sets its width. As everywhere, ``infer_latent`` chooses between
    predicting the next symbol and predicting the state."""

    def _build(self, config):
        spec = _spec(config, "structure", "structure_kwargs")
        # The HMM treats every structure as its adjacency graph: the latent walks the graph's
        # nodes (tree included — all nodes are states, not only leaves), so the transition and
        # the bump emission share one geometry. gamma / cooccurrence are required.
        L = adjacency_conditional(structure_adjacency(spec), spec["gamma"], spec["cooccurrence"])
        self.n_items = self.n_states = len(L)
        self.bipartite = True                                     # emissions are never the states
        self.features = None

        T = torch.tensor(np.exp(L), dtype=torch.float64, device=self.device)
        self.transition = T / T.sum(-1, keepdim=True)
        self.prior = self._stationary(self.transition)            # prior = stationary distribution
        return L, self._emissions(config, spec)

    def _emissions(self, config, spec):
        """Emission matrix (n_states, n_symbols); also fixes ``n_symbols``."""
        self.n_symbols = int(config.get("n_symbols", self.n_states))
        emission = config.get("emission", "bump")
        sigma = float(config.get("emission_sigma", 1.0))
        if emission == "bump":
            # Gaussian in hop distance on the structure's graph; symbols correspond to states
            # (M = n_states), so sigma reads as 'how many hops does an emission alias'
            Em = np.exp(-(structure_distance(spec) ** 2) / (2 * sigma ** 2))
            self.n_symbols = self.n_states
            return torch.tensor(Em, dtype=torch.float64, device=self.device)
        if emission == "permuted":
            # one peaked template over the M symbols, independently permuted per state
            j = np.arange(self.n_symbols); template = np.exp(-(j ** 2) / (2 * sigma ** 2))
            g = torch.Generator(device=self.device); g.manual_seed(config.random_seed + 1)
            perms = torch.stack([torch.randperm(self.n_symbols, generator=g, device=self.device)
                                 for _ in range(self.n_states)])
            return torch.tensor(template, dtype=torch.float64, device=self.device)[perms]
        raise ValueError(f"unknown emission {emission!r} (expected 'bump' or 'permuted')")

    @staticmethod
    def _stationary(T, iters=2000):
        v = torch.full((T.shape[0],), 1.0 / T.shape[0], dtype=T.dtype, device=T.device)
        for _ in range(iters):
            v = v @ T
        return v / v.sum()

    def _posterior_predictive(self, obs):
        """The state moves between emissions, so the posterior is the forward recursion rather
        than a running product: transition, then reweight by the emission."""
        beliefs = self.forward_algorithm(obs)                     # (B, L, n_states) = b_1 .. b_L
        return self._predict_from(beliefs @ self.transition, len(obs))

    def forward_algorithm(self, obs):
        """Per-position posteriors ``b_t = p(s_t | x_{1:t})``. obs: (B, L) -> (B, L, n_states)."""
        emission = self.P.double()
        b = self.prior.expand(len(obs), -1).clone()
        out = []
        for t in range(obs.shape[1]):
            if t > 0:
                b = b @ self.transition                           # not before the first emission
            b = b * emission[:, obs[:, t]].T                      # emission update
            b = b / b.sum(-1, keepdim=True)
            out.append(b.clone())
        return torch.stack(out, 1)

    @property
    def latent_conditional(self):
        """Log-conditional of the latent transition — factor for the predicted latent geometry."""
        return self._cond


# =====================================================================================
# Generative processes: the conditional each setting exposes for the geometry oracle to
# factor and for the sampler to draw from.
# =====================================================================================

# ── Graph co-occurrence (order-1, symmetric): a graph -> diffusion -> log p(word|context) ──

def ring(n):
    """Cyclic graph Z_n (periodic) -> Fourier modes -> circle."""
    A = np.zeros((n, n)); idx = np.arange(n)
    A[idx, (idx + 1) % n] = 1.0; A[idx, (idx - 1) % n] = 1.0
    return A


def chain(n):
    """Path graph (open boundary) -> standing-wave cosines -> horseshoe/arch."""
    A = np.zeros((n, n)); idx = np.arange(n - 1)
    A[idx, idx + 1] = 1.0; A[idx + 1, idx] = 1.0
    return A


def grid(n, m=None):
    """2-D lattice, n rows x m cols (m defaults to n) -> a distorted sheet."""
    m = n if m is None else m
    N = n * m; A = np.zeros((N, N)); ix = lambda i, j: i * m + j
    for i in range(n):
        for j in range(m):
            if i + 1 < n: A[ix(i, j), ix(i + 1, j)] = A[ix(i + 1, j), ix(i, j)] = 1.0
            if j + 1 < m: A[ix(i, j), ix(i, j + 1)] = A[ix(i, j + 1), ix(i, j)] = 1.0
    return A


def sphere(n, k=8):
    """kNN graph over ~uniform (Fibonacci) sphere points -> degree-1 harmonics -> sphere."""
    idx = np.arange(n); gold = np.pi * (1 + 5 ** 0.5)
    zc = 1 - 2 * (idx + 0.5) / n; rc = np.sqrt(1 - zc ** 2); ph = gold * idx
    pts = np.stack([rc * np.cos(ph), rc * np.sin(ph), zc], 1)
    D2 = ((pts[:, None] - pts[None]) ** 2).sum(-1)
    A = np.zeros((n, n))
    for i in range(n):
        nb = np.argsort(D2[i])[1:k + 1]; A[i, nb] = A[nb, i] = 1.0
    return A


def communities(n=25, k=3, p_in=0.45, p_out=0.04, seed=0):
    """Stochastic block model: k communities of n nodes each -> community-block geometry."""
    rng = np.random.default_rng(seed); lab = np.repeat(np.arange(k), n)
    N = len(lab); A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < (p_in if lab[i] == lab[j] else p_out):
                A[i, j] = A[j, i] = 1.0
    return A


def cube_codes(n):
    """Binary feature codes of the hypercube's 2^n nodes, (2^n, n), in node order."""
    return np.array([[(v >> b) & 1 for b in range(n)] for v in range(2 ** n)], float)


def hypercube(n):
    """Binary feature cube: n features -> 2^n nodes, neighbors differ in one bit -> analogies."""
    B = cube_codes(n)
    H = (B[:, None] != B[None]).sum(-1)
    return (H == 1).astype(float)


def complete(n):
    """Complete graph (no structure) -> every item equidistant -> a simplex."""
    return 1.0 - np.eye(n)


def cartesian_product(A, B):
    """Product graph on nodes (i, j): move along one factor at a time (shared support)."""
    nA, nB = len(A), len(B); N = nA * nB; G = np.zeros((N, N)); ix = lambda i, j: i * nB + j
    for i in range(nA):
        for j in range(nB):
            for k in np.where(A[i] > 0)[0]: G[ix(i, j), ix(k, j)] = 1.0
            for l in np.where(B[j] > 0)[0]: G[ix(i, j), ix(i, l)] = 1.0
    return G


def block_diagonal(*mats):
    """Disjoint union (block-diagonal adjacency): separate structures on separate words."""
    N = sum(len(m) for m in mats); G = np.zeros((N, N)); o = 0
    for m in mats:
        k = len(m); G[o:o + k, o:o + k] = m; o += k
    return G


def walk_cooccur(A, gamma=0.8):
    """Diffusion co-occurrence ``(I - gamma P)^-1``, symmetrized (decays with diffusion distance)."""
    A = np.asarray(A, dtype=np.float64)
    deg = A.sum(1); P = A / deg[:, None]; pi = deg / deg.sum()
    M = np.linalg.inv(np.eye(len(A)) - gamma * P)
    J = pi[:, None] * M
    return 0.5 * (J + J.T)


def heat_cooccur(A, beta=0.8):
    """Heat-kernel co-occurrence ``exp(beta * A)``. On a Cartesian-product graph it factorizes
    into per-factor kernels, so the log-conditional is additive over the factors."""
    from scipy.linalg import expm
    return expm(beta * np.asarray(A, dtype=np.float64))


def logcond_from_sim(K):
    """Symmetric co-occurrence kernel -> row-normalized log-conditional ``log p(word|context)``."""
    K = np.asarray(K, dtype=np.float64)
    return np.log(K / K.sum(1, keepdims=True) + 1e-12)


def geodesic_cooccur(A, gamma=0.8):
    """Shortest-path (geodesic) kernel ``gamma ** d(i,j)``: full self-similarity on the diagonal
    (``gamma**0 = 1``, unlike adjacency's zero diagonal), decaying by graph distance."""
    from scipy.sparse.csgraph import shortest_path
    D = shortest_path(np.asarray(A, dtype=np.float64) > 0, directed=False, unweighted=True)
    return gamma ** D


def cooccur_logcond(A, gamma=0.8):
    """Graph adjacency -> ``log p(token | nearby context)`` via diffusion co-occurrence."""
    return logcond_from_sim(walk_cooccur(A, gamma))


def tree_conditional(n=3, k=2, parents=None, gamma=0.85, leaves_only=True):
    """Tree diffusion log-conditional over all nodes. Balanced (depth n, branching k) or
    unbalanced via an explicit `parents` list (``parents[i]`` = parent index, root = -1);
    ``leaves_only=True`` restricts to leaves."""
    parents = tree_parents(n, k, parents)
    J = walk_cooccur(tree_adjacency(parents), gamma)
    if leaves_only:
        internal = {p for p in parents if p is not None and p >= 0}
        ids = [i for i in range(len(parents)) if i not in internal]
        J = J[np.ix_(ids, ids)]
    return logcond_from_sim(J)


def tree_parents(n=3, k=2, parents=None):
    """Parent-index list for a tree (root = -1). Explicit ``parents`` if given, else a balanced
    depth-n, branching-k tree in DFS pre-order, so each node is immediately followed by its whole
    subtree — subtrees stay contiguous, giving the conditional its nested-block (ultrametric) look."""
    if parents is not None:
        return list(parents)
    out = []
    def build(depth, parent):
        me = len(out); out.append(parent)
        if depth < n:
            for _ in range(k):
                build(depth + 1, me)
    build(0, -1)
    return out


def tree_adjacency(parents):
    """Symmetric parent-child adjacency for a parent-index list."""
    N = len(parents); A = np.zeros((N, N))
    for c, p in enumerate(parents):
        if p is not None and p >= 0:
            A[c, p] = A[p, c] = 1.0
    return A


def _tree_graph(n=2, k=2, parents=None, **_):
    """Tree adjacency over *all* nodes (balanced depth-n branch-k, or an explicit ``parents`` list).
    ``**_`` swallows conditional-only keys (gamma / leaves_only)."""
    return tree_adjacency(tree_parents(n, k, parents))


def feature_cube(N=256, m=8, p=0.3, decay=0.9, scale=1.75, seed=0,
                 spectrum="geometric", beta=1.5, rho=0.0, blocks=1,
                 paired=False, all_bundles=False):
    """Bipartite word -> feature structure: N words with i.i.d. Bernoulli(p) features
    Z in {0, 1}, outcomes the m feature tokens, Lc(x, i) = alpha_i Z_xi and
    L = log_softmax(Lc) (p = sparsity, scale = sharpness). ``spectrum`` sets the importance
    ladder: "geometric" gives alpha_i = scale*decay^i, "power" gives alpha_i = scale*(i+1)^-beta.
    ``rho`` > 0 correlates the features through a Gaussian copula, preserving the Bernoulli(p)
    marginals; ``blocks`` splits them into groups that each share their own factor
    (blocks=1 is equicorrelation). ``all_bundles`` makes the words every bundle in {0,1}^m
    once (N ignored). ``paired`` gives each feature a yes/no token pair, columns
    [1+, ..., m+, 1-, ..., m-], with Lc(x, i+) = alpha_i Z_xi and Lc(x, i-) = alpha_i (1 - Z_xi).
    Returns (L (n_words, m or 2m), {"Z" (n_words, m), "alpha" (m,)})."""
    from scipy.special import logsumexp
    alpha = scale * {"geometric": decay ** np.arange(m),
                     "power": (1.0 + np.arange(m)) ** -beta}[spectrum]
    rng = np.random.default_rng(seed)
    if all_bundles:
        Z = cube_codes(m)
    elif rho == 0.0:
        Z = (rng.random((N, m)) < p).astype(float)
    else:                                          # one-factor Gaussian copula
        from scipy.stats import norm
        f = rng.normal(size=(N, blocks))[:, np.arange(m) * blocks // m]
        g = np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.normal(size=(N, m))
        Z = (g > norm.ppf(1 - p)).astype(float)
    Lc = np.c_[Z * alpha, (1 - Z) * alpha] if paired else Z * alpha
    return Lc - logsumexp(Lc, axis=1, keepdims=True), {"Z": Z, "alpha": alpha, "bipartite": True}


# ── Hypothesis spaces (bipartite): hypotheses -> observations, strong sampling p(x|h) = 1/|h| ──
# Zero cells are exact zeros (log -inf) so the dataset's lapse applies; each structure supplies
# its own prior over hypotheses.

def _size_prior(sizes, prior, sigma):
    """Prior over hypotheses from their sizes: uniform, or Erlang ``s exp(-s / sigma)`` (T&G 2001)."""
    sizes = np.asarray(sizes, dtype=float)
    g = {"uniform": np.ones_like(sizes), "erlang": sizes * np.exp(-sizes / sigma)}[prior]
    return g / g.sum()


def _uniform_on(members, n_items):
    """log p(x | h) for hypotheses given as boolean masks over items: ``-log|h|`` inside, -inf outside."""
    M = np.asarray(members, dtype=float)
    with np.errstate(divide="ignore"):
        return np.log(M / M.sum(1, keepdims=True))


def interval_hypotheses(n=10, prior="erlang", sigma=3.0):
    """Bayesian generalization on a line (Shepard 1987; Tenenbaum & Griffiths 2001): items 1..n,
    hypotheses every interval [a, b], p(x | h) = 1/|h| on the interval. Returns (L (n(n+1)/2, n),
    {"hypotheses": (a, b) rows, "prior", "bipartite"})."""
    hyps = np.array([(a, b) for a in range(1, n + 1) for b in range(a, n + 1)])
    x = np.arange(1, n + 1)
    members = (x[None] >= hyps[:, :1]) & (x[None] <= hyps[:, 1:])
    sizes = hyps[:, 1] - hyps[:, 0] + 1
    return _uniform_on(members, n), {"hypotheses": hyps, "prior": _size_prior(sizes, prior, sigma),
                                     "bipartite": True}


def rectangle_hypotheses(n=6, prior="erlang", sigma=2.0):
    """T&G 2001's two-dimensional case: items the cells of an n x n grid (row-major), hypotheses
    every axis-aligned rectangle [a1, b1] x [a2, b2], p(x | h) = 1/area. The prior factors over
    the two side lengths. Returns (L ((n(n+1)/2)^2, n^2), {"hypotheses": (a1, b1, a2, b2) rows,
    "positions" (n^2, 2), "prior", "bipartite"})."""
    iv = np.array([(a, b) for a in range(1, n + 1) for b in range(a, n + 1)])
    hyps = np.array([(a1, b1, a2, b2) for a1, b1 in iv for a2, b2 in iv])
    pos = np.array([(i, j) for i in range(1, n + 1) for j in range(1, n + 1)])
    members = ((pos[None, :, 0] >= hyps[:, :1]) & (pos[None, :, 0] <= hyps[:, 1:2]) &
               (pos[None, :, 1] >= hyps[:, 2:3]) & (pos[None, :, 1] <= hyps[:, 3:4]))
    side = lambda k: hyps[:, k + 1] - hyps[:, k] + 1
    p = _size_prior(side(0), prior, sigma) * _size_prior(side(2), prior, sigma)
    return _uniform_on(members, n * n), {"hypotheses": hyps, "positions": pos, "prior": p / p.sum(),
                                         "bipartite": True}


def subtree_hypotheses(n=3, k=3, parents=None):
    """Xu & Tenenbaum's taxonomy: items the leaves of a tree, hypotheses every node (the set of
    leaves below it), p(x | h) = 1/|leaves(h)|, uniform prior over nodes. Returns
    (L (n_nodes, n_leaves), {"parents", "leaves" (leaf ids per node), "prior", "bipartite"})."""
    parents = tree_parents(n, k, parents)
    children = {i: [c for c, p in enumerate(parents) if p == i] for i in range(len(parents))}
    leaves_of = {}
    def collect(i):
        leaves_of[i] = [i] if not children[i] else sum((collect(c) for c in children[i]), [])
        return leaves_of[i]
    collect(0)
    leaf_ids = leaves_of[0]
    members = np.array([[x in set(leaves_of[h]) for x in leaf_ids] for h in range(len(parents))])
    return _uniform_on(members, len(leaf_ids)), {
        "parents": parents, "leaves": [leaves_of[h] for h in range(len(parents))],
        "prior": np.full(len(parents), 1.0 / len(parents)), "bipartite": True}


# Krumhansl & Kessler (1982) probe-tone profiles, C major / C minor, pitch classes C..B
KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def key_profiles(beta=1.0):
    """Key finding (Krumhansl & Schmuckler): items the 12 pitch classes, hypotheses the 24 keys,
    p(x | key) proportional to the key's probe-tone profile to the power ``beta`` (sharpness).
    Returns (L (24, 12), {"keys": [(tonic, mode)], "prior", "bipartite"})."""
    rows, keys = [], []
    for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            rows.append(np.roll(beta * np.log(prof), tonic)); keys.append((tonic, mode))
    L = np.array(rows)
    L = L - np.log(np.exp(L).sum(1, keepdims=True))
    return L, {"keys": keys, "prior": np.full(24, 1.0 / 24), "bipartite": True}


# One structure = one graph: it builds an adjacency (used by the HMM latent walk and by drawing)
# and may additionally define its own conditional (tree's ultrametric), which overrides the
# generic adjacency->cooccurrence derivation. A bipartite structure has no adjacency and only a
# conditional, whose outcomes are a token block of their own.
_Structure = namedtuple("_Structure", ["adjacency", "conditional"], defaults=[None])
_STRUCTURES = {
    "ring": _Structure(ring), "chain": _Structure(chain), "grid": _Structure(grid),
    "sphere": _Structure(sphere), "communities": _Structure(communities),
    "hypercube": _Structure(hypercube), "complete": _Structure(complete),
    "tree": _Structure(_tree_graph, tree_conditional),
    # bipartite (no adjacency): words -> features; hypothesis spaces -> observations
    "feature_cube": _Structure(None, feature_cube),
    "intervals": _Structure(None, interval_hypotheses),
    "rectangles": _Structure(None, rectangle_hypotheses),
    "subtrees": _Structure(None, subtree_hypotheses),
    "keys": _Structure(None, key_profiles),
}


def _build_adjacency(name, params):
    params = {k: v for k, v in params.items() if k != "view"}   # 'view' is presentation-only (plotting)
    if name in ("product", "stack"):
        As = [_build_adjacency(f["structure"], {k: v for k, v in f.items() if k != "structure"})
              for f in params["factors"]]
        return reduce(cartesian_product, As) if name == "product" else block_diagonal(*As)
    if name not in _STRUCTURES:
        raise ValueError(f"Unknown structure {name!r}; known: {sorted(_STRUCTURES)}")
    return _STRUCTURES[name].adjacency(**params)


def graph_build(spec, gamma=0.8, cooccurrence="diffusion"):
    """``(L, info)`` for a structure spec: the log-conditional plus what the structure exposes
    about its items: ``bipartite`` (a separate outcome set, so the input can be hidden) and,
    for a feature structure, the item codes ``Z``."""
    return _graph_conditional(spec, gamma, cooccurrence)


def graph_conditional(spec, gamma=0.8, cooccurrence="diffusion"):
    """The log-conditional of a structure spec (see ``graph_build`` for the item info)."""
    return graph_build(spec, gamma, cooccurrence)[0]


def _graph_conditional(spec, gamma=0.8, cooccurrence="diffusion"):
    """Log-conditional for a structure spec ``{'structure': name, **params}``. A structure with its
    own conditional (tree) uses it directly; otherwise ``adjacency_conditional`` converts the
    adjacency. ``gamma`` and ``cooccurrence`` may also be passed as spec keys, overriding the args."""
    spec = dict(spec)
    name = spec.pop("structure")
    gamma = spec.pop("gamma", gamma)
    cooccurrence = spec.pop("cooccurrence", cooccurrence)
    spec.pop("view", None)                                    # 'view' is presentation-only (plotting)
    struct = _STRUCTURES.get(name)
    if struct is not None and struct.conditional is not None:
        # pass the task gamma only to conditionals that take it (tree does)
        if "gamma" in inspect.signature(struct.conditional).parameters:
            spec.setdefault("gamma", gamma)
        out = struct.conditional(**spec)
        return out if isinstance(out, tuple) else (out, {"bipartite": False})
    if cooccurrence == "gaussian":
        # Gaussian tuning in the structure's canonical coordinates (not the adjacency):
        # log p(x|i) = -gamma/2 ||p_i - q_x||^2 + const_i — rank <= dim(p) + 1 before centering,
        # exact von Mises (kappa = gamma) on the ring, the quadratic log-kernel on the chain.
        # ``m`` puts the outcomes on a coarse grid over the same manifold (continuous latent,
        # few outcomes), making the kernel bipartite.
        from scipy.special import logsumexp
        m = spec.pop("m", None)
        pos = structure_positions(name, spec)
        out = pos if m is None else structure_positions(name, spec, count=m)
        lo = -gamma / 2 * ((pos[:, None] - out[None]) ** 2).sum(-1)
        return lo - logsumexp(lo, axis=1, keepdims=True), {"bipartite": m is not None}
    return adjacency_conditional(_build_adjacency(name, spec), gamma, cooccurrence), {"bipartite": False}


def structure_positions(name, params, count=None):
    """Canonical node coordinates for the Gaussian kernel: chain = the integer line, ring = the
    unit circle, complete = simplex vertices (identity basis). ``count`` places that many points
    over the same manifold (the chain's segment, the ring's full circle)."""
    n = params["n"]
    if name == "chain":
        p = np.arange(n, dtype=float) if count is None else np.linspace(0, n - 1, count)
        return p[:, None]
    if name == "ring":
        k = n if count is None else count
        th = 2 * np.pi * np.arange(k) / k
        return np.c_[np.cos(th), np.sin(th)]
    if name == "complete":
        if count is not None:
            raise ValueError("complete has no coarser outcome grid")
        return np.eye(n)
    if name == "grid":
        if count is not None or "m" in params:
            raise ValueError("grid coordinates are the square n x n lattice (m is the Gaussian's outcome count)")
        return np.array([(i, j) for i in range(n) for j in range(n)], dtype=float)   # row-major, as `grid`
    raise ValueError(f"No canonical coordinates for structure {name!r}")


def adjacency_conditional(A, gamma=0.8, cooccurrence="diffusion"):
    """Turn an adjacency into a log-conditional: ``"diffusion"`` (``(I-gamma P)^-1``), ``"geodesic"``
    (``gamma**d``), ``"adjacency"`` (raw one-step neighbours), or ``"heat"`` (``exp(gamma * A)``).
    ``gamma`` is the decay for diffusion/geodesic and the diffusion time for heat."""
    if cooccurrence == "diffusion":
        return cooccur_logcond(A, gamma)
    if cooccurrence == "geodesic":
        return logcond_from_sim(geodesic_cooccur(A, gamma))
    if cooccurrence == "adjacency":
        return logcond_from_sim(A)
    if cooccurrence == "heat":
        return logcond_from_sim(heat_cooccur(A, gamma))
    raise ValueError(f"Unknown cooccurrence {cooccurrence!r}; use diffusion | geodesic | adjacency | heat")


def structure_adjacency(spec):
    """Adjacency matrix of a structure's graph — for the HMM latent walk and for drawing. Every
    structure has one (tree included); its special conditional lives in graph_conditional.
    Strips the task-level keys (gamma / cooccurrence)."""
    params = {k: v for k, v in dict(spec).items() if k not in ("structure", "gamma", "cooccurrence")}
    return _build_adjacency(spec["structure"], params)


def structure_distance(spec):
    """Hop (geodesic) distance matrix of a structure's graph — one scale-free metric for every
    structure, straight from its adjacency (the HMM bump emission's ``sigma`` is in hops)."""
    from scipy.sparse.csgraph import shortest_path
    return shortest_path(structure_adjacency(spec) > 0, directed=False, unweighted=True)


def stacked_conditional(specs, mode="union", gamma=0.8):
    """Log-conditional combining several structure specs on one representation. ``mode``:
    ``"union"`` (block-diagonal adjacency, disjoint supports), ``"product"`` (Cartesian, shared
    support), or ``"share"`` (summed adjacency, same nodes)."""
    As = [structure_adjacency(s) for s in specs]
    if mode == "union":
        A = block_diagonal(*As)
    elif mode == "product":
        A = reduce(cartesian_product, As)
    elif mode == "share":
        A = sum(As)
    else:
        raise ValueError(f"Unknown stacking mode {mode!r}; use union | product | share")
    return cooccur_logcond(A, gamma)


# ── Relations (order >= 2): the latent structure is a relation c = a*b ──

def modular_add(p=11, arity=2):
    """c = (a + b + ...) mod p  -> a circle (turning addition into rotation)."""
    T = np.zeros((p,) * arity + (p,))
    for idx in itertools.product(range(p), repeat=arity):
        T[idx + (sum(idx) % p,)] = 1.0
    return T


def standard_add(m=10):
    """c = a + b (no wraparound) -> an open arc."""
    T = np.zeros((m, m, 2 * m - 1))
    for a in range(m):
        for b in range(m):
            T[a, b, a + b] = 1.0
    return T


def modular_mult(p=13, g=2):
    """c = (a * b) mod p over the units (indexed by discrete log) -> a circle."""
    units = list(range(1, p)); dlog = {}; v = 1
    for x in range(p - 1):
        dlog[v] = x; v = (v * g) % p
    n = p - 1; T = np.zeros((n, n, n))
    for a in units:
        for b in units:
            T[dlog[a], dlog[b], dlog[(a * b) % p]] = 1.0
    return T


def standard_mult(m=10):
    """c = a * b -> the log number line, arched (x2 = equal steps)."""
    T = np.zeros((m, m, m * m))
    for a in range(1, m + 1):
        for b in range(1, m + 1):
            T[a - 1, b - 1, a * b - 1] = 1.0
    return T


def product(p=14, q=9):
    """Two parallel modular additions (mod p and mod q) on paired items -> a torus."""
    els = [(i, j) for i in range(p) for j in range(q)]; n = p * q
    idx = {e: k for k, e in enumerate(els)}
    T = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            c = ((els[a][0] + els[b][0]) % p, (els[a][1] + els[b][1]) % q)
            T[a, b, idx[c]] = 1.0
    return T


def maximum(n=8):
    """c = max(a, b) on a total order -> an ordered line."""
    T = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            T[a, b, max(a, b)] = 1.0
    return T


def averaging(grid=4, sigma=0.7):
    """c = midpoint(a, b) on a 2-D grid (soft) -> a flat (Euclidean) plane."""
    P = np.array([(i, j) for i in range(grid) for j in range(grid)], float); n = len(P)
    T = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            w = np.exp(-((P - (P[a] + P[b]) / 2) ** 2).sum(1) / (2 * sigma ** 2))
            T[a, b] = w / w.sum()
    return T


def _graded_set(nbit, bit_op, salience=None):
    """Draw one of the ``nbit`` features uniformly and report whether ``bit_op`` holds for it:
    outcomes are ``(feature, yes)`` then ``(feature, no)``.

    Sampling from the *result set* instead makes a one-member answer a sharper row than a
    three-member one, so the cube comes out stretched and needs a null outcome to reach rank nbit
    at all (a distribution over k outcomes has only k-1 free numbers). Asking about every feature
    gives each one an equal binary coordinate, so the code is the hypercube exactly."""
    w = np.full(nbit, 1.0) if salience is None else np.asarray(salience, float)[:nbit]
    w = w / w.sum()                                   # how often each feature is the one asked about
    n = 2 ** nbit
    T = np.zeros((n, n, 2 * nbit))
    for a in range(n):
        for b in range(n):
            for i in range(nbit):
                hit = bit_op(a >> i & 1, b >> i & 1)
                T[a, b, i if hit else nbit + i] = w[i]
    return T


def set_intersection(nbit=3, graded=True):
    """c = a & b over feature sets (Boolean lattice) -> the hypercube. ``graded``: uniform over the
    members of the intersection (Tversky's *common* features); else one-hot on the subset itself."""
    if graded:
        return _graded_set(nbit, lambda x, y: x and y)
    n = 2 ** nbit; T = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            T[a, b, a & b] = 1.0
    return T


def set_difference(nbit=3, graded=True, salience=(1.0, 0.5, 0.25)):
    """c = a & ~b: Tversky's *distinctive* features of a. Asymmetric — complementing one slot is
    not a relabelling the shared embedding can absorb. ``salience`` is how often each feature is
    the one asked about, Tversky's feature weights: equal weights make two axes the same length and
    leave that plane fixed only up to a rotation, unequal ones give a box with pinned edges."""
    m = 2 ** nbit - 1
    if graded:
        return _graded_set(nbit, lambda x, y: x and not y, salience)
    n = 2 ** nbit; T = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            T[a, b, a & ~b & m] = 1.0
    return T


def feature_taxonomy(chains=2, depth=2):
    """Shared features when the features form chains: the specific one entails the general, so an
    item is a *consistent* feature set — one level per chain. The answer to "draw a feature; do
    both have it?" is then the pair of shared levels, so the code is a grid, one axis per chain,
    rather than the cube independent features would give."""
    items = list(itertools.product(range(depth + 1), repeat=chains))
    feats = [(c, l) for c in range(chains) for l in range(1, depth + 1)]
    T = np.zeros((len(items), len(items), 2 * len(feats)))
    for i, A in enumerate(items):
        for j, B in enumerate(items):
            shared = [min(x, y) for x, y in zip(A, B)]
            for m, (c, l) in enumerate(feats):
                T[i, j, m if shared[c] >= l else len(feats) + m] = 1.0 / len(feats)
    return T


def pitch_transpose(n=12, n_out=23, sigma=5.0, kappa=1.5, per_octave=12):
    """Transpose a note by an interval and name the result. Responses confuse neighbouring pitches
    (Gaussian in height) and notes an octave apart (von Mises in chroma) — Shepard's two components
    of pitch. ``log p`` is then quadratic plus cosine in ``c - (a + b)``, so the code is one linear
    dimension plus a cos/sin pair: a helix. Dense, so no lapse is applied — but the line is carried
    by the quadratic tails, and KL weights a logit by the true probability, so ``sigma`` and
    ``n_out`` must keep the smallest entry KL-visible (here 9e-6); a sharper kernel leaves the far
    logits unscored and networks fold the line into an arc."""
    s = (np.arange(n * n) // n + np.arange(n * n) % n).reshape(n, n)
    c = np.arange(n_out)
    L = (-(c[None, None, :] - s[..., None]) ** 2 / (2 * sigma ** 2)
         + kappa * np.cos(2 * np.pi * (c[None, None, :] - s[..., None]) / per_octave))
    P = np.exp(L - L.max(-1, keepdims=True))
    return P / P.sum(-1, keepdims=True)


def symmetric_difference(nbit=3, graded=True):
    """c = a ^ b: Tversky's *distinctive* features, and addition in Z_2^nbit. As a class label the
    group makes it flat-spectrum and gauge-only; as a set it is rank nbit with the code on the bits.
    Only ``a ^ a`` is empty, so the null outcome carries 1/2^nbit of the rows."""
    if graded:
        return _graded_set(nbit, lambda x, y: x != y)
    n = 2 ** nbit; T = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            T[a, b, a ^ b] = 1.0
    return T


def between(n=8):
    """The interval spanned by a and b on a total order: uniform over [min, max]. Unlike ``maximum``
    it cannot discard the smaller operand — every unordered pair gets its own answer."""
    T = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            lo, hi = min(a, b), max(a, b)
            T[a, b, lo:hi + 1] = 1.0 / (hi - lo + 1)
    return T


def common_divisors(n=12):
    """Common divisors of a and b — the divisors of gcd(a, b), the meet in the divisibility lattice.
    Items and outcomes are 1..n; 1 divides everything, so the answer is never empty."""
    from math import gcd
    T = np.zeros((n, n, n))
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            d = [c for c in range(1, n + 1) if gcd(a, b) % c == 0]
            T[a - 1, b - 1, [c - 1 for c in d]] = 1.0 / len(d)
    return T


def approx_add(m=12, weber=0.7):
    """Estimate a + b under scalar variability: noise proportional to magnitude, i.e. Gaussian in
    log space (Weber-Fechner). Items are the magnitudes 1..m and outcomes the achievable sums
    2..2m, both laid out as plain integers — so the logarithmic geometry comes from the noise law,
    not from the outcome positions. With constant-width noise the same construction predicts a
    linear line instead, which is what makes the log line a claim rather than an assumption.

    The line is exactly ``log(a+b)`` at any ``weber``, but a sharper kernel drops the smallest
    entries below what KL can score, and the unscored far logits bow the learned line."""
    from scipy.special import softmax
    out = np.arange(2, 2 * m + 1, dtype=float)
    T = np.zeros((m, m, len(out)))
    for a in range(1, m + 1):
        for b in range(1, m + 1):
            T[a - 1, b - 1] = softmax(-((np.log(out) - np.log(a + b)) ** 2) / (2 * weber ** 2))
    return T


def entailment(depth=3, branch=2, graded=True):
    """The common ancestors of a and b on an is-a tree -> a hierarchy embedding. Those are exactly
    the ancestors of LCA(a, b), so ``graded`` spreads over that path (never empty: the root is
    always common) while ``graded=False`` keeps the one-hot on the LCA alone."""
    parent = [None]; children = [[]]; frontier = [0]
    for _ in range(depth):
        new = []
        for u in frontier:
            for _ in range(branch):
                i = len(parent); parent.append(u); children.append([]); children[u].append(i); new.append(i)
        frontier = new
    n = len(parent)
    anc = []
    for a in range(n):
        chain = []; u = a
        while u is not None:
            chain.append(u); u = parent[u]
        anc.append(chain)
    T = np.zeros((n, n, n))
    for a in range(n):
        sa = set(anc[a])
        for b in range(n):
            shared = [u for u in anc[b] if u in sa]          # ancestors of the LCA, root last
            if graded:
                T[a, b, shared] = 1.0 / len(shared)
            else:
                T[a, b, shared[0]] = 1.0                     # the LCA itself
    return T


# Depth-3 binary tree, breadth-first, matching what ``entailment`` builds.
_ENTAILMENT_TREE = [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]


def answer_classes(relation, **kwargs):
    """Which answer class each input pair falls into, and how to draw that class set.

    Pairs sharing a target row share an answer, so the geometry lives over classes rather than
    pairs. Where the classes carry a structure a figure can draw — the cube's vertices, the tree's
    nodes — they are indexed to match it; otherwise identical target rows are simply grouped.
    Returns ``(class per pair, draw spec or None)``."""
    T = relation_tensor({"relation": relation, **kwargs})
    n, flat = T.shape[0], T.reshape(-1, T.shape[-1])
    a, b = np.arange(n * n) // n, np.arange(n * n) % n
    if relation in ("set_intersection", "set_difference", "symmetric_difference"):
        nbit = int(np.log2(n))
        op = {"set_intersection": lambda x, y: x & y,
              "set_difference": lambda x, y: x & ~y & (n - 1),
              "symmetric_difference": lambda x, y: x ^ y}[relation]
        return op(a, b), {"structure": "hypercube", "n": nbit}
    if relation == "entailment":
        parents = _ENTAILMENT_TREE
        def ancestors(v):
            out = []
            while v != -1:
                out.append(v); v = parents[v]
            return out
        lca = [max(set(ancestors(x)) & set(ancestors(y)), key=lambda u: len(ancestors(u)))
               for x, y in zip(a, b)]
        return np.array(lca), {"structure": "tree", "parents": parents}
    return np.unique(flat, axis=0, return_inverse=True)[1].ravel(), None


_RELATION_BUILDERS = {
    "modular_add": modular_add, "standard_add": standard_add, "modular_mult": modular_mult,
    "standard_mult": standard_mult, "product": product, "maximum": maximum,
    "averaging": averaging, "set_intersection": set_intersection, "entailment": entailment,
    "symmetric_difference": symmetric_difference,
    "set_difference": set_difference, "feature_taxonomy": feature_taxonomy,
    "pitch_transpose": pitch_transpose, "between": between,
    "common_divisors": common_divisors, "approx_add": approx_add,
}
GROUP_RELATIONS = {"modular_add", "modular_mult", "standard_add", "standard_mult", "product"}


def relation_tensor(spec):
    """Build the conditional tensor ``T`` for a relation spec ``{'relation': name, **params}``.
    Graded ops (max/averaging/entailment) factor cleanly; group ops are degenerate (the trained
    model selects a single frequency -> a circle)."""
    params = {k: v for k, v in dict(spec).items() if k != "relation"}
    name = spec["relation"]
    if name not in _RELATION_BUILDERS:
        raise ValueError(f"Unknown relation {name!r}; known: {sorted(_RELATION_BUILDERS)}")
    return _RELATION_BUILDERS[name](**params)
