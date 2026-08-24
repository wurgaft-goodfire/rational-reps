import inspect
import itertools
from collections import namedtuple
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .config import Config

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

    * co-occurrence ``z -> x``: one input slot; a graph's outcomes are its own items, a
      bipartite structure's (words -> attributes) are a separate alphabet;
    * relational ``(a, b) -> c``: two or more input slots;
    * hidden Markov ``x_{1:t} -> z -> x_{t+1}``: the outcome is reached through a latent walk,
      and the target is either the next token or the latent itself.

    Sampling follows from the kernel. ``pointwise`` feeds every input tuple with its exact
    outcome distribution as a soft target. ``sequence`` chains the kernel when its outcomes
    are its inputs (a Markov walk, which needs a start token); otherwise the input is held
    fixed and the outcomes are exchangeable draws from one row. That input can be **observed**
    (``[z, x_1 ... x_L]``, z its own prefix) or **latent** (``latent: true`` — the run sees
    only ``[start, x_1 ... x_L]`` and has to infer z), which is the in-context inference case.
    A latent input walks a transition matrix between draws: the identity holds z fixed, and
    the HMM supplies a real one. Bayes is then the posterior predictive from the prior over z,
    not an oracle that reads z off the input. ``labels`` picks the target: the next token, or
    the latent itself. Subclasses override ``_build`` and only what they genuinely differ on."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(config.random_seed)
        self.sample_mode = config.get("sample_mode", "pointwise")
        self.arity = 1

        self.labels = config.get("labels", "tokens")
        self.prior = self.transition = None                       # set by _build when z is latent
        self._cond, P = self._build(config)
        self.P = P / P.sum(-1, keepdim=True)                      # the kernel p(outcome | z)
        self.n_outcomes = self.P.shape[-1]
        # an unchainable kernel may hide its input; without a transition, z is held fixed
        self.latent = bool(config.get("latent", False)) or self.transition is not None
        if self.latent and self.transition is None:
            self.transition = torch.eye(self.n_items, dtype=torch.float64, device=self.device)
            self.prior = torch.full((self.n_items,), 1.0 / self.n_items,
                                    dtype=torch.float64, device=self.device)
        self._layout()

    def _build(self, config):
        """``(conditional, P)``: the log-conditional the oracle factors and the row-wise
        outcome distribution. Also sets ``n_items`` and any task-specific tensors."""
        spec = _spec(config, "structure", "structure_kwargs")
        L, self.info = graph_build(spec)
        self.n_items = len(L)
        self.features = self.info.get("Z")                        # item feature codes, if any
        return L, torch.tensor(np.exp(L), dtype=torch.float32, device=self.device)

    def _layout(self):
        """Token layout, which follows from the sampler. A latent input is never a token, so
        the stream is outcomes plus a start token (plus a latent block when the targets are
        the states themselves). An observed input is a token: a chained kernel reuses it for
        its outcomes and needs a start token, an exchangeable one gives outcomes their own
        block and uses the input as the prefix, and a pointwise task emits neither."""
        self.chained = self.n_items == self.n_outcomes and not self.latent
        sequence = self.sample_mode == "sequence"
        if self.latent:
            self.label_tokens = torch.arange(self.n_outcomes)
            self.start_token = self.n_outcomes
            latent_targets = self.labels == "latent"
            self.vocab_size = self.n_outcomes + 1 + self.n_items * latent_targets
            self.item_tokens = (self.n_outcomes + 1 + torch.arange(self.n_items) if latent_targets
                                else torch.arange(self.n_outcomes))
            return
        self.item_tokens = torch.arange(self.n_items)
        outcome_block = sequence and not self.chained
        self.label_tokens = (torch.arange(self.n_items, self.n_items + self.n_outcomes)
                             if outcome_block else torch.arange(self.n_outcomes))
        self.start_token = self.n_items if sequence and self.chained else None
        self.vocab_size = self.n_items + self.n_outcomes * outcome_block + (self.start_token is not None)

    # ── interface read by the oracle / trainer / evaluator ──────────────────
    @property
    def conditional(self):
        """The conditional the geometry oracle factors."""
        return self._cond

    @property
    def num_dims(self):
        """Number of target classes: outcomes, or the latent states when they are the target."""
        return self.n_items if self.labels == "latent" else self.n_outcomes

    @property
    def seq_len(self):
        if self.sample_mode != "sequence":
            return self.arity + 1
        return self.context_length + 1                            # prefix (start or z) + L draws

    @property
    def context_length(self):
        return self.config.training_kwargs["context_length"]

    def __len__(self):
        return 1

    def _batch_size(self):
        return self.config.training_kwargs["batch_size"]

    def _markov_walk(self, P, init, length):
        """Sample (B, length) walks under transition ``P``, starting from states ``init``."""
        walk = torch.empty(len(init), length, dtype=torch.long, device=P.device)
        walk[:, 0] = init
        for t in range(1, length):
            walk[:, t] = torch.multinomial(P[walk[:, t - 1]], 1, generator=self.rng).squeeze(1)
        return walk

    def _start_column(self, B):
        return torch.full((B, 1), self.start_token, dtype=torch.long, device=self.device)

    def __getitem__(self, i=None):
        if self.sample_mode != "sequence":
            # full batch: every input tuple with its exact outcome distribution as a soft target
            return {"input_ids": self.item_tokens.to(self.device).unsqueeze(1), "labels": self.P}
        B, L = self._batch_size(), self.context_length
        if self.latent:                                           # [start, x_1 ... x_L], z hidden
            z = torch.multinomial(self.prior, B, replacement=True, generator=self.rng)
            s = self._markov_walk(self.transition, z, L)          # identity transition holds z fixed
            x = torch.multinomial(self.P[s.reshape(-1)], 1, generator=self.rng).reshape(B, L)
            seq = torch.cat([self._start_column(B), x], dim=1)
            labels = seq.clone()
            if self.labels == "latent":
                labels[:, 1:] = s + self.n_outcomes + 1
            return {"input_ids": seq, "labels": labels}
        if self.chained:                                          # [start, x_1 ... x_L]
            init = torch.randint(0, self.n_items, (B,), generator=self.rng, device=self.device)
            seq = torch.cat([self._start_column(B), self._markov_walk(self.P, init, L)], dim=1)
        else:                                                     # [z, x_1 ... x_L], x_t iid | z
            z = torch.randint(self.n_items, (B, 1), generator=self.rng, device=self.device)
            x = torch.multinomial(self.P[z[:, 0]], L, replacement=True, generator=self.rng)
            seq = torch.cat([z, self.n_items + x], dim=1)
        return {"input_ids": seq, "labels": seq.clone()}

    def optimal_predictions(self, input_ids):
        """Bayes targets aligned with the causal shift, i.e. ``logits[:, :-1]``. Pointwise
        (B, 1) -> (B, C); sequence (B, T) -> (B, T-1, C)."""
        if self.sample_mode != "sequence":
            return self.P[input_ids[:, 0]]
        if self.latent:
            return self._posterior_predictive(input_ids[:, 1:])
        if self.chained:
            marginal = self.P.mean(0, keepdim=True)               # dist after the start token
            return torch.cat([self.P, marginal], 0)[input_ids[:, :-1]]
        # z observed as the prefix: every later position predicts the same row P(.|z)
        return self.P[input_ids[:, 0], None].expand(-1, input_ids.shape[1] - 1, -1)

    def _posterior_predictive(self, obs):
        """Bayes for a hidden z: start from the prior, filter the prefix, push through the
        transition. Returns the next-token (or next-state) distribution at every position."""
        beliefs = self.forward_algorithm(obs)                     # (B, L, n_items) = b_1..b_L
        pushed = beliefs[:, :-1] @ self.transition
        if self.labels == "latent":
            prior = self.prior.expand(len(obs), 1, -1)            # predict z_1 before seeing x_1
            return torch.cat([prior, pushed], dim=1).float()
        first = (self.prior @ self.P.double()).expand(len(obs), 1, -1)   # x_1 from the prior alone
        return torch.cat([first, pushed @ self.P.double()], dim=1).float()

    def forward_algorithm(self, obs):
        """Per-position posteriors ``b_t = p(z_t | x_{1:t})``. obs: (B, L) -> (B, L, n_items)."""
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

    def compute_metrics(self, eval_pred, return_all=True):
        """KL(model || true) over the outcome distribution. Handles pointwise (logits (B,C)) and
        sequence (logits (B,T,C)) shapes; for causal sequences drop the final unmatched prediction."""
        logits, labels = eval_pred
        logits = torch.as_tensor(logits, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        if logits.ndim == 3 and labels.ndim == 3:
            logits = logits[:, :-1, :]                                # causal shift
        elif logits.ndim == 3 and labels.ndim == 2:
            logits = logits[:, -1, :]                                 # pointwise input through a sequence model

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
    """Operation task c = a*b: ``.conditional`` is the tensor ``T[a, b, ..., c]``, with arity
    inferred from its shape. Every input tuple carries its exact ``p(c | a, b, ...)``."""

    def _build(self, config):
        T = relation_tensor(_spec(config, "operation", "operation_kwargs"))
        self.arity = T.ndim - 1                                   # (n_items,)*arity + (n_outcomes,)
        self.n_items = T.shape[0]
        self.features = None
        # per-input-tuple outcome distribution, flattened over the tuple for fast lookup
        flat = torch.tensor(T.reshape(-1, T.shape[-1]), dtype=torch.float32, device=self.device)
        return T, flat

    def __getitem__(self, i=None):
        # full batch: every input tuple with its exact p(c | a, b, ...) as a soft target
        items = self.item_tokens.to(self.device)
        inputs = torch.cartesian_prod(*([items] * self.arity))    # (n^arity, arity)
        if self.arity == 1:
            inputs = inputs.unsqueeze(1)
        flat = torch.zeros(len(inputs), dtype=torch.long, device=self.device)
        for j in range(self.arity):
            flat = flat * self.n_items + inputs[:, j]             # row-major over the item axes
        return {"input_ids": inputs, "labels": self.P[flat]}


##### Hidden-Markov dataset (latent graph walk + emissions) #####

class HMMDataset(KernelDataset):
    """The kernel is an emission matrix and the latent walks the structure's graph, so this is
    the latent case with a real transition in place of the identity. ``emission``: "bump"
    (Gaussian in hop distance, symbols = states) or "permuted" (a peaked template permuted per
    state); ``emission_sigma`` sets its width. Sequence-only."""

    def _build(self, config):
        spec = _spec(config, "structure", "structure_kwargs")
        # The HMM treats every structure as its adjacency graph: the latent walks the graph's
        # nodes (tree included — all nodes are states, not only leaves), so the transition and
        # the bump emission share one geometry. gamma / cooccurrence are required.
        L = adjacency_conditional(structure_adjacency(spec), spec["gamma"], spec["cooccurrence"])
        self.n_items = self.n_states = len(L)
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
    """Binary attribute codes of the hypercube's 2^n nodes, (2^n, n), in node order."""
    return np.array([[(v >> b) & 1 for b in range(n)] for v in range(2 ** n)], float)


def hypercube(n):
    """Binary attribute cube: n attributes -> 2^n nodes, neighbors differ in one bit -> analogies."""
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


def word_attribute(N=256, m=8, p=0.3, decay=0.9, scale=1.75, seed=0):
    """Bipartite word -> attribute structure: N words with i.i.d. Bernoulli(p) features
    Z in {0, 1}, outcomes the m attribute tokens, log p(i|x) = log_softmax_i(alpha_i Z_xi)
    with alpha_i = scale * decay^i (p = sparsity, decay = importance, scale = sharpness).
    Returns (L (N, m), {"Z" (N, m), "alpha" (m,)})."""
    from scipy.special import logsumexp
    alpha = scale * decay ** np.arange(m)
    Z = (np.random.default_rng(seed).random((N, m)) < p).astype(float)
    G = Z * alpha
    return G - logsumexp(G, axis=1, keepdims=True), {"Z": Z, "alpha": alpha}


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
    # bipartite (no adjacency): words -> attributes
    "word_attribute": _Structure(None, word_attribute),
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
    """``(L, info)`` for a structure spec: the log-conditional plus whatever the structure
    exposes about its items (e.g. the feature codes ``Z`` of a bipartite structure)."""
    out = _graph_conditional(spec, gamma, cooccurrence)
    return out if isinstance(out, tuple) else (out, {})


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
        return struct.conditional(**spec)
    return adjacency_conditional(_build_adjacency(name, spec), gamma, cooccurrence)


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


# ── Relational operations (order >= 2): the latent structure is an operation c = a*b ──

def modular_add(p=12, arity=2):
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


def set_intersection(nbit=3):
    """c = a & b over feature sets (Boolean lattice) -> the hypercube (one binary coord/feature)."""
    n = 2 ** nbit; T = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            T[a, b, a & b] = 1.0
    return T


def entailment(depth=2, branch=3):
    """c = LCA(a, b) (the meet on an is-a tree) -> a hierarchy embedding."""
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
            T[a, b, next(u for u in anc[b] if u in sa)] = 1.0
    return T


_RELATION_BUILDERS = {
    "modular_add": modular_add, "standard_add": standard_add, "modular_mult": modular_mult,
    "standard_mult": standard_mult, "product": product, "maximum": maximum,
    "averaging": averaging, "set_intersection": set_intersection, "entailment": entailment,
}
GROUP_OPERATIONS = {"modular_add", "modular_mult", "standard_add", "standard_mult", "product"}


def relation_tensor(spec):
    """Build the conditional tensor ``T`` for an operation spec ``{'operation': name, **params}``.
    Graded ops (max/averaging/entailment) factor cleanly; group ops are degenerate (the trained
    model selects a single frequency -> a circle)."""
    params = {k: v for k, v in dict(spec).items() if k != "operation"}
    name = spec["operation"]
    if name not in _RELATION_BUILDERS:
        raise ValueError(f"Unknown operation {name!r}; known: {sorted(_RELATION_BUILDERS)}")
    return _RELATION_BUILDERS[name](**params)
