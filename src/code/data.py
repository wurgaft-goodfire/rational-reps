import inspect
import itertools
from collections import namedtuple
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .config import Config
from .geometry import predicted_geometry

##### Init dataset #####
def init_dataset(config, overwrite_data: bool = False):
    """Initialize a dataset from the config's `setting`."""
    if config.setting == "graph-cooccurrence":
        return GraphCooccurrenceDataset(config)
    if config.setting == "distribution-family":
        return DistributionFamilyDataset(config)
    if config.setting == "relational":
        return RelationalDataset(config)
    if config.setting == "hmm":
        return HMMDataset(config)
    if config.setting == "lpcfg":
        return LPCFGDataset(config)
    raise ValueError(f"Unsupported setting: {config.setting}")


##### Graph co-occurrence dataset (order-1, symmetric) #####

class GraphCooccurrenceDataset(Dataset):
    """Order-1 latent-structure task. A graph -> diffusion co-occurrence -> the log-conditional
    `log p(token | nearby context)` (exposed as `.conditional`, the object the geometry oracle
    factors). Items are the graph's nodes; the generative transition is the row-normalized
    conditional, so what a network learns and what we predict come from the same object.

    Two `sample_mode`s share that transition:
      - "pointwise": (item -> neighbor) pairs; makes the arity-1 linear readout well-posed.
      - "sequence":  Markov walks under the transition; for the transformer / RNN and future ICL.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(config.random_seed)

        from omegaconf import OmegaConf
        sk = config.get("structure_kwargs")
        structure_kwargs = OmegaConf.to_container(sk, resolve=True) if sk else {}
        # gamma / cooccurrence (the co-occurrence kernel) ride in structure_kwargs; graph_conditional
        # reads them off the spec (like family_kwargs / operation_kwargs for the other settings).
        spec = {"structure": config.get("structure"), **structure_kwargs}
        self._L = graph_conditional(spec)                        # (n, n) log-conditional (numpy)

        self.n_nodes = len(self._L)
        self.arity = 1
        self.sample_mode = config.get("sample_mode", "pointwise")
        self.item_tokens = torch.arange(self.n_nodes)
        self.start_token = self.n_nodes                           # extra token for sequence starts

        # generative transition P[i] = p(next | i), row-normalized exp(conditional)
        P = torch.tensor(np.exp(self._L), dtype=torch.float32, device=self.device)
        self.P = P / P.sum(-1, keepdim=True)
        self.marginal = self.P.mean(0)                            # dist after the start token

    # ── interface read by the oracle / trainer / evaluator ──────────────────
    @property
    def conditional(self):
        """The order-1 log-conditional the geometry oracle factors."""
        return self._L

    @property
    def num_dims(self):
        """Number of outcome classes (= nodes); used by the Evaluator vocab remap + k-linear head."""
        return self.n_nodes

    @property
    def vocab_size(self):
        return self.n_nodes + 1                                    # items + start token

    @property
    def seq_len(self):
        if self.sample_mode == "sequence":
            return self.config.training_kwargs["context_length"] + 1
        return self.arity + 1

    def __len__(self):
        return 1

    def _batch_size(self):
        return self.config.training_kwargs["batch_size"]

    def __getitem__(self, i=None):
        if self.sample_mode == "sequence":
            return self._sequence_batch()
        return self._pointwise_batch()

    def _pointwise_batch(self):
        # full batch: every item with its exact neighbour distribution as a soft target
        items = torch.arange(self.n_nodes, device=self.device)
        return {"input_ids": items.unsqueeze(1), "labels": self.P, "eval_labels": self.P}

    def _sequence_batch(self):
        B = self._batch_size()
        L = self.config.training_kwargs["context_length"]
        walk = torch.empty(B, L, dtype=torch.long, device=self.device)
        walk[:, 0] = torch.randint(0, self.n_nodes, (B,), generator=self.rng, device=self.device)
        for t in range(1, L):
            walk[:, t] = torch.multinomial(self.P[walk[:, t - 1]], 1, generator=self.rng).squeeze(1)
        start = torch.full((B, 1), self.start_token, dtype=torch.long, device=self.device)
        input_ids = torch.cat([start, walk], dim=1)               # (B, L+1)
        # true next-token dist at each prefix: P[token_j] (start token -> item marginal).
        dist_table = torch.cat([self.P, self.marginal.unsqueeze(0)], dim=0)  # (vocab, C)
        eval_labels = dist_table[input_ids[:, :-1]]               # (B, L, C), aligns with logits[:, :-1]
        return {"input_ids": input_ids, "labels": input_ids.clone(), "eval_labels": eval_labels}

    def compute_metrics(self, eval_pred, return_all=True):
        return _kl_metrics(eval_pred, return_all)


##### Distribution-family dataset (order-1, asymmetric) #####

class DistributionFamilyDataset(Dataset):
    """Order-1 asymmetric task. Items are parametric distributions; the conditional is
    `log p(x|theta)` on an x-grid (exposed as `.conditional`). Factoring it recovers the code =
    natural parameters and decoder = sufficient statistic. Pointwise sampler: an item index ->
    a sampled (binned) outcome, so the learned per-item embedding matches the predicted code.
    (Sequence/in-context inference of theta from samples is the future LLM case, kept separate.)
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(config.random_seed)

        from omegaconf import OmegaConf
        fk = config.get("family_kwargs")
        family_kwargs = OmegaConf.to_container(fk, resolve=True) if fk else {}
        spec = {"family": config.get("family"), **family_kwargs}
        self._L = family_conditional(spec)                # (n_items, n_x) log-conditional

        self.n_items, self.n_outcomes = self._L.shape
        self.arity = 1
        self.item_tokens = torch.arange(self.n_items)
        P = torch.tensor(np.exp(self._L), dtype=torch.float32, device=self.device)
        self.P = P / P.sum(-1, keepdim=True)                      # p(x | theta) over outcome bins

    @property
    def conditional(self):
        return self._L

    @property
    def num_dims(self):
        return self.n_outcomes                                    # number of outcome classes

    @property
    def vocab_size(self):
        return self.n_items                                       # embedding table size (one row per item)

    @property
    def seq_len(self):
        return self.arity + 1

    def __len__(self):
        return 1

    def __getitem__(self, i=None):
        # full batch: every item (distribution) with its exact p(x|theta) as a soft target
        items = torch.arange(self.n_items, device=self.device)
        return {"input_ids": items.unsqueeze(1), "labels": self.P, "eval_labels": self.P}

    def compute_metrics(self, eval_pred, return_all=True):
        return _kl_metrics(eval_pred, return_all)


##### Relational dataset (order >= 2, combination) #####

class RelationalDataset(Dataset):
    """Order-2+ task. The latent structure is an operation c = a*b; the conditional is a tensor
    `T[a, b, ..., c]` (exposed as `.conditional`) read out bilinearly by a shared code. The
    pointwise sampler emits (a, b, ...) -> c so an arity-k readout learns e(a)^T M e(b). arity is
    inferred from the tensor. For graded ops (max/averaging/entailment) factor_tensor gives the
    geometry; for group ops it is degenerate and the trained model selects a single frequency.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(config.random_seed)

        from omegaconf import OmegaConf
        ok = config.get("operation_kwargs")
        operation_kwargs = OmegaConf.to_container(ok, resolve=True) if ok else {}
        spec = {"operation": config.get("operation"), **operation_kwargs}
        T = relation_tensor(spec)                          # (n_items,)*arity + (n_outcomes,)

        self.arity = T.ndim - 1
        self.n_items = T.shape[0]
        self.n_outcomes = T.shape[-1]
        self._T = T
        self.item_tokens = torch.arange(self.n_items)
        # per-input-tuple outcome distribution, flattened over the input tuple for fast lookup
        flat = T.reshape(-1, self.n_outcomes)
        P = torch.tensor(flat, dtype=torch.float32, device=self.device)
        self.P = P / P.sum(-1, keepdim=True)

    @property
    def conditional(self):
        return self._T                                            # tensor -> factor_tensor in the oracle

    @property
    def num_dims(self):
        return self.n_outcomes

    @property
    def vocab_size(self):
        return self.n_items

    @property
    def seq_len(self):
        return self.arity + 1

    def __len__(self):
        return 1

    def __getitem__(self, i=None):
        # full batch: every input tuple with its exact p(c | a, b, ...) as a soft target
        items = torch.arange(self.n_items, device=self.device)
        inputs = torch.cartesian_prod(*([items] * self.arity))   # (n^arity, arity)
        if self.arity == 1:
            inputs = inputs.unsqueeze(1)
        flat = torch.zeros(len(inputs), dtype=torch.long, device=self.device)
        for j in range(self.arity):
            flat = flat * self.n_items + inputs[:, j]             # row-major over the item axes
        dist = self.P[flat]
        return {"input_ids": inputs, "labels": dist, "eval_labels": dist}

    def compute_metrics(self, eval_pred, return_all=True):
        return _kl_metrics(eval_pred, return_all)


##### Hidden-Markov dataset (latent graph walk + emissions) #####

class HMMDataset(Dataset):
    """Latent states walk a transition given by any log-conditional the structure defines (``_L``);
    each state emits an observable token. The network sees only tokens, so the latent geometry lives
    in the belief ``p(s_t | x_{1:t})`` (the forward algorithm), not the token embedding. Every state is
    given the *same* emission shape (uniform per-state distinguishability -> even belief coverage);
    ``emission`` picks how the shape is placed and ``emission_sigma`` its width (small -> sharp/high
    SNR, large -> aliased):

    - ``bump``: a localized kernel over the *latent geometry*, ``Em[k,j] ~ exp(-d(k,j)^2 / 2 sigma^2)``
      with ``d`` the distance in the predicted geometry (in units of nearest-neighbour spacing, so
      ``sigma`` is scale-free). Works for any conditional, not only an adjacency; symbols correspond to
      states, so ``M = n_states``, and on a symmetric geometry (ring, torus, ...) every state is a shift.
    - ``permuted``: one peaked template distribution over the ``M`` symbols, independently permuted
      per state -- identical entropy, different symbols (``n_symbols < n_states`` forces aliasing).

    Sequence-only. ``sigma -> 0`` with ``bump`` recovers the fully-observed graph geometry up to a
    token permutation."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device); self.rng.manual_seed(config.random_seed)

        from omegaconf import OmegaConf
        sk = config.get("structure_kwargs")
        structure_kwargs = OmegaConf.to_container(sk, resolve=True) if sk else {}
        spec = {"structure": config.get("structure"), **structure_kwargs}
        # The latent transition is any log-conditional the structure defines (tree ultrametric,
        # balanced product torus, log_line's log-metric, ...) — not restricted to an adjacency.
        # gamma / cooccurrence ride in structure_kwargs (graph_conditional reads them off the spec).
        self._L = graph_conditional(spec)
        self.n_states = len(self._L)
        self.arity = 1

        T = torch.tensor(np.exp(self._L), dtype=torch.float32, device=self.device)
        self.T = T / T.sum(-1, keepdim=True)
        self.pi = self._stationary(self.T)                        # prior = stationary distribution

        self.n_symbols = int(config.get("n_symbols", self.n_states))
        emission = config.get("emission", "bump")
        sigma = float(config.get("emission_sigma", 1.0))
        if emission == "bump":
            # localized kernel over the latent geometry (any conditional, not just an adjacency);
            # symbols correspond to states (M = n_states)
            E = predicted_geometry(self._L)
            D = np.sqrt(((E[:, None] - E[None]) ** 2).sum(-1))
            nn = np.sort(D + np.eye(len(D)) * D.max(), axis=1)[:, 1].mean()   # mean nearest-neighbour spacing
            D = D / nn                                                        # sigma in neighbour-spacing units
            Em = np.exp(-(D ** 2) / (2 * sigma ** 2)); self.n_symbols = self.n_states
            self.Em = torch.tensor(Em / Em.sum(1, keepdims=True), dtype=torch.float32, device=self.device)
        elif emission == "permuted":
            # one peaked template over the M symbols, independently permuted per state
            j = np.arange(self.n_symbols); template = np.exp(-(j ** 2) / (2 * sigma ** 2))
            g = torch.Generator(device=self.device); g.manual_seed(config.random_seed + 1)
            perms = torch.stack([torch.randperm(self.n_symbols, generator=g, device=self.device)
                                 for _ in range(self.n_states)])
            self.Em = torch.tensor(template / template.sum(), dtype=torch.float32, device=self.device)[perms]
        else:
            raise ValueError(f"unknown emission {emission!r} (expected 'bump' or 'permuted')")

        self.item_tokens = torch.arange(self.n_symbols)           # emission symbols = outcome classes
        self.start_token = self.n_symbols                         # extra token to begin a sequence

    @staticmethod
    def _stationary(T, iters=2000):
        v = torch.full((T.shape[0],), 1.0 / T.shape[0], device=T.device)
        for _ in range(iters):
            v = v @ T
        return v / v.sum()

    @property
    def latent_conditional(self):
        """Log-conditional of the latent transition — factor for the predicted latent geometry."""
        return self._L

    @property
    def num_dims(self):
        return self.n_symbols

    @property
    def vocab_size(self):
        return self.n_symbols + 1                                 # symbols + start token

    @property
    def seq_len(self):
        return self.config.training_kwargs["context_length"] + 1

    def __len__(self):
        return 1

    def _batch_size(self):
        return self.config.training_kwargs["batch_size"]

    def __getitem__(self, i=None):
        B = self._batch_size(); L = self.config.training_kwargs["context_length"]
        s = torch.empty(B, L, dtype=torch.long, device=self.device)   # latent walk: s_1 ~ pi
        s[:, 0] = torch.multinomial(self.pi, B, replacement=True, generator=self.rng)
        for t in range(1, L):
            s[:, t] = torch.multinomial(self.T[s[:, t - 1]], 1, generator=self.rng).squeeze(1)
        x = torch.multinomial(self.Em[s.reshape(-1)], 1, generator=self.rng).reshape(B, L)  # emit

        start = torch.full((B, 1), self.start_token, dtype=torch.long, device=self.device)
        input_ids = torch.cat([start, x], dim=1)                  # (B, L+1)
        # Bayes-optimal next-token distributions p(x_{t+1} | x_{1:t}) via the forward algorithm
        beliefs = self.forward_algorithm(x)                       # (B, L, n_states) = b_1..b_L
        pred_first = (self.pi @ self.Em).expand(B, 1, -1)         # predict x_1 from the prior
        pred_rest = (beliefs[:, :-1] @ self.T) @ self.Em          # predict x_{t+1}, t = 1..L-1
        eval_labels = torch.cat([pred_first, pred_rest], dim=1)   # (B, L, M)
        return {"input_ids": input_ids, "labels": input_ids.clone(), "eval_labels": eval_labels}

    def forward_algorithm(self, obs):
        """Per-position posteriors ``b_t = p(s_t | x_{1:t})`` via the forward algorithm. obs: (B, L)."""
        B, L = obs.shape; b = self.pi.expand(B, -1).clone(); out = []
        for t in range(L):
            if t > 0:
                b = b @ self.T                                    # transition (not before the 1st emission)
            b = b * self.Em[:, obs[:, t]].T                       # emission update
            b = b / b.sum(-1, keepdim=True)
            out.append(b.clone())
        return torch.stack(out, 1)                                # (B, L, n_states)

    def compute_metrics(self, eval_pred, return_all=True):
        return _kl_metrics(eval_pred, return_all)


##### Latent-variable PCFG dataset (latent states walk a *tree* + emit terminals) #####

class LPCFGDataset(Dataset):
    """The tree generalization of the HMM setting: latent states diffuse *down a tree* (branching
    diffusion) rather than along a chain, and each preterminal emits an observable terminal. The
    network sees only the flat terminal string; the tree skeleton, latent states, and constituent
    boundaries are hidden. The Bayes-optimal next-token distribution (``lpcfg_next_token``, the
    left-to-right prefix parser) and the per-position belief over the next emitter's latent state
    are the objects the geometry oracle and analysis compare against — the analogue of the HMM's
    forward algorithm. Sequence-only; identity emissions by default (terminal == leaf state)."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = np.random.default_rng(config.random_seed)
        self._want_eval_labels = True

        from omegaconf import OmegaConf
        gk = config.get("grammar_kwargs")
        grammar_kwargs = OmegaConf.to_container(gk, resolve=True) if gk else {}
        spec = {"grammar": config.get("grammar"), **grammar_kwargs}
        self.g = build_grammar(spec)
        self.m = len(self.g.pi)                                    # latent states
        self.V = self.g.Em.shape[1]                               # terminal symbols
        self.max_len = int(config.get("max_len", 24))             # max terminals per sentence
        self.arity = 1

        self.eos_token = self.V
        self.start_token = self.V + 1
        self.pad_token = self.V + 2
        # outcome classes for the KL metric = terminals + EOS; item_tokens map full-vocab logits down.
        self.item_tokens = torch.arange(self.V + 1)

        # latent transition as a log-conditional the geometry oracle factors (branching diffusion on a
        # graph => the graph's own conditional; identity emission => square, no faithfulness caveat).
        self._L = np.log(self.g.S + 1e-12)

    @property
    def latent_conditional(self):
        return self._L

    @property
    def num_dims(self):
        return self.V + 1                                          # terminals + EOS

    @property
    def vocab_size(self):
        return self.V + 3                                          # terminals + eos + start + pad

    @property
    def seq_len(self):
        return self.max_len + 2                                    # start + max_len terminals + eos

    def __len__(self):
        return 1

    def _batch_size(self):
        return self.config.training_kwargs["batch_size"]

    def _sample_sentence(self):
        """Top-down tree sample; rejection to <= max_len terminals. Returns (terminals, leaf_states)."""
        g = self.g
        while True:
            terms, states = [], []
            ok = [True]

            def rec(h):
                if len(terms) > self.max_len:
                    ok[0] = False; return
                if self.rng.random() < g.p_branch[h]:
                    rec(int(self.rng.choice(self.m, p=g.S[h])))
                    rec(int(self.rng.choice(self.m, p=g.Tt[h])))
                else:
                    states.append(h); terms.append(int(self.rng.choice(self.V, p=g.Em[h])))

            rec(int(self.rng.choice(self.m, p=g.pi)))
            if ok[0] and 1 <= len(terms) <= self.max_len:
                return terms, states

    def __getitem__(self, i=None):
        B = self._batch_size(); Lin = self.seq_len; Tpred = self.max_len + 1
        input_ids = np.full((B, Lin), self.pad_token, dtype=np.int64)
        input_ids[:, 0] = self.start_token
        eval_labels = np.zeros((B, Tpred, self.num_dims), dtype=np.float32)
        beliefs = np.zeros((B, Tpred, self.m), dtype=np.float32)   # for analysis (not used by model)
        states_pad = np.full((B, Tpred), -1, dtype=np.int64)
        for b in range(B):
            terms, states = self._sample_sentence(); k = len(terms)
            input_ids[b, 1:1 + k] = terms
            input_ids[b, 1 + k] = self.eos_token
            states_pad[b, :k] = states
            if self._want_eval_labels:
                dists, bel = lpcfg_next_token(np.array(terms, dtype=int), self.g)
                eval_labels[b, :k + 1] = dists                    # rows 0..k = p(next | x[:t])
                beliefs[b, :k + 1] = bel
        input_ids = torch.from_numpy(input_ids).to(self.device)
        labels = input_ids.clone()
        labels[input_ids == self.pad_token] = -100                # ignore post-EOS padding in the LM loss
        out = {"input_ids": input_ids, "labels": labels,
               "eval_labels": torch.from_numpy(eval_labels).to(self.device)}
        self.last_beliefs = torch.from_numpy(beliefs)             # side-channel for the notebook
        self.last_states = torch.from_numpy(states_pad)
        return out

    def compute_metrics(self, eval_pred, return_all=True):
        return _kl_metrics(eval_pred, return_all)


def _kl_metrics(eval_pred, return_all=True):
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
        # Positions with an all-zero target row are padding (e.g. post-EOS in the PCFG); average the
        # KL over valid predictions only, so trivial padded positions don't dilute the Bayes gap.
        valid = labels.sum(dim=-1) > 0 if labels.ndim == logits.ndim else None
        if valid is not None and not bool(valid.all()):
            loss = loss[valid]
        loss = loss.reshape(loss.shape[0] if loss.ndim else 1, -1)
    results = {"loss": loss.mean().item()}
    if return_all:
        results["all_loss"] = loss.detach().cpu()
    return results


# =====================================================================================
# Generative processes: the conditional each setting exposes for the geometry oracle to
# factor and for the sampler to draw from. Ported from src/notebooks/demo_graphs.ipynb.
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


def hypercube(n):
    """Binary attribute cube: n attributes -> 2^n nodes, neighbors differ in one bit -> analogies."""
    B = np.array([[(v >> b) & 1 for b in range(n)] for v in range(2 ** n)])
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
    """Heat-kernel co-occurrence ``exp(beta * A)``. On a Cartesian-product graph (hypercube, torus,
    grid) this factorizes into a Kronecker product of per-factor heat kernels — ``exp`` turns the
    adjacency's Kronecker *sum* into a *product* — so the factors stay independent, the log-conditional
    is additive over them, and only single-factor directions survive (no interaction harmonics). This
    is the regime in which linear analogies emerge (Korchinski et al.); the walk resolvent, by
    contrast, does not factorize and couples the factors."""
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
    """Branching-diffusion tree -> ultrametric co-occurrence -> nested (hierarchical) geometry,
    over *all* nodes (inner nodes are items too, as in Park et al.). Balanced (depth n, branching
    k) or unbalanced via an explicit `parents` list (``parents[i]`` = parent index, root = -1).
    ``leaves_only=True`` restricts to leaves (the older branching-diffusion setup)."""
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


def log_line_conditional(n=30, lam=2.0):
    """A 1-D line with a logarithmic metric (Weber/Fechner): large magnitudes compress."""
    x = np.log1p(np.arange(n)); dist = np.abs(x[:, None] - x[None])
    return logcond_from_sim(np.exp(-lam * dist))


def _tree_graph(n=2, k=2, parents=None, **_):
    """Tree adjacency over *all* nodes (balanced depth-n branch-k, or an explicit ``parents`` list).
    ``**_`` swallows conditional-only keys (gamma / leaves_only)."""
    return tree_adjacency(tree_parents(n, k, parents))


def _line_graph(n=30, **_):
    """log_line's graph is a plain line — the log-metric lives in its conditional, not its edges."""
    return chain(n)


# One structure = one graph. Every structure builds an adjacency (used by the HMM latent walk and by
# drawing); a structure may additionally define its own conditional (tree's ultrametric, log_line's
# log-distance kernel), which overrides the generic adjacency->cooccurrence derivation.
_Structure = namedtuple("_Structure", ["adjacency", "conditional"], defaults=[None])
_STRUCTURES = {
    "ring": _Structure(ring), "chain": _Structure(chain), "grid": _Structure(grid),
    "sphere": _Structure(sphere), "communities": _Structure(communities),
    "hypercube": _Structure(hypercube), "complete": _Structure(complete),
    "tree": _Structure(_tree_graph, tree_conditional),
    "log_line": _Structure(_line_graph, log_line_conditional),
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


def graph_conditional(spec, gamma=0.8, cooccurrence="diffusion"):
    """Log-conditional for a structure spec ``{'structure': name, **params}``. A structure with its
    own conditional (tree, log_line) uses it directly; otherwise ``cooccurrence`` turns the adjacency
    into a conditional: ``"diffusion"`` (walk co-occurrence ``(I-gamma P)^-1``, integrating multi-step
    context), ``"geodesic"`` (shortest-path kernel ``gamma**d``, self-similar diagonal),
    ``"adjacency"`` (raw one-step neighbours, zero diagonal), or ``"heat"`` (heat kernel
    ``exp(gamma*A)``, which keeps product factors independent → additive log-conditional → analogies) —
    a ``product``/``stack`` builds its
    (Cartesian / block) adjacency and co-occurs that. ``gamma`` and ``cooccurrence`` may also be passed
    as spec keys (overriding the args), so one spec dict configures the task."""
    spec = dict(spec)
    name = spec.pop("structure")
    gamma = spec.pop("gamma", gamma)
    cooccurrence = spec.pop("cooccurrence", cooccurrence)
    spec.pop("view", None)                                    # 'view' is presentation-only (plotting)
    struct = _STRUCTURES.get(name)
    if struct is not None and struct.conditional is not None:
        # pass the task gamma only to conditionals that take it (tree does; log_line uses lam)
        if "gamma" in inspect.signature(struct.conditional).parameters:
            spec.setdefault("gamma", gamma)
        return struct.conditional(**spec)
    return adjacency_conditional(_build_adjacency(name, spec), gamma, cooccurrence)


def adjacency_conditional(A, gamma=0.8, cooccurrence="diffusion"):
    """Turn an adjacency into a log-conditional: ``"diffusion"`` (walk co-occurrence ``(I-gamma P)^-1``),
    ``"geodesic"`` (shortest-path kernel), ``"adjacency"`` (raw one-step neighbours), or ``"heat"``
    (heat kernel ``exp(gamma * A)``, which preserves factor independence on product graphs → additive
    log-conditional → linear analogies). ``gamma`` is the decay for diffusion/geodesic and the
    diffusion time ``beta`` for heat. The graph-walk view of a structure — used directly by the HMM
    latent transition, whose states are the graph's nodes."""
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
    structure has one (tree/log_line included); their special conditionals live in graph_conditional.
    Strips the task-level keys (gamma / cooccurrence)."""
    params = {k: v for k, v in dict(spec).items() if k not in ("structure", "gamma", "cooccurrence")}
    return _build_adjacency(spec["structure"], params)


def stacked_conditional(specs, mode="union", gamma=0.8):
    """Combine several structures on one representation (demo_graphs §2 stack / share / superpose):

      - ``"union"``   : disjoint supports (block-diagonal) -> the geometries stack as a union;
      - ``"product"`` : shared support, independent factors (Cartesian) -> the product geometry;
      - ``"share"``   : aligned relations on the same nodes (summed adjacency) -> one shared code.

    The over-capacity ``superpose`` regime (more co-occurring features than dimensions) is the
    nonlinear ``train_sparse_autoencoder`` in models.py.
    """
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


# ── Distribution families (order-1, asymmetric): items are distributions, L = log p(x|theta) ──

def gaussian1d(n_mu=13, n_sig=6, x_n=200, x_lim=7.0, mu_lim=2.5, sig_lo=0.6, sig_hi=1.8):
    """N(mu, sigma): sufficient statistic (x, x^2) -> Hermite; natural params (mu/s^2, 1/s^2)."""
    mus = np.linspace(-mu_lim, mu_lim, n_mu); sigs = np.linspace(sig_lo, sig_hi, n_sig)
    theta = np.array([(m, s) for s in sigs for m in mus])
    x = np.linspace(-x_lim, x_lim, x_n)
    L = -0.5 * ((x[None] - theta[:, 0:1]) / theta[:, 1:2]) ** 2 - np.log(theta[:, 1:2])
    natural = np.stack([theta[:, 0] / theta[:, 1] ** 2, 1.0 / theta[:, 1] ** 2], 1)
    return L, {"x": x, "params": theta, "natural": natural}


def von_mises(n_mu=20, kappas=(1.5, 3.0, 4.5), x_n=200):
    """p ∝ exp(kappa cos(x-mu)): sufficient statistic (cos x, sin x); natural (k cos mu, k sin mu)."""
    mus = np.linspace(0, 2 * np.pi, n_mu, endpoint=False)
    theta = np.array([(m, k) for k in kappas for m in mus])
    x = np.linspace(0, 2 * np.pi, x_n, endpoint=False)
    L = theta[:, 1:2] * np.cos(x[None] - theta[:, 0:1])
    natural = np.stack([theta[:, 1] * np.cos(theta[:, 0]), theta[:, 1] * np.sin(theta[:, 0])], 1)
    return L, {"x": x, "params": theta, "natural": natural}


def gaussian2d(n_mu=6, sigmas=(0.8, 1.3, 2.0), grid_n=16, xy_lim=5.0, mu_lim=2.0):
    """N(mu, s^2 I): sufficient statistic (x1, x2, |x|^2); natural (mu1/s^2, mu2/s^2, 1/s^2)."""
    mm = np.linspace(-mu_lim, mu_lim, n_mu)
    theta = np.array([(a, b, s) for s in sigmas for b in mm for a in mm])
    a1 = np.linspace(-xy_lim, xy_lim, grid_n)
    X1, X2 = np.meshgrid(a1, a1); xy = np.stack([X1.ravel(), X2.ravel()], 1)
    diff2 = ((xy[None, :, :] - theta[:, None, :2]) ** 2).sum(-1)   # (N, M)
    L = -0.5 * diff2 / theta[:, 2:3] ** 2 - np.log(theta[:, 2:3])
    natural = np.stack([theta[:, 0] / theta[:, 2] ** 2, theta[:, 1] / theta[:, 2] ** 2,
                        1.0 / theta[:, 2] ** 2], 1)
    return L, {"xy": xy, "params": theta, "natural": natural}


def categorical(K=3, n_items=200, alpha=1.0, seed=0):
    """Flat-Dirichlet categoricals (the null case): nothing to compress."""
    P = np.random.default_rng(seed).dirichlet(np.full(K, alpha), size=n_items)
    return np.log(P + 1e-12), {"P": P}


def mixture(x_n=300):
    """Two families (Gaussian + von Mises) on shared support -> the superposition case: the
    decoder is the union of their sufficient statistics."""
    x = np.linspace(-np.pi, np.pi, x_n)
    gm = np.linspace(-2, 2, 9); gs = np.linspace(0.5, 1.1, 3)
    G = np.array([(m, s) for s in gs for m in gm])
    vmm = np.linspace(-np.pi, np.pi, 9, endpoint=False); vk = np.array([1.5, 2.5, 4.0])
    V = np.array([(m, k) for k in vk for m in vmm])
    Lg = -0.5 * ((x[None] - G[:, 0:1]) / G[:, 1:2]) ** 2 - np.log(G[:, 1:2])
    Lv = V[:, 1:2] * np.cos(x[None] - V[:, 0:1])
    return np.vstack([Lg, Lv]), {"x": x, "family": np.array([0] * len(Lg) + [1] * len(Lv))}


_FAMILY_BUILDERS = {"gaussian1d": gaussian1d, "gaussian2d": gaussian2d, "von_mises": von_mises,
                    "categorical": categorical, "mixture": mixture}


def family_build(spec):
    """Build ``(L, info)`` for a family spec ``{'family': name, **params}``."""
    params = {k: v for k, v in dict(spec).items() if k != "family"}
    name = spec["family"]
    if name not in _FAMILY_BUILDERS:
        raise ValueError(f"Unknown family {name!r}; known: {sorted(_FAMILY_BUILDERS)}")
    return _FAMILY_BUILDERS[name](**params)


def family_conditional(spec):
    """The log-conditional ``L`` the geometry oracle factors."""
    return family_build(spec)[0]


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

# =====================================================================================
# Latent-variable PCFG (L-PCFG): the tree generalization of the HMM setting.
#
# One in-terminal S and one preterminal, latent states h in [m]. A node in state h either
# branches (prob p_branch[h]) into a left child (state ~ S[h]) and a right child (state ~ Tt[h]),
# or emits a terminal (prob 1 - p_branch[h]) with distribution Em[h]. The HMM is the right-
# branching special case; here the latent state diffuses *down a tree* instead of along a chain,
# so two terminals are correlated through their lowest common ancestor (tree distance, not string
# distance). Everything compiles to one internal parameterization (pi, p_branch, S, Tt, Em) — any
# named grammar is absorbed into the latent state — so the sampler and the exact oracles never care
# where the matrices came from. See ``.claude/P12-1024.pdf`` (Cohen et al. 2012).
# =====================================================================================

LPCFGGrammar = namedtuple("LPCFGGrammar", ["pi", "p_branch", "S", "Tt", "Em"])
# pi (m,) root-state prior; p_branch (m,) branch-vs-emit prob per state; S/Tt (m,m) left/right child
# state transitions (row-stochastic); Em (m,V) emission (row-stochastic; identity => terminals ARE
# leaf states). Subcritical (mean offspring 2*p_branch < 1) so trees terminate a.s.


def branching_diffusion_grammar(spec):
    """Branching diffusion on a graph: latent states = graph nodes, each tree edge is one diffusion
    step on the graph (the same row-normalized ``exp(graph_conditional)`` the HMM uses as its chain
    transition, now wired into a tree). Identity emission by default (terminal == leaf state)."""
    spec = dict(spec)
    p_branch = spec.pop("p_branch", 0.4)
    L = graph_conditional({"structure": spec.pop("structure"), **spec})
    m = len(L)
    P = np.exp(L); S = P / P.sum(1, keepdims=True)
    pi = _stationary_dist(S)
    return LPCFGGrammar(pi=pi, p_branch=np.full(m, float(p_branch)), S=S, Tt=S.copy(), Em=np.eye(m))


def _stationary_dist(P, iters=2000):
    v = np.full(P.shape[0], 1.0 / P.shape[0])
    for _ in range(iters):
        v = v @ P
    return v / v.sum()


_GRAMMAR_BUILDERS = {"branching_diffusion": branching_diffusion_grammar}


def build_grammar(spec):
    """Build an ``LPCFGGrammar`` from a spec ``{'grammar': name, **params}``."""
    spec = dict(spec)
    name = spec.pop("grammar")
    if name not in _GRAMMAR_BUILDERS:
        raise ValueError(f"Unknown grammar {name!r}; known: {sorted(_GRAMMAR_BUILDERS)}")
    return _GRAMMAR_BUILDERS[name](spec)


# ── Exact Bayes oracle: inside chart + left-to-right prefix / next-token (Stolcke-style) ──

def lpcfg_inside_chart(x, g: LPCFGGrammar):
    """Complete inside probabilities ``In[i,j][h] = p(state-h subtree yields x_i..x_j)`` for a
    terminal sequence ``x``. Standard CKY over the latent-annotated binary grammar."""
    e = 1.0 - g.p_branch
    k = len(x)
    In = {}
    for i in range(k):
        In[(i, i)] = e * g.Em[:, x[i]]
    for length in range(2, k + 1):
        for i in range(0, k - length + 1):
            j = i + length - 1
            acc = np.zeros(len(g.pi))
            for split in range(i, j):
                acc += (g.S @ In[(i, split)]) * (g.Tt @ In[(split + 1, j)])
            In[(i, j)] = g.p_branch * acc
    return In


def lpcfg_prefix_chart(x, g: LPCFGGrammar, In):
    """Prefix-inside ``PIn[i,j][h] = p(state-h subtree yields x_i..x_j as a *prefix* of its full
    yield)``. ``R = (I - diag(p_branch) S)^-1`` resolves the left-spine descent: the prefix may end
    arbitrarily deep in the left child (complete OR proper prefix, both already inside ``PIn``), with
    every right sibling unconstrained. The split sum (k < j) handles the prefix continuing past a
    *completed* left child into the right child."""
    m = len(g.pi); k = len(x); pb = g.p_branch
    R = np.linalg.inv(np.eye(m) - (pb[:, None] * g.S))
    PIn = {}
    for j in range(k):
        for i in range(j, -1, -1):
            acc = np.zeros(m)
            for split in range(i, j):
                acc += (g.S @ In[(i, split)]) * (g.Tt @ PIn[(split + 1, j)])
            rhs = (In[(i, i)] if i == j else 0.0) + pb * acc
            PIn[(i, j)] = R @ rhs
    return PIn


def lpcfg_next_token(x, g: LPCFGGrammar):
    """Exact left-to-right Bayes predictions for a terminal sequence ``x`` (length k).

    Returns ``(dists, beliefs)``:
      - ``dists`` (k+1, V+1): row t = ``p(next | x[:t])`` over the V terminals plus a final EOS
        column, for every prefix length t = 0..k (t=0 predicts the first terminal from the prior).
      - ``beliefs`` (k+1, m): row t = ``p(state of the node that emits x_t | x[:t])`` — the fixed-
        dimensional belief that lands on the latent geometry (the analogue of the HMM forward belief).
    """
    m = len(g.pi); V = g.Em.shape[1]; k = len(x); pb = g.p_branch; e = 1.0 - pb
    R = np.linalg.inv(np.eye(m) - (pb[:, None] * g.S))
    In = lpcfg_inside_chart(x, g)
    PIn = lpcfg_prefix_chart(x, g, In) if k else {}

    def _normalize(v):                                              # a distribution, or zeros if the
        s = v.sum()                                                # prefix cannot continue (next is EOS)
        return v / s if s > 0 else v
    dists = np.zeros((k + 1, V + 1))
    beliefs = np.zeros((k + 1, m))
    # t = 0: first-emit state distribution from the prior (descend the left spine to the first emit).
    b0 = g.pi @ (R * e[None, :])                                     # (m,) state of the first emitter
    beliefs[0] = _normalize(b0)
    dists[0, :V] = beliefs[0] @ g.Em                                 # no EOS: sentences are non-empty
    for t in range(1, k + 1):
        Z = g.pi @ PIn[(0, t - 1)]                                   # prefix probability p(x[:t])
        eos = g.pi @ In[(0, t - 1)]                                  # prefix is a complete sentence
        o = _lpcfg_boundary(x, t, g, In, R)                         # prefix-outside boundary vector
        state = o * e                                                # state of the node emitting x_t
        dists[t, :V] = (state @ g.Em) / Z
        dists[t, V] = eos / Z
        beliefs[t] = _normalize(state)                              # next-emitter posterior (0 if terminal)
    return dists, beliefs


def _lpcfg_boundary(x, t, g: LPCFGGrammar, In, R):
    """Prefix-outside boundary vector ``o`` at position t: ``p(x[:t] w) = sum_h o[h] e[h] Em[h,w]``.
    Back-substitutes the linear map ``P_i`` sending the appended symbol's base vector to
    ``PIn_ext[i,t]`` (the prefix chart of ``x[:t]`` with a hypothetical symbol at index t)."""
    m = len(g.pi); diag_pb = g.p_branch[:, None]
    P = {t: R}                                                       # PIn_ext[t,t] = R @ u_w
    for i in range(t - 1, -1, -1):
        acc = np.zeros((m, m))
        for split in range(i, t):
            acc += (g.S @ In[(i, split)])[:, None] * (g.Tt @ P[split + 1])
        P[i] = R @ (diag_pb * acc)
    return g.pi @ P[0]


# ── Classic context-free grammars, encoded in the binary-leaf latent-state form ──
# Both use a single skeletal rule (branch S->S S, or emit a leaf); the counter / stack that a
# regular chain cannot track is pushed into the latent state h. Transitions are deterministic
# (one-hot rows, p_branch in {0,1}); the only randomness is the root prior (sentence length / bracket
# types) and, for Dyck, the choice of the next nested bracket type. Bounded depth => finite trees.

def _grammar_from_rules(states, leaves, rules, root_prior, V, emit):
    """Assemble an ``LPCFGGrammar`` from named states. ``leaves`` maps a leaf-state name to its
    terminal id (p_branch=0); ``rules`` maps an internal-state name to ``(left_name, right_name)``
    or ``{left_name: p}`` distributions ``(left_dist, right_name_or_dist)`` (p_branch=1);
    ``root_prior`` maps a state name to its prior mass; ``emit`` is unused padding for internals."""
    idx = {name: i for i, name in enumerate(states)}
    m = len(states)
    pi = np.zeros(m); pb = np.zeros(m)
    S = np.zeros((m, m)); Tt = np.zeros((m, m)); Em = np.full((m, V), 1.0 / V)
    for name, term in leaves.items():
        Em[idx[name]] = 0.0; Em[idx[name], term] = 1.0                # deterministic emission
    for name, (left, right) in rules.items():
        h = idx[name]; pb[h] = 1.0
        for child, dist in ((S, left), (Tt, right)):
            if isinstance(dist, dict):
                for nm, p in dist.items():
                    child[h, idx[nm]] += p
            else:
                child[h, idx[dist]] = 1.0
    for name, p in root_prior.items():
        pi[idx[name]] = p
    pi /= pi.sum()
    return LPCFGGrammar(pi=pi, p_branch=pb, S=S, Tt=Tt, Em=Em)


def anbn_grammar(spec):
    """a^n b^n (n ~ uniform on 1..D). Binary parse: S_n -> A (M_{n-1}), M_k -> S_k B, S_1 -> A B.
    The prefix's sufficient statistic is the count of unmatched a's -> a *line* (chain) geometry
    over depth, which no finite-state chain can represent."""
    D = int(dict(spec).get("depth", 8))
    a, b = 0, 1
    states = ["A", "B"] + [f"S{n}" for n in range(1, D + 1)] + [f"M{k}" for k in range(1, D)]
    leaves = {"A": a, "B": b}
    rules = {"S1": ("A", "B")}
    for n in range(2, D + 1):
        rules[f"S{n}"] = ("A", f"M{n - 1}")
    for k in range(1, D):
        rules[f"M{k}"] = (f"S{k}", "B")
    root = {f"S{n}": 1.0 for n in range(1, D + 1)}                    # uniform over lengths
    return _grammar_from_rules(states, leaves, rules, root, V=2, emit=None)


def dyck_grammar(spec):
    """Bounded-depth nested Dyck-k: strings ``(_i1 (_i2 ... )_i2 )_i1`` up to depth D, each level's
    bracket type chosen uniformly. The prefix's sufficient statistic is the *stack* of open bracket
    types -- a node of the depth-D, branching-k tree -> the nested (ultrametric) tree geometry
    (``tree_conditional(D, k)``). Terminals: open_i = i, close_i = k+i (V = 2k)."""
    sp = dict(spec); k = int(sp.get("k", 2)); D = int(sp.get("depth", 4))
    states = [f"Op{i}" for i in range(k)] + [f"Cl{i}" for i in range(k)]
    leaves = {f"Op{i}": i for i in range(k)} | {f"Cl{i}": k + i for i in range(k)}
    rules = {}

    def add(stack):                                                  # stack: tuple of bracket types
        d = len(stack); last = stack[-1]; g = "G" + "".join(map(str, stack))
        if d == D:
            rules[g] = (f"Op{last}", f"Cl{last}")                    # innermost: bare pair
            states.append(g); return
        body = "B" + "".join(map(str, stack))
        rules[g] = (f"Op{last}", body)
        rules[body] = ({f"G{''.join(map(str, stack + (j,)))}": 1.0 / k for j in range(k)}, f"Cl{last}")
        states.extend([g, body])
        for j in range(k):
            add(stack + (j,))

    for i in range(k):
        add((i,))
    root = {f"G{i}": 1.0 for i in range(k)}                          # uniform over first bracket type
    return _grammar_from_rules(states, leaves, rules, root, V=2 * k, emit=None)


_GRAMMAR_BUILDERS.update({"anbn": anbn_grammar, "dyck": dyck_grammar})


def antbn_grammar(spec):
    """a^n T b^n (n ~ uniform on 1..D): a^n b^n with an explicit trigger terminal T marking the
    switch from the opening run to the closing run. S_n -> A (M_{n-1}), M_k -> S_k B, S_0 -> T.
    Same unmatched-a counter as a^n b^n, but T makes the phase boundary a single observable event."""
    D = int(dict(spec).get("depth", 8))
    a, b, trig = 0, 1, 2
    states = ["A", "B"] + [f"S{n}" for n in range(0, D + 1)] + [f"M{k}" for k in range(0, D)]
    leaves = {"A": a, "B": b, "S0": trig}                            # S0 is the trigger leaf
    rules = {}
    for n in range(1, D + 1):
        rules[f"S{n}"] = ("A", f"M{n - 1}")
    for k in range(0, D):
        rules[f"M{k}"] = (f"S{k}", "B")
    root = {f"S{n}": 1.0 for n in range(1, D + 1)}                   # uniform over n (>=1)
    return _grammar_from_rules(states, leaves, rules, root, V=3, emit=None)


_GRAMMAR_BUILDERS["antbn"] = antbn_grammar


def prefix_arith_grammar(spec):
    """Prefix (Polish-notation) arithmetic — a tiny programming language of S-expressions:
    ``E -> op E E`` (an operator applied to two sub-expressions) with prob ``p``, else ``E -> num``.
    Strings like ``+ * 1 2 3`` (= ``(1*2)+3``). Binary-leaf form: E branches into (Op-leaf, R) and
    R branches into (E, E); E otherwise emits a number. The left-to-right prefix belief is the
    *arity debt* — the number of sub-expressions still owed — which does a random walk (op: +1,
    num: -1) absorbed at 0 (the program is complete). Terminals: operators 0..n_op-1, numbers
    n_op..n_op+n_num-1 (V = n_op + n_num)."""
    sp = dict(spec)
    n_op = int(sp.get("n_op", 2)); n_num = int(sp.get("n_num", 3)); p = float(sp.get("p", 0.4))
    V = n_op + n_num
    E, Op, R = 0, 1, 2                                               # latent states
    pi = np.array([1.0, 0.0, 0.0])                                   # start at an expression E
    p_branch = np.array([p, 0.0, 1.0])                              # E: apply-op w.p. p else num; R: always
    S = np.zeros((3, 3)); Tt = np.zeros((3, 3))
    S[E, Op] = 1.0; Tt[E, R] = 1.0                                   # E -> Op R   (operator, then the rest)
    S[R, E] = 1.0; Tt[R, E] = 1.0                                    # R -> E E    (two sub-expressions)
    Em = np.full((3, V), 1.0 / V)
    Em[E] = 0.0; Em[E, n_op:] = 1.0 / n_num                          # E emits a number
    Em[Op] = 0.0; Em[Op, :n_op] = 1.0 / n_op                         # Op emits an operator
    return LPCFGGrammar(pi=pi, p_branch=p_branch, S=S, Tt=Tt, Em=Em)


_GRAMMAR_BUILDERS["prefix_arith"] = prefix_arith_grammar
