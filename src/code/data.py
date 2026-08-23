import inspect
import itertools
from collections import namedtuple
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .config import Config

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
    if config.setting == "dirichlet-categorical":
        return DirichletCategoricalDataset(config)
    if config.setting == "exchangeable-family":
        return ExchangeableFamilyDataset(config)
    if config.setting == "pcfg":
        return PCFGDataset(config)
    if config.setting == "feature_cube":
        return FeatureCubeDataset(config)
    raise ValueError(f"Unsupported setting: {config.setting}")


##### Graph co-occurrence dataset (order-1, symmetric) #####

class GraphCooccurrenceDataset(Dataset):
    """Graph nodes as tokens; ``.conditional`` is the diffusion log-conditional
    ``log p(token | context)`` (n, n), and its row-normalized exp is the sampling transition.
    ``sample_mode``: "pointwise" (item -> neighbor pairs) or "sequence" (Markov walks)."""

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
        return {"input_ids": items.unsqueeze(1), "labels": self.P}

    def _sequence_batch(self):
        B = self._batch_size()
        L = self.config.training_kwargs["context_length"]
        walk = torch.empty(B, L, dtype=torch.long, device=self.device)
        walk[:, 0] = torch.randint(0, self.n_nodes, (B,), generator=self.rng, device=self.device)
        for t in range(1, L):
            walk[:, t] = torch.multinomial(self.P[walk[:, t - 1]], 1, generator=self.rng).squeeze(1)
        start = torch.full((B, 1), self.start_token, dtype=torch.long, device=self.device)
        input_ids = torch.cat([start, walk], dim=1)               # (B, L+1)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    def optimal_predictions(self, input_ids):
        """Bayes next-token distribution at every prefix: P[token] (start -> item marginal).
        Pointwise (B, 1) -> (B, C); sequence (B, L+1) -> (B, L, C), aligned with logits[:, :-1]."""
        if self.sample_mode == "pointwise":
            return self.P[input_ids[:, 0]]
        dist_table = torch.cat([self.P, self.marginal.unsqueeze(0)], dim=0)  # (vocab, C)
        return dist_table[input_ids[:, :-1]]

    def compute_metrics(self, eval_pred, return_all=True):
        return _kl_metrics(eval_pred, return_all)


##### Feature-cube dataset (hypercube heat kernel + sparse item measure) #####

def feature_cube(p, gamma, w_decay=1.0):
    """Hypercube product kernel with per-feature couplings w_i = w0 * w_decay^i
    (w0 = half log coth gamma; importance) plus a sparse item measure (feature i active with
    probability p[i]; frequency), so log p(x'|x) = const + Z @ Lf exactly.
    Returns (logP (2^m, 2^m), pi (2^m,), Z (2^m, m) in ±1, Lf (m, 2^m))."""
    p = np.asarray(p, float)
    m = len(p)
    Z = 2.0 * cube_codes(m) - 1.0
    w = 0.5 * np.log(1 / np.tanh(gamma)) * w_decay ** np.arange(m)
    f = 1.0 / (1.0 + np.exp(2.0 * w))                          # per-feature flip probability
    logP = (Z * w) @ Z.T + 0.5 * np.log(f * (1 - f)).sum()
    Lf = w[:, None] * Z.T
    on = Z > 0
    pi = (p ** on * (1 - p) ** ~on).prod(1)
    return logP, pi, Z, Lf



class FeatureCubeDataset(Dataset):
    """Hypercube heat-kernel task under a sparse item measure (feature i active with probability
    sparsity * p_decay^i). One token per item; exposes ``conditional``, the measure ``pi``, and
    the ±1 codes ``Z`` / directions ``Lf``. ``sample_mode``: "pointwise" or "sequence"."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device); self.rng.manual_seed(config.random_seed)

        m = config.get("n_features")
        p = config.get("sparsity") * config.get("p_decay") ** np.arange(m)
        logP, self.pi, self.Z, self.Lf = feature_cube(p, config.get("gamma"),
                                                      config.get("w_decay", 1.0))
        self._L = logP
        self.n_items = len(logP)
        self.arity = 1
        self.sample_mode = config.get("sample_mode", "pointwise")
        self.item_tokens = torch.arange(self.n_items)
        self.start_token = self.n_items                           # sequence mode only
        self.P = torch.tensor(np.exp(logP), dtype=torch.float32, device=self.device)
        self._pi = torch.tensor(self.pi, dtype=torch.float32, device=self.device)
        self._dist_table = torch.cat([self.P, self._pi.unsqueeze(0)])  # start -> the item measure

    @property
    def conditional(self):
        return self._L

    @property
    def num_dims(self):
        return self.n_items

    @property
    def vocab_size(self):
        return self.n_items + 1 if self.sample_mode == "sequence" else self.n_items

    @property
    def seq_len(self):
        if self.sample_mode == "sequence":
            return self.config.training_kwargs["context_length"] + 1
        return self.arity + 1

    def __len__(self):
        return 1

    def __getitem__(self, i=None):
        B = self.config.training_kwargs["batch_size"]
        if self.sample_mode == "sequence":
            return self._sequence_batch(B)
        items = torch.multinomial(self._pi, B, replacement=True, generator=self.rng)
        return {"input_ids": items.unsqueeze(1), "labels": self.P[items]}

    def _sequence_batch(self, B):
        L = self.config.training_kwargs["context_length"]
        walk = torch.empty(B, L, dtype=torch.long, device=self.device)
        walk[:, 0] = torch.multinomial(self._pi, B, replacement=True, generator=self.rng)
        for t in range(1, L):
            walk[:, t] = torch.multinomial(self.P[walk[:, t - 1]], 1, generator=self.rng).squeeze(1)
        start = torch.full((B, 1), self.start_token, dtype=torch.long, device=self.device)
        input_ids = torch.cat([start, walk], dim=1)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    def optimal_predictions(self, input_ids):
        """Bayes next-token distribution at every prefix: P[token] (start -> the item measure)."""
        if self.sample_mode == "pointwise":
            return self.P[input_ids[:, 0]]
        return self._dist_table[input_ids[:, :-1]]

    def compute_metrics(self, eval_pred, return_all=True):
        return _kl_metrics(eval_pred, return_all)


##### Distribution-family dataset (order-1, asymmetric) #####

class DistributionFamilyDataset(Dataset):
    """Items are parametric distributions; ``.conditional`` is ``log p(x|theta)`` on an x-grid,
    (n_items, n_x). Pointwise sampler: every item with its exact binned ``p(x|theta)`` as a
    soft target."""

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
        return {"input_ids": items.unsqueeze(1), "labels": self.P}

    def compute_metrics(self, eval_pred, return_all=True):
        return _kl_metrics(eval_pred, return_all)


##### Relational dataset (order >= 2, combination) #####

class RelationalDataset(Dataset):
    """Operation task c = a*b: ``.conditional`` is the tensor ``T[a, b, ..., c]``, with arity
    inferred from its shape. Pointwise sampler: every input tuple (a, b, ...) with its exact
    ``p(c | a, b, ...)`` as a soft target."""

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
        return {"input_ids": inputs, "labels": dist}

    def compute_metrics(self, eval_pred, return_all=True):
        return _kl_metrics(eval_pred, return_all)


##### Hidden-Markov dataset (latent graph walk + emissions) #####

class HMMDataset(Dataset):
    """Latent states walk the structure's graph; each state emits an observable token and the
    network sees only tokens. ``emission``: "bump" (Gaussian in hop distance, symbols = states) or
    "permuted" (a peaked template permuted per state); ``emission_sigma`` sets its width. Sequence-only."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device); self.rng.manual_seed(config.random_seed)

        from omegaconf import OmegaConf
        sk = config.get("structure_kwargs")
        structure_kwargs = OmegaConf.to_container(sk, resolve=True) if sk else {}
        spec = {"structure": config.get("structure"), **structure_kwargs}
        # The HMM treats every structure as its adjacency graph: the latent walks the graph's nodes
        # (tree included — all nodes are states, not only leaves), so the transition and the bump
        # emission share one geometry. gamma / cooccurrence are required structure_kwargs.
        self._L = adjacency_conditional(structure_adjacency(spec), spec["gamma"], spec["cooccurrence"])
        self.n_states = len(self._L)
        self.arity = 1
        self.labels = config.get("labels", "tokens")

        T = torch.tensor(np.exp(self._L), dtype=torch.float64, device=self.device)
        self.T = T / T.sum(-1, keepdim=True)
        self.pi = self._stationary(self.T)                        # prior = stationary distribution

        self.n_symbols = int(config.get("n_symbols", self.n_states))
        emission = config.get("emission", "bump")
        sigma = float(config.get("emission_sigma", 1.0))
        if emission == "bump":
            # Gaussian in hop distance on the structure's graph; symbols correspond to states
            # (M = n_states), so sigma reads directly as 'how many hops does an emission alias'
            D = structure_distance(spec)
            Em = np.exp(-(D ** 2) / (2 * sigma ** 2)); self.n_symbols = self.n_states
            self.Em = torch.tensor(Em / Em.sum(1, keepdims=True), dtype=torch.float64, device=self.device)
        elif emission == "permuted":
            # one peaked template over the M symbols, independently permuted per state
            j = np.arange(self.n_symbols); template = np.exp(-(j ** 2) / (2 * sigma ** 2))
            g = torch.Generator(device=self.device); g.manual_seed(config.random_seed + 1)
            perms = torch.stack([torch.randperm(self.n_symbols, generator=g, device=self.device)
                                 for _ in range(self.n_states)])
            self.Em = torch.tensor(template / template.sum(), dtype=torch.float64, device=self.device)[perms]
        else:
            raise ValueError(f"unknown emission {emission!r} (expected 'bump' or 'permuted')")

        self.start_token = self.n_symbols                         # extra token to begin a sequence
        offset = self.n_symbols + 1 if self.labels == "latent" else 0
        self.item_tokens = offset + torch.arange(self.num_dims)

    @staticmethod
    def _stationary(T, iters=2000):
        v = torch.full((T.shape[0],), 1.0 / T.shape[0], dtype=T.dtype, device=T.device)
        for _ in range(iters):
            v = v @ T
        return v / v.sum()

    @property
    def latent_conditional(self):
        """Log-conditional of the latent transition — factor for the predicted latent geometry."""
        return self._L

    @property
    def num_dims(self):
        return self.n_states if self.labels == "latent" else self.n_symbols

    @property
    def vocab_size(self):
        return self.n_symbols + 1 + (self.n_states if self.labels == "latent" else 0)

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
        labels = input_ids.clone()
        if self.labels == "latent":
            labels[:, 1:] = s + self.n_symbols + 1
        return {"input_ids": input_ids, "labels": labels}

    def optimal_predictions(self, input_ids):
        """Bayes next-token or next-latent distributions via the forward algorithm."""
        x = input_ids[:, 1:]
        beliefs = self.forward_algorithm(x)                       # (B, L, n_states) = b_1..b_L
        if self.labels == "latent":
            prior = self.pi.expand(len(x), 1, -1)                 # predict s_1 before seeing x_1
            return torch.cat([prior, beliefs[:, :-1] @ self.T], dim=1).float()
        pred_first = (self.pi @ self.Em).expand(len(x), 1, -1)    # predict x_1 from the prior
        pred_rest = (beliefs[:, :-1] @ self.T) @ self.Em          # predict x_{t+1}, t = 1..L-1
        return torch.cat([pred_first, pred_rest], dim=1).float()

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


##### Exchangeable datasets (in-context inference over observables) #####

class ExchangeableDataset(Dataset):
    """Each sequence draws emission probs ``theta`` from a prior and emits iid tokens over ``K``
    outcome classes; the Bayes next-token distribution is the posterior predictive. Sequence-only.
    Subclasses supply ``K``, ``sample_emission``, ``prior_predictive``, ``posterior_predictive``."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device); self.rng.manual_seed(config.random_seed)
        self.arity = 1
        self.labels = config.get("labels", "tokens")

    @property
    def num_dims(self):
        return self.M if self.labels == "latent" else self.K

    @property
    def vocab_size(self):
        return self.K + 1 + (self.M if self.labels == "latent" else 0)

    @property
    def item_tokens(self):
        offset = self.K + 1 if self.labels == "latent" else 0
        return offset + torch.arange(self.num_dims)

    @property
    def start_token(self):
        return self.K                                             # extra token to begin a sequence

    @property
    def seq_len(self):
        return self.config.training_kwargs["context_length"] + 1

    def __len__(self):
        return 1

    def __getitem__(self, i=None):
        B, L = self.config.training_kwargs["batch_size"], self.config.training_kwargs["context_length"]
        theta, latent = self.sample_emission(B, return_labels=True)
        x = torch.multinomial(theta, L, replacement=True, generator=self.rng)   # (B, L) iid | theta

        start = torch.full((B, 1), self.start_token, dtype=torch.long, device=self.device)
        input_ids = torch.cat([start, x], dim=1)                  # (B, L+1)
        labels = input_ids.clone()
        if self.labels == "latent":
            labels[:, 1:] = latent[:, None] + self.K + 1
        return {"input_ids": input_ids, "labels": labels}

    def optimal_predictions(self, input_ids):
        """Bayes next-token or latent distributions at each prefix, returned as float32."""
        x = input_ids[:, 1:]
        if self.labels == "latent":
            prior = torch.full((len(x), 1, self.M), 1.0 / self.M, dtype=torch.float64,
                               device=self.device)
            return torch.cat([prior, self.posterior_weights(x)[:, :-1]], dim=1).float()
        pred = self.posterior_predictive(x)                       # p(x_{t+1} | x_{1:t}), t = 1..L
        prior = self.prior_predictive.expand(len(x), 1, -1)
        return torch.cat([prior, pred[:, :-1]], dim=1).float()

    def logit_code(self, obs):
        """Centered log posterior predictive (float64): the predicted in-context code. (B,L) -> (B,L,K)."""
        g = self.posterior_predictive(obs).double().log()
        return g - g.mean(-1, keepdim=True)

    def compute_metrics(self, eval_pred, return_all=True):
        return _kl_metrics(eval_pred, return_all)


class DirichletCategoricalDataset(ExchangeableDataset):
    """Exchangeable task with the conjugate prior ``theta ~ Dir(alpha)``: the posterior predictive
    has the closed form ``(alpha + counts_t)/(alpha_0 + t)``."""

    def __init__(self, config: Config):
        super().__init__(config)
        # torch.distributions.Dirichlet takes no generator; numpy keeps theta draws reproducible
        self.np_rng = np.random.default_rng(config.random_seed)
        self.K = int(config.get("n_categories"))
        self.alpha = torch.full((self.K,), float(config.get("alpha")), dtype=torch.float64,
                                device=self.device)
        self._alpha_np = np.full(self.K, float(config.get("alpha")))

    @property
    def prior_predictive(self):
        return self.alpha / self.alpha.sum()

    def sample_emission(self, B, return_labels=False):
        theta = torch.tensor(self.np_rng.dirichlet(self._alpha_np, size=B),
                             dtype=torch.float64, device=self.device)
        return (theta, None) if return_labels else theta

    def counts(self, obs):
        """Cumulative per-category counts of ``x_{1:t}`` at every position t — the belief state
        (with ``alpha``, the posterior Dirichlet parameters). obs: (B, L) -> (B, L, K)."""
        return torch.cumsum(nn.functional.one_hot(obs, self.K).double(), dim=1)

    def posterior_predictive(self, obs):
        """``(alpha + counts_t) / (alpha_0 + t)``: the posterior predictive after ``x_{1:t}``."""
        n = self.alpha + self.counts(obs)
        return n / n.sum(-1, keepdim=True)


class ExchangeableFamilyDataset(ExchangeableDataset):
    """Exchangeable task over any distribution family: ``theta`` is an atom from the family's
    pool (uniform prior), tokens are the binned support, and the posterior predictive is exact
    Bayes filtering over the atoms. Generalizes the Dirichlet task to arbitrary priors."""

    def __init__(self, config: Config):
        super().__init__(config)
        from omegaconf import OmegaConf
        fk = config.get("family_kwargs")
        family_kwargs = OmegaConf.to_container(fk, resolve=True) if fk else {}
        spec = {"family": config.get("family"), **family_kwargs}
        L, self.info = family_build(spec)                         # (M, K) log p(x|theta) on the grid
        # normalize in log space (float64): exp-then-log would quantize or underflow deep tails
        from scipy.special import logsumexp
        logP = L - logsumexp(L, axis=1, keepdims=True)
        self._logatoms = logP
        self.log_atoms = torch.tensor(logP, dtype=torch.float64, device=self.device)
        self.log_atoms_T = self.log_atoms.T.contiguous()          # (K, M) for per-step gathers
        self.atoms = self.log_atoms.exp()                         # (M, K) emission probs per atom
        self.M, self.K = self.atoms.shape
        self._prior_predictive = self.atoms.mean(0)

    @property
    def atom_logcond(self):
        """Normalized ``log p(x|theta)`` (float64) — the matrix the geometry oracle factors.
        Rows are atoms (not tokens), so this is deliberately not ``conditional``: the in-context
        code lives in hidden states over contexts, not item embeddings."""
        return self._logatoms

    @property
    def prior_predictive(self):
        return self._prior_predictive

    def sample_emission(self, B, return_labels=False):
        idx = torch.randint(self.M, (B,), device=self.device, generator=self.rng)
        return (self.atoms[idx], idx) if return_labels else self.atoms[idx]

    def posterior_weights(self, obs):
        """Posterior over atoms after ``x_{1:t}`` at every position t — the belief state.
        obs: (B, L) -> (B, L, M)."""
        loglik = torch.cumsum(self.log_atoms_T[obs], dim=1)       # (B, L, M) cumulative log p(x_{1:t}|m)
        return torch.softmax(loglik, dim=-1)                      # uniform prior over atoms

    def posterior_predictive(self, obs):
        """``sum_m w_t(m) theta_m``: the posterior predictive after ``x_{1:t}``."""
        return self.posterior_weights(obs) @ self.atoms

    def logit_code(self, obs):
        """Centered log posterior predictive via logsumexp (float64), so tail bins beyond the
        float32 exp range stay finite. (B, L) -> (B, L, K)."""
        loglik = torch.cumsum(self.log_atoms_T[obs], dim=1)                   # (B, L, M) float64
        logw = loglik - loglik.logsumexp(-1, keepdim=True)                    # log posterior weights
        g = (logw.unsqueeze(-1) + self.log_atoms).logsumexp(2)                # log mixed predictive
        return g - g.mean(-1, keepdim=True)


##### PCFG dataset (latent states walk a *tree* + emit terminals) #####

class PCFGDataset(Dataset):
    """Latent states branch down a tree and leaves emit terminals; the network sees only the flat
    terminal string. Bayes next-token distributions come from ``pcfg_next_token``. Sequence-only;
    identity emissions by default (terminal == leaf state)."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = np.random.default_rng(config.random_seed)

        from omegaconf import OmegaConf
        gk = config.get("grammar_kwargs")
        grammar_kwargs = OmegaConf.to_container(gk, resolve=True) if gk else {}
        spec = {"grammar": config.get("grammar"), **grammar_kwargs}
        self.g = build_grammar(spec)
        self.m = len(self.g.pi)                                    # latent states
        self.V = self.g.Em.shape[1]                               # terminal symbols
        self.max_len = int(config.cfg.max_len)                    # max terminals per sentence (required)
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
        B = self._batch_size(); Lin = self.seq_len
        input_ids = np.full((B, Lin), self.pad_token, dtype=np.int64)
        input_ids[:, 0] = self.start_token
        for b in range(B):
            terms, _ = self._sample_sentence(); k = len(terms)
            input_ids[b, 1:1 + k] = terms
            input_ids[b, 1 + k] = self.eos_token
        input_ids = torch.from_numpy(input_ids).to(self.device)
        labels = input_ids.clone()
        labels[input_ids == self.pad_token] = -100                # ignore post-EOS padding in the LM loss
        return {"input_ids": input_ids, "labels": labels}

    def optimal_predictions(self, input_ids):
        """Bayes next-token distributions p(next | x[:t]) via the inside/prefix charts, per
        sentence read off the tokens (rows past EOS are zero). (B, Lin) -> (B, max_len+1, V+1)."""
        ids = input_ids.cpu().numpy()
        B, Tpred = len(ids), self.max_len + 1
        eval_labels = np.zeros((B, Tpred, self.num_dims), dtype=np.float32)
        for b in range(B):
            terms = ids[b, 1:]
            terms = terms[:int(np.argmax(terms == self.eos_token))]   # tokens before EOS
            eval_labels[b, :len(terms) + 1] = pcfg_next_token(terms.astype(int), self.g)[0]
        return torch.from_numpy(eval_labels).to(input_ids.device)

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


# One structure = one graph. Every structure builds an adjacency (used by the HMM latent walk and by
# drawing); a structure may additionally define its own conditional (tree's ultrametric), which
# overrides the generic adjacency->cooccurrence derivation.
_Structure = namedtuple("_Structure", ["adjacency", "conditional"], defaults=[None])
_STRUCTURES = {
    "ring": _Structure(ring), "chain": _Structure(chain), "grid": _Structure(grid),
    "sphere": _Structure(sphere), "communities": _Structure(communities),
    "hypercube": _Structure(hypercube), "complete": _Structure(complete),
    "tree": _Structure(_tree_graph, tree_conditional),
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


def binary_concepts(p=0.65):
    """Two hidden concepts emitting noisy yes/no observations."""
    P = np.array([[p, 1 - p], [1 - p, p]])
    return np.log(P), {"latent": np.array([-1.0, 1.0]), "observations": np.array([-1.0, 1.0])}


def temperature_concepts(n_theta=5, n_observations=5, theta_lim=1.0, observation_lim=1.0):
    """Discrete temperature line with sufficient statistics ``(1, mu_x)``."""
    theta = np.linspace(-theta_lim, theta_lim, n_theta)
    mu = np.linspace(-observation_lim, observation_lim, n_observations)
    L = theta[:, None] * mu[None] - 0.5 * mu[None] ** 2
    return L, {"latent": theta, "observations": mu}


def word_attribute(m=8, gamma=0.03, alpha_decay=0.9, beta_step=0.0):
    """Word -> attribute kernel: words are the 2^m corners of the feature cube, outcomes the
    m attribute tokens plus a null reference (logit 0), with
    log p(i|x)/p(null|x) = alpha_i x_i + beta_i, alpha_i = w0(gamma) * alpha_decay^i
    (reliability), beta_i = -i * beta_step (attribute frequency).
    Returns (L (2^m, m+1), {"Z" (2^m, m) in ±1, "alpha" (m,), "beta" (m,)})."""
    from scipy.special import logsumexp
    alpha = 0.5 * np.log(1 / np.tanh(gamma)) * alpha_decay ** np.arange(m)
    beta = -beta_step * np.arange(m)
    Z = 2.0 * cube_codes(m) - 1.0
    G = np.column_stack([Z * alpha + beta, np.zeros(len(Z))])
    L = G - logsumexp(G, axis=1, keepdims=True)
    return L, {"Z": Z, "alpha": alpha, "beta": beta}


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
                    "categorical": categorical, "mixture": mixture,
                    "binary_concepts": binary_concepts, "temperature_concepts": temperature_concepts,
                    "word_attribute": word_attribute}


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
# PCFG (PCFG): the tree generalization of the HMM setting.
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

PCFGGrammar = namedtuple("PCFGGrammar", ["pi", "p_branch", "S", "Tt", "Em"])
# pi (m,) root-state prior; p_branch (m,) branch-vs-emit prob per state; S/Tt (m,m) left/right child
# state transitions (row-stochastic); Em (m,V) emission (row-stochastic; identity => terminals ARE
# leaf states). Subcritical (mean offspring 2*p_branch < 1) so trees terminate a.s.


_GRAMMAR_BUILDERS = {}                                            # populated after each builder below


def build_grammar(spec):
    """Build an ``PCFGGrammar`` from a spec ``{'grammar': name, **params}``."""
    spec = dict(spec)
    name = spec.pop("grammar")
    if name not in _GRAMMAR_BUILDERS:
        raise ValueError(f"Unknown grammar {name!r}; known: {sorted(_GRAMMAR_BUILDERS)}")
    return _GRAMMAR_BUILDERS[name](spec)


# ── Exact Bayes oracle: inside chart + left-to-right prefix / next-token (Stolcke-style) ──

def pcfg_inside_chart(x, g: PCFGGrammar):
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


def pcfg_prefix_chart(x, g: PCFGGrammar, In):
    """Prefix-inside ``PIn[i,j][h] = p(state-h subtree yields x_i..x_j as a *prefix* of its full
    yield)``. ``R = (I - diag(p_branch) S)^-1`` resolves the left-spine descent."""
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


def pcfg_next_token(x, g: PCFGGrammar):
    """Exact left-to-right Bayes predictions for a terminal sequence ``x`` (length k).
    Returns ``(dists, beliefs)``: ``dists`` (k+1, V+1), row t = ``p(next | x[:t])`` over the V
    terminals plus a final EOS column; ``beliefs`` (k+1, m), row t = the posterior over the next
    emitter's latent state."""
    m = len(g.pi); V = g.Em.shape[1]; k = len(x); pb = g.p_branch; e = 1.0 - pb
    R = np.linalg.inv(np.eye(m) - (pb[:, None] * g.S))
    In = pcfg_inside_chart(x, g)
    PIn = pcfg_prefix_chart(x, g, In) if k else {}

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
        o = _pcfg_boundary(x, t, g, In, R)                         # prefix-outside boundary vector
        state = o * e                                                # state of the node emitting x_t
        dists[t, :V] = (state @ g.Em) / Z
        dists[t, V] = eos / Z
        beliefs[t] = _normalize(state)                              # next-emitter posterior (0 if terminal)
    return dists, beliefs


def _pcfg_boundary(x, t, g: PCFGGrammar, In, R):
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
    """Assemble a ``PCFGGrammar`` from named states. ``leaves`` maps a leaf-state name to its
    terminal id; ``rules`` maps an internal-state name to (left, right), each a name or a
    ``{name: p}`` distribution; ``root_prior`` maps a state name to its prior mass; ``emit`` is unused."""
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
    return PCFGGrammar(pi=pi, p_branch=pb, S=S, Tt=Tt, Em=Em)


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
    bracket type chosen uniformly. Terminals: open_i = i, close_i = k+i (V = 2k)."""
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


def mini_nl_grammar(spec):
    """Miniature phrase-structure grammar, binarized: S -> NP VP; NP -> Det Nom; Nom -> noun |
    Adj Nom (up to ``max_adj`` adjectives); VP -> V | V NP | V NP PP; PP -> Prep NP.
    Terminals: det, adj, noun, verb, prep (V = 5)."""
    n_adj = int(dict(spec).get("max_adj", 2))
    det, adj, noun, verb, prep = range(5)
    leaves = {"Det": det, "Adj": adj, "Noun": noun, "Verb": verb, "Prep": prep}
    states = list(leaves); rules = {}

    def nom(level):                                                  # noun-phrase tail: noun, or adjective + tail
        return "Noun" if level >= n_adj else {"Noun": 0.6, f"Adj{level}": 0.4}
    for level in range(n_adj):
        rules[f"Adj{level}"] = ("Adj", nom(level + 1)); states.append(f"Adj{level}")
    rules["NP"] = ("Det", nom(0))
    rules["VObj"] = ("Verb", "NP")                                   # transitive: verb + object
    rules["VObjPP"] = ("Verb", "VPP"); rules["VPP"] = ("NP", "PP")   # verb + object + prepositional phrase
    rules["PP"] = ("Prep", "NP")
    rules["S"] = ("NP", {"Verb": 0.4, "VObj": 0.4, "VObjPP": 0.2})   # intransitive / transitive / + PP
    states += ["NP", "VObj", "VObjPP", "VPP", "PP", "S"]
    return _grammar_from_rules(states, leaves, rules, {"S": 1.0}, V=5, emit=None)


def mini_pl_grammar(spec):
    """Miniature control-flow language, statements nesting to depth ``D``: an assignment ``asgn``,
    ``if cond then S end``, ``while cond do S end``, or a sequence ``asgn ; S``.
    Terminals: asgn, if, then, while, do, end, cond, semi (V = 8)."""
    D = int(dict(spec).get("depth", 3))
    asgn, iftok, then, whiletok, do, end, cond, semi = range(8)
    leaves = {"Asgn": asgn, "If": iftok, "Then": then, "While": whiletok, "Do": do,
              "End": end, "Cond": cond, "Semi": semi}
    states = list(leaves); rules = {}

    def stmt(d):                                                     # statement choices at nesting budget d
        if d >= D:
            return {"Asgn": 1.0}                                     # budget exhausted: atomic only
        return {"Asgn": 0.4, f"IF{d}": 0.25, f"WH{d}": 0.2, f"SEQ{d}": 0.15}
    for d in range(D):
        rules[f"IF{d}"] = ("If", f"IF{d}a"); rules[f"IF{d}a"] = ("Cond", f"IF{d}b")
        rules[f"IF{d}b"] = ("Then", f"IF{d}c"); rules[f"IF{d}c"] = (stmt(d + 1), "End")
        rules[f"WH{d}"] = ("While", f"WH{d}a"); rules[f"WH{d}a"] = ("Cond", f"WH{d}b")
        rules[f"WH{d}b"] = ("Do", f"WH{d}c"); rules[f"WH{d}c"] = (stmt(d + 1), "End")
        rules[f"SEQ{d}"] = ("Asgn", f"SEQ{d}a"); rules[f"SEQ{d}a"] = ("Semi", stmt(d + 1))
        states += [f"IF{d}", f"IF{d}a", f"IF{d}b", f"IF{d}c",
                   f"WH{d}", f"WH{d}a", f"WH{d}b", f"WH{d}c", f"SEQ{d}", f"SEQ{d}a"]
    return _grammar_from_rules(states, leaves, rules, stmt(0), V=8, emit=None)


_GRAMMAR_BUILDERS.update({"anbn": anbn_grammar, "dyck": dyck_grammar,
                          "mini_nl": mini_nl_grammar, "mini_pl": mini_pl_grammar})


# ── interpretable prefix labels: each grammar's hand sufficient statistic (for coloring analyses) ──

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
