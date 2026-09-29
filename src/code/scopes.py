"""Scope-stack programs: a pushdown process over code-like tokens. A block (`for` / `def` / `if` / `with`) is
opened by its kind token and a fresh name, and stays open until `end`. The next-token scores are additive over
the open scopes, log p(token | stack) = base + sum_s [b 1(token = name_s) + kind effect + depth effect] + const,
so the Bayes next-token state is the stack itself and its code is one step per open scope."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import KernelDataset

KINDS = ["for", "def", "if", "with"]
KEYS = ["break", "continue", "return", "yield", "pass"]
KIND_EFFECTS = {"for": {"break": 3.0, "continue": 3.0}, "def": {"return": 3.0, "yield": 1.5}, "if": {"pass": 1.5}, "with": {}}
DEPTH_EFFECTS = {"end": 1.0, **{k: -0.7 for k in KINDS}}          # deeper: closing likelier, opening rarer
BASE = {"name": -3.0, "key": -2.0, "pass": -1.0, "end": -4.0}
FRESH, TAKEN = 0.0, -6.0                                          # the name after a kind: avoid open names
MAX_DEPTH = 32
POOL_CHUNK = 16384


class ScopeDataset(Dataset):
    """Streams of scope-stack programs for a causal LM, with exact next-token targets. ``scope_kwargs``:
    ``n_names`` (bound-name tokens), ``name_bonus`` (b, how much an open scope's name is used),
    ``kind_strength`` (scales the kind effects; 0 is the null, where a block's kind changes nothing),
    ``pool_size`` (programs sampled once, which the batches draw from)."""

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(config.random_seed)
        kw = config.get("scope_kwargs")
        self.kinds, self.names = KINDS, [f"n{i}" for i in range(kw["n_names"])]
        self.vocab = self.kinds + self.names + KEYS + ["end"]
        V = len(self.vocab); ix = {t: i for i, t in enumerate(self.vocab)}
        self.ix = ix
        base = np.zeros(V)
        base[[ix[n] for n in self.names]] = BASE["name"]; base[[ix[k] for k in KEYS]] = BASE["key"]
        base[ix["pass"]] = BASE["pass"]; base[ix["end"]] = BASE["end"]
        W = np.zeros((len(KINDS), len(self.names), V))           # the step each (kind, name) scope adds
        for a, k in enumerate(KINDS):
            for tok, v in KIND_EFFECTS[k].items():
                W[a, :, ix[tok]] += kw["kind_strength"] * v
            for tok, v in DEPTH_EFFECTS.items():
                W[a, :, ix[tok]] += v
            W[a, np.arange(len(self.names)), [ix[n] for n in self.names]] += kw["name_bonus"]
        self.base = torch.tensor(base, device=self.device)
        self.W = torch.tensor(W, device=self.device)
        self.P = torch.cat([self.base, self.W.flatten()])        # the run fingerprint (`kernel_hash`)

        self.sample_mode, self.sequential, self.arity = "sequence", True, 1
        self.infer_latent, self.features, self.info = False, None, {}
        self.label_tokens = torch.arange(V)
        self.item_tokens = torch.arange(V)
        self.start_token = V
        self.vocab_size = V + 1
        self.num_dims = V
        self._pool = None

    compute_metrics = KernelDataset.compute_metrics

    @property
    def context_length(self):
        return self.config.training_kwargs["context_length"]

    @property
    def seq_len(self):
        return self.context_length + 1

    def __len__(self):
        return 1

    def __getitem__(self, i=None):
        return self.batch()

    # ── the process ─────────────────────────────────────────────────────────
    def _init_state(self, B):
        z = lambda *s, dt=torch.long: torch.zeros(*s, dtype=dt, device=self.device)
        return {"S": z(B, len(self.vocab), dt=torch.float64), "kind": z(B, MAX_DEPTH), "name": z(B, MAX_DEPTH),
                "depth": z(B), "taken": z(B, len(self.names), dt=torch.float64),
                "expect": z(B, dt=torch.bool), "pending": z(B)}

    def _logits(self, st):
        """Unnormalized log p(next token) for every stream in the batch, (B, V)."""
        normal = self.base + st["S"]
        fresh = torch.full_like(normal, -30.0)
        n0 = len(self.kinds)
        fresh[:, n0:n0 + len(self.names)] = FRESH + TAKEN * st["taken"]
        return torch.where(st["expect"][:, None], fresh, normal)

    def _step(self, st, tok):
        """Advance every stream by one token (B,)."""
        n0, rows = len(self.kinds), torch.arange(len(tok), device=self.device)
        push = st["expect"]
        name = (tok - n0).clamp(0, len(self.names) - 1)
        d = st["depth"]
        st["kind"][rows[push], d[push]] = st["pending"][push]
        st["name"][rows[push], d[push]] = name[push]
        st["S"][push] += self.W[st["pending"][push], name[push]]
        st["taken"][rows[push], name[push]] += 1
        st["depth"] = d + push.long()
        opened = ~push & (tok < n0)
        st["pending"] = torch.where(opened, tok, st["pending"])
        st["expect"] = opened
        pop = ~push & (tok == self.ix["end"]) & (st["depth"] > 0)
        top = (st["depth"] - 1).clamp(min=0)
        k, n = st["kind"][rows, top], st["name"][rows, top]
        st["S"][pop] -= self.W[k[pop], n[pop]]
        st["taken"][rows[pop], n[pop]] -= 1
        st["depth"] = st["depth"] - pop.long()
        assert int(st["depth"].max()) < MAX_DEPTH

    def log_row(self, stack):
        """log p(next token | an explicit stack of (kind, name)), numpy (V,), outside a pending name."""
        lo = self.base.clone()
        for k, n in stack:
            lo += self.W[self.kinds.index(k), self.names.index(n)]
        return torch.log_softmax(lo, -1).cpu().numpy()

    # ── interface read by the trainer / evaluator ───────────────────────────
    def _sample(self, B, rng):
        """B programs of ``context_length`` tokens after the start token, token by token: (B, T+1)."""
        st = self._init_state(B)
        seq = [torch.full((B,), self.start_token, dtype=torch.long, device=self.device)]
        for _ in range(self.context_length):
            tok = torch.multinomial(torch.softmax(self._logits(st), -1), 1, generator=rng).squeeze(1)
            self._step(st, tok)
            seq.append(tok)
        return torch.stack(seq, 1)

    def batch(self, size=None):
        """``size`` programs drawn from a pool of ``scope_kwargs.pool_size``, sampled once with a generator of its
        own (sampling token by token at every training step is too slow)."""
        if self._pool is None:
            n, rng = self.config.get("scope_kwargs")["pool_size"], torch.Generator(device=self.device)
            rng.manual_seed(self.config.random_seed + 1)
            self._pool = torch.cat([self._sample(min(POOL_CHUNK, n - i), rng) for i in range(0, n, POOL_CHUNK)])
        B = size or self.config.training_kwargs["batch_size"]
        seq = self._pool[torch.randint(len(self._pool), (B,), generator=self.rng, device=self.device)]
        return {"input_ids": seq, "labels": seq.clone()}

    def block_entries(self, input_ids, H):
        """``(kind, name, depth, step)`` for every block opened in the streams: the code after the bound name
        minus the code before the kind token. ``H`` (B, T, d) is aligned with ``optimal_predictions`` (position
        t is the state after token t)."""
        toks = np.asarray(input_ids.cpu() if hasattr(input_ids, "cpu") else input_ids)
        out = []
        for b in range(len(toks)):
            depth = 0
            for t in range(1, min(toks.shape[1] - 1, H.shape[1] - 1)):
                tok = self.vocab[toks[b, t]]
                if tok in self.kinds:
                    out.append((tok, self.vocab[toks[b, t + 1]], depth, H[b, t + 1] - H[b, t - 1]))
                depth += (tok in self.kinds) - (tok == "end" and depth > 0)
        return out

    def blocks(self, ids):
        """The nesting tree of one stream (start token first): ``(position, parent, kind, name)`` per node, the
        module first (position 0), then each block at the position of its bound name; parent indexes this list."""
        out, stack = [(0, -1, None, None)], [0]
        for t in range(1, len(ids) - 1):
            tok = self.vocab[ids[t]]
            if tok in self.kinds:
                out.append((t + 1, stack[-1], tok, self.vocab[ids[t + 1]]))
                stack.append(len(out) - 1)
            elif tok == "end" and len(stack) > 1:
                stack.pop()
        return out

    def optimal_predictions(self, input_ids):
        """Exact next-token distributions aligned with ``logits[:, :-1]``: (B, T) -> (B, T-1, V)."""
        st = self._init_state(len(input_ids))
        out = []
        for t in range(input_ids.shape[1] - 1):
            if t > 0:                                             # position 0 is the start token
                self._step(st, input_ids[:, t])
            out.append(torch.softmax(self._logits(st), -1))
        return torch.stack(out, 1)
