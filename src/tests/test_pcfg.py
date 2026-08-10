"""PCFG: latent states diffuse down a *tree* (the tree generalization of the HMM's
chain) and each leaf emits a terminal. The exact left-to-right Bayes oracle (inside chart + prefix
parser) is checked against brute-force enumeration and Monte-Carlo; the grammars against language
membership; the dataset against the batch contract shared with the other settings."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import (PCFGGrammar, PCFGDataset, build_grammar, _kl_metrics,
                           pcfg_inside_chart, pcfg_next_token)
from src.code.eval import cca


# ── grammar gallery (one entry per builder) ──────────────────────────────────
GRAMMARS = [
    ("anbn", {"depth": 5}),
    ("dyck", {"k": 2, "depth": 3}),
    ("mini_nl", {"max_adj": 2}),
    ("mini_pl", {"depth": 3}),
]


def _cfg(grammar="dyck", grammar_kwargs=None, max_len=24, seed=0):
    gk = grammar_kwargs or {"k": 2, "depth": 3}
    return Config(OmegaConf.create({
        "setting": "pcfg", "grammar": grammar, "grammar_kwargs": gk, "max_len": max_len,
        "sample_mode": "sequence", "random_seed": seed,
        "model_kwargs": {"model_type": "transformer"},
        "training_kwargs": {"batch_size": 16, "context_length": 40},
        "eval_kwargs": {"num_eval_batches": 1, "output_hidden_states": False},
    }))


# ── brute-force + Monte-Carlo ground truth for the oracle ────────────────────
def _rand_grammar(m, V, seed):
    """A random subcritical grammar (mean offspring 2*p_branch < 1 => finite trees a.s.)."""
    rng = np.random.default_rng(seed)
    return PCFGGrammar(pi=rng.dirichlet(np.ones(m)), p_branch=rng.uniform(0.15, 0.35, m),
                        S=rng.dirichlet(np.ones(m), m), Tt=rng.dirichlet(np.ones(m), m),
                        Em=rng.dirichlet(np.ones(V), m))


def _brute_sentence_probs(g, max_terms):
    """Exact ``{terminal-tuple: prob}`` by enumerating every tree with <= max_terms leaves."""
    e = 1.0 - g.p_branch; m = len(g.pi); V = g.Em.shape[1]; memo = {}

    def sub(h, cap):
        if (h, cap) in memo:
            return memo[(h, cap)]
        out = {}
        for w in range(V):
            if g.Em[h, w] > 0:
                out[(w,)] = out.get((w,), 0.0) + e[h] * g.Em[h, w]
        if cap >= 2:
            for hL in range(m):
                dl = sub(hL, cap - 1)
                for hR in range(m):
                    pref = g.p_branch[h] * g.S[h, hL] * g.Tt[h, hR]
                    if pref == 0:
                        continue
                    dr = sub(hR, cap - 1)
                    for yl, pl in dl.items():
                        room = cap - len(yl)
                        for yr, pr in dr.items():
                            if len(yr) <= room:
                                y = yl + yr; out[y] = out.get(y, 0.0) + pref * pl * pr
        memo[(h, cap)] = out
        return out

    total = {}
    for h in range(m):
        for y, p in sub(h, max_terms).items():
            total[y] = total.get(y, 0.0) + g.pi[h] * p
    return total


def _brute_next(sp, x, V):
    """Exact ``p(next | x)`` over the V terminals + EOS from precomputed sentence probabilities."""
    x = tuple(x); d = np.zeros(V + 1)
    for y, p in sp.items():
        if y[:len(x)] == x:
            if len(y) == len(x):
                d[V] += p
            elif len(y) > len(x):
                d[y[len(x)]] += p
    return d / d.sum()


def _sample(g, srng, cap=400):
    m = len(g.pi); V = g.Em.shape[1]; terms = []

    def rec(h):
        if len(terms) > cap:
            return
        if srng.random() < g.p_branch[h]:
            rec(int(srng.choice(m, p=g.S[h]))); rec(int(srng.choice(m, p=g.Tt[h])))
        else:
            terms.append(int(srng.choice(V, p=g.Em[h])))

    rec(int(srng.choice(m, p=g.pi)))
    return terms


# ── grammar layer ────────────────────────────────────────────────────────────
def test_build_grammar_unknown_raises():
    with pytest.raises(ValueError):
        build_grammar({"grammar": "does_not_exist"})


@pytest.mark.parametrize("name,kw", GRAMMARS)
def test_grammar_matrices_are_valid(name, kw):
    g = build_grammar({"grammar": name, **kw})
    assert np.isclose(g.pi.sum(), 1.0)
    assert ((g.p_branch >= 0) & (g.p_branch <= 1)).all()
    assert np.allclose(g.Em.sum(1), 1.0)                      # every emission row is a distribution
    branch = g.p_branch > 0                                   # only branching states have child dists
    assert np.allclose(g.S[branch].sum(1), 1.0)
    assert np.allclose(g.Tt[branch].sum(1), 1.0)


def _is_anbn(s):
    na = sum(t == 0 for t in s); nb = len(s) - na
    return na >= 1 and na == nb and s == [0] * na + [1] * nb


def _is_dyck(s, k=2):
    st = []
    for t in s:
        if t < k:
            st.append(t)
        else:
            if not st or st[-1] != t - k:
                return False
            st.pop()
    return len(st) == 0 and len(s) >= 2


@pytest.mark.parametrize("name,kw,checker", [
    ("anbn", {"depth": 6}, _is_anbn),
    ("dyck", {"k": 2, "depth": 4}, lambda s: _is_dyck(s, 2)),
    ("mini_nl", {"max_adj": 2}, None),
    ("mini_pl", {"depth": 3}, None),
])
def test_sampled_strings_are_valid_language_members(name, kw, checker):
    g = build_grammar({"grammar": name, **kw}); srng = np.random.default_rng(0)
    for _ in range(200):
        s = _sample(g, srng)
        In = pcfg_inside_chart(np.array(s, dtype=int), g)             # every sampled string has positive
        assert g.pi @ In[(0, len(s) - 1)] > 0                         # probability under the exact chart
        if checker is not None:
            assert checker(s)                                         # and matches its closed-form language


# ── exact oracle vs brute force / Monte-Carlo ───────────────────────────────
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_inside_chart_matches_brute_force(seed):
    """``p(sentence)`` from the CKY inside chart equals exact enumeration (tight — a short sentence's
    probability is fully counted regardless of the enumeration cap)."""
    g = _rand_grammar(m=3, V=2, seed=seed)
    sp = _brute_sentence_probs(g, max_terms=8)
    for x in [[0], [1], [0, 1], [1, 0], [0, 0], [1, 1], [0, 1, 0]]:
        In = pcfg_inside_chart(np.array(x, dtype=int), g)
        p_oracle = g.pi @ In[(0, len(x) - 1)]
        assert abs(p_oracle - sp.get(tuple(x), 0.0)) < 1e-9


@pytest.mark.parametrize("seed", [0, 1])
def test_next_token_matches_brute_force(seed):
    g = _rand_grammar(m=2, V=2, seed=seed)
    sp = _brute_sentence_probs(g, max_terms=11)               # enumerate once; reuse for every prefix
    for x in [[], [0], [1], [0, 1], [1, 0]]:
        d, _ = pcfg_next_token(np.array(x, dtype=int), g)
        bf = _brute_next(sp, x, g.Em.shape[1])               # truncated tail -> loose tolerance
        assert np.abs(d[len(x)] - bf).max() < 5e-3


@pytest.mark.parametrize("name,kw", GRAMMARS)
def test_next_token_rows_are_distributions(name, kw):
    g = build_grammar({"grammar": name, **kw}); srng = np.random.default_rng(0)
    for _ in range(5):
        d, _ = pcfg_next_token(np.array(_sample(g, srng), dtype=int), g)
        assert np.allclose(d.sum(1), 1.0, atol=1e-5)
        assert (d >= -1e-9).all()


def test_next_token_matches_monte_carlo():
    """Sampler and oracle are independent implementations; they must agree on a branching grammar."""
    g = build_grammar({"grammar": "mini_nl", "max_adj": 2})
    srng = np.random.default_rng(1); N = 40000
    seqs = [_sample(g, srng) for _ in range(N)]
    # well-sampled prefixes only (deeper prefixes are covered exactly by the brute-force test).
    for x in [[], [0]]:
        cnt = np.zeros(g.Em.shape[1] + 1); tot = 0
        for s in seqs:
            if s[:len(x)] == x:
                tot += 1
                cnt[-1 if len(s) == len(x) else s[len(x)]] += 1
        d, _ = pcfg_next_token(np.array(x, dtype=int), g)
        assert np.abs(d[len(x)] - cnt / tot).max() < 0.02


def test_anbn_known_closed_form_values():
    D = 5
    g = build_grammar({"grammar": "anbn", "depth": D})
    d0, _ = pcfg_next_token(np.zeros(0, dtype=int), g)
    assert np.isclose(d0[0, 0], 1.0)                          # always starts with 'a'
    # p(sentence "ab") = p(n=1) = 1/D under the uniform length prior
    In = pcfg_inside_chart(np.array([0, 1], dtype=int), g)
    assert np.isclose(g.pi @ In[(0, 1)], 1.0 / D)


# ── Hankel predictive-state basis (string-level spectral) ───────────────────
def test_hankel_forward_state_is_the_sufficient_statistic():
    """The Hankel forward state (SVD of ``H[u,v]=p(uv)``) is the minimal sufficient statistic of the
    future: it has finite rank, and the exact next-token distribution is a *linear* function of it."""
    from src.code.geometry import hankel_forward_states
    g = build_grammar({"grammar": "dyck", "k": 2, "depth": 3}); srng = np.random.default_rng(0)
    seqs = [_sample(g, srng) for _ in range(300)]
    P = sorted({tuple(s[:t]) for s in seqs for t in range(len(s) + 1)})
    Suf = sorted({tuple(s[t:]) for s in seqs for t in range(len(s) + 1)})
    fwd, info = hankel_forward_states(g, P, Suf)
    assert 1 < info["rank"] < len(P)                         # finite, non-trivial predictive rank
    hists = [tuple(s[:t]) for s in seqs[:60] for t in range(len(s) + 1)]
    F = np.array([fwd(h) for h in hists])
    D = np.array([pcfg_next_token(np.array(h, dtype=int), g)[0][len(h)] for h in hists])
    W = np.linalg.lstsq(np.c_[F, np.ones(len(F))], D, rcond=None)[0]
    r2 = 1 - ((D - np.c_[F, np.ones(len(F))] @ W) ** 2).sum() / ((D - D.mean(0)) ** 2).sum()
    assert r2 > 0.999                                        # next-token dist is linear in the forward state


def test_predictive_states_are_sufficient_for_every_observed_prefix():
    """``pcfg_predictive_states`` returns the Hankel forward state at each *distinct* prefix of a
    corpus. Being the minimal sufficient statistic of the future, the exact next-token distribution
    must be a linear function of it — the property the network's hidden state is tested against."""
    from src.code.geometry import pcfg_predictive_states
    g = build_grammar({"grammar": "dyck", "k": 2, "depth": 3}); srng = np.random.default_rng(0)
    seqs = [_sample(g, srng) for _ in range(200)]
    prefixes, F, info = pcfg_predictive_states(g, seqs, max_tests=40)
    assert prefixes == sorted({tuple(s[:t]) for s in seqs for t in range(len(s) + 1)})
    assert F.shape == (len(prefixes), len(info["tests"]) + 1)
    assert [tuple(seqs[b][:t]) for b, t in info["loc"]] == prefixes   # rows index the hidden states
    D = np.array([pcfg_next_token(np.array(p, dtype=int), g)[0][len(p)] for p in prefixes])
    Fa = np.c_[F, np.ones(len(F))]
    r2 = 1 - ((D - Fa @ np.linalg.lstsq(Fa, D, rcond=None)[0]) ** 2).sum() / ((D - D.mean(0)) ** 2).sum()
    assert r2 > 0.999
    # the tests separate every predictive state: rank stops growing once they are long enough
    assert pcfg_predictive_states(g, seqs, test_len=5, max_tests=40)[2]["rank"] == info["rank"]


# ── dataset ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name,kw", GRAMMARS)
def test_dataset_batch_contract(name, kw):
    ds = PCFGDataset(_cfg(name, kw, max_len=16)); b = ds[0]
    B = ds._batch_size()
    assert b["input_ids"].shape == (B, ds.seq_len)
    assert b["labels"].shape == (B, ds.seq_len)
    pad = b["input_ids"] == ds.pad_token
    assert (b["labels"][pad] == -100).all()                  # padding ignored in the LM loss
    el = ds.optimal_predictions(b["input_ids"])
    assert el.shape == (B, ds.max_len + 1, ds.num_dims)
    valid = el.sum(-1) > 0
    ones = torch.ones(int(valid.sum()), device=el.device)
    assert torch.allclose(el[valid].sum(-1), ones, atol=1e-4)   # valid Bayes rows are distributions
    assert torch.allclose(el[~valid], torch.zeros_like(el[~valid]))   # padded rows are zero


def test_dataset_vocab_and_token_layout():
    ds = PCFGDataset(_cfg("mini_nl", {"max_adj": 2}))
    assert ds.V == 5                                          # det, adj, noun, verb, prep
    assert ds.num_dims == 6 and ds.vocab_size == 8           # terminals+EOS ; terminals+eos+start+pad
    assert ds.eos_token == 5 and ds.start_token == 6 and ds.pad_token == 7
    assert ds.item_tokens.tolist() == list(range(6))         # maps logits -> terminals + EOS


# ── shared KL metric (padding mask) ─────────────────────────────────────────
def test_kl_metrics_averages_over_valid_positions_only():
    # 2 valid target rows (uniform) + 2 padded (all-zero) rows; logits mispredict the valid rows.
    labels = torch.zeros(1, 4, 2); labels[0, 0] = labels[0, 1] = torch.tensor([0.5, 0.5])
    logits = torch.zeros(1, 5, 2); logits[0, 0] = logits[0, 1] = torch.log(torch.tensor([0.9, 0.1]))
    per_row = float((0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1)))
    out = _kl_metrics((logits, labels), return_all=False)
    assert np.isclose(out["loss"], per_row, atol=1e-4)       # mean over the 2 valid rows, not diluted by padding


# ── integration: a transformer recovers the belief ──────────────────────────
@pytest.mark.slow
def test_transformer_recovers_anbn_belief():
    from src.code.train import train_inline
    model, ds, ev, hist, config = train_inline(
        ["runner=pcfg_anbn", "name=test_anbn", "wandb.enabled=false", "overwrite=true",
         "eval_kwargs.output_hidden_states=true", "grammar_kwargs.depth=4", "max_len=9",
         "training_kwargs.train_steps=2000"])
    seqs = [ds._sample_sentence()[0] for _ in range(150)]
    ids = np.full((len(seqs), ds.seq_len), ds.pad_token); ids[:, 0] = ds.start_token
    for b, s in enumerate(seqs):
        ids[b, 1:1 + len(s)] = s; ids[b, 1 + len(s)] = ds.eos_token
    ids = torch.tensor(ids, dtype=torch.long, device=ds.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    H, Bel, kls = [], [], []
    marg = []
    def count(pre): return sum(t == 0 for t in pre) - sum(t == 1 for t in pre)
    keys = sorted({count(s[:t]) for s in seqs for t in range(len(s) + 1)})
    kid = {k: i for i, k in enumerate(keys)}
    pm = torch.softmax(out["logits"][:, :-1][..., ds.item_tokens], -1).cpu().numpy()
    for b, s in enumerate(seqs):
        d, _ = pcfg_next_token(np.array(s, dtype=int), ds.g)
        for t in range(len(s) + 1):
            H.append(out["hidden_states"][-1][b, t].cpu().numpy())
            oh = np.zeros(len(keys)); oh[kid[count(s[:t])]] = 1.0; Bel.append(oh)
            marg.append(d[t]); kls.append(float((d[t] * (np.log(d[t] + 1e-9) - np.log(pm[b, t] + 1e-9))).sum()))
    H = np.array(H); Bel = np.array(Bel)
    kl_marg = np.mean([float((r * (np.log(r + 1e-9) - np.log(np.mean(marg, 0) + 1e-9))).sum()) for r in marg])
    assert cca(H, Bel) > 0.9                                  # belief linearly present in the hidden
    assert np.mean(kls) < 0.5 * kl_marg                       # and behavior well below the marginal baseline


@pytest.mark.parametrize("runner", ["pcfg_anbn", "pcfg_dyck", "pcfg_mininl", "pcfg_minipl"])
def test_shipped_pcfg_config_has_hierarchical_signal(runner):
    """Guardrail on the *shipped* runner configs: the Bayes predictor must beat a bigram baseline
    by a clear margin — a Markov-solvable grammar would make the hierarchy claims vacuous (the
    PCFG analog of the HMM's Bayes-equals-stationary failure)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from pyprojroot import here
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(here() / "conf")):
        cfg = compose(config_name="config", overrides=[f"runner={runner}"])
    cfg.training_kwargs.batch_size = 1024
    ds = PCFGDataset(Config(cfg))
    b = ds[0]
    bayes = ds.optimal_predictions(b["input_ids"]).cpu().numpy()
    ids = b["input_ids"].cpu().numpy()
    T, C = bayes.shape[1], bayes.shape[2]
    valid = bayes.sum(-1) > 0.5

    nxt = ids[:, 1:T + 1].copy(); nxt[nxt == ds.eos_token] = C - 1   # outcome class = terminal | EOS
    prev = ids[:, 0:T]                                               # conditioning token (start at t=0)
    bigram = np.full((ds.vocab_size, C), 1e-9)
    np.add.at(bigram, (prev[valid], nxt[valid]), 1)
    bigram /= bigram.sum(1, keepdims=True)

    kl = (bayes[valid] * (np.log(bayes[valid] + 1e-9) - np.log(bigram[prev[valid]] + 1e-9))).sum(-1)
    assert kl.mean() > 0.05                                          # structure beyond one-step stats


@pytest.mark.parametrize("runner", ["pcfg_anbn_rnn", "pcfg_dyck_rnn", "pcfg_mininl_rnn", "pcfg_minipl_rnn"])
def test_shipped_pcfg_rnn_runner_forward_contract(runner):
    """The GRU runners compose, size from the dataset, and expose per-position hidden states —
    the contract the belief-decoding analysis reads (recurrent state at every prefix position)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict
    from pyprojroot import here
    from src.code.models import init_model
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(here() / "conf")):
        cfg = compose(config_name="config", overrides=[f"runner={runner}"])
    cfg.training_kwargs.batch_size = 8
    config = Config(cfg)
    ds = PCFGDataset(config)
    mc = config.cfg.model_kwargs.model_config
    with open_dict(mc):
        mc.vocab_size = ds.vocab_size; mc.num_classes = ds.num_dims; mc.arity = ds.arity
    model = init_model(config)
    b = ds[0]
    out = model(b["input_ids"].to(ds.device), labels=b["labels"].to(ds.device),
                output_hidden_states=True)
    assert out["hidden_states"][-1].shape == (8, ds.seq_len, mc.hidden_size)
    assert out["logits"].shape == (8, ds.seq_len, ds.vocab_size)
    assert torch.isfinite(out["loss"])


from src.code.data import anbn_count, dyck_stack, pl_stack_phase, pl_nesting


# ── interpretable prefix labels (each grammar's hand sufficient statistic) ──
def test_anbn_count_is_unmatched_a():
    assert anbn_count([]) == 0
    assert anbn_count([0, 0, 0]) == 3                       # aaa -> 3 unmatched
    assert anbn_count([0, 0, 1]) == 1                       # aab -> 1 unmatched
    assert anbn_count([0, 0, 1, 1]) == 0                    # aabb -> balanced


def test_dyck_stack_tracks_open_brackets():
    assert dyck_stack([], k=2) == ()
    assert dyck_stack([0, 1], k=2) == (0, 1)                # two opens
    assert dyck_stack([0, 1, 3], k=2) == (0,)              # close type-1 -> pop
    assert dyck_stack([0, 1, 3, 2], k=2) == ()             # close type-0 -> empty


def test_pl_stack_phase_pushes_if_while_pops_end():
    # if(1) cond(6) then(2) while(3) cond(6) do(4) asgn(0) end(5)
    pre = [1, 6, 2, 3, 6, 4, 0, 5]
    stack, phase = pl_stack_phase(pre)
    assert stack == ("if",)                                 # inner while closed, outer if still open
    assert phase == 5                                       # just saw `end`
    assert pl_nesting([1, 6, 2, 1, 6, 2, 0]) == 2           # two nested ifs open
