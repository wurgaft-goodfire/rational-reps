"""Scope-stack programs: blocks (for / def / if) bind a fresh name and stay open until `end`, and every
open scope adds its effect to the next-token scores, so the Bayes next-token state is the stack and its code
is a sum of one step per open scope: a translation code, the same step at any depth."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import init_dataset
from src.code.geometry import factor_logcond, predicted_inference_geometry
from src.code.scopes import ScopeDataset


def _dataset(kind_strength=1.0, T=48, B=512):
    cfg = OmegaConf.create({"setting": "scopes", "random_seed": 0, "sample_mode": "sequence",
                            "scope_kwargs": {"n_names": 32, "name_bonus": 4.0, "kind_strength": kind_strength, "pool_size": 8192},
                            "training_kwargs": {"context_length": T, "batch_size": B}})
    return ScopeDataset(cfg)


def _code(ds, batch):
    """Oracle code at every position of a batch, (B, T, r), from the stacked exact rows."""
    P = ds.optimal_predictions(batch["input_ids"].to(ds.device)).double().cpu().numpy()   # (B, T-1, V)
    R = np.log(P).reshape(-1, P.shape[-1])
    _, _, D, S = factor_logcond(R)
    r = int((S > 1e-8 * S[0]).sum())
    return predicted_inference_geometry(R, D[:r]).reshape(P.shape[0], P.shape[1], r)


def _entries(ds, batch, H):
    return ds.block_entries(batch["input_ids"], H)


def test_rows_are_distributions_over_the_program_tokens():
    ds = _dataset()
    batch = ds.batch()
    P = ds.optimal_predictions(batch["input_ids"].to(ds.device))
    assert P.shape == (512, 48, len(ds.label_tokens)) and torch.allclose(P.sum(-1), torch.ones(1, device=P.device).double(), atol=1e-6)


def test_sampler_draws_from_the_exact_predictions():
    # the empirical next-token frequencies over a large batch match the mean exact prediction
    ds = _dataset(B=4096)
    batch = ds.batch()
    ids = batch["input_ids"].to(ds.device)
    P = ds.optimal_predictions(ids).double()                                   # (B, T-1, V): row t predicts token t+1
    onehot = torch.nn.functional.one_hot(ids[:, 1:], len(ds.label_tokens)).double()
    assert torch.allclose(onehot.mean((0, 1)), P.mean((0, 1)), atol=3e-3)


@pytest.mark.parametrize("kind_strength", [0.0, 1.0])
def test_entering_a_block_adds_the_same_step_at_any_depth(kind_strength):
    ds = _dataset(kind_strength)
    batch = ds.batch()
    ent = _entries(ds, batch, _code(ds, batch))
    by_key = {}
    for k, n, d, v in ent:
        by_key.setdefault((k, n), []).append((d, v))
    spread = [np.abs(np.array([v for _, v in vs]) - vs[0][1]).max() for vs in by_key.values()
              if len({d for d, _ in vs}) > 1]
    assert len(spread) > 5 and max(spread) < 1e-8


def test_kinds_have_directions_only_when_they_change_the_tokens():
    # null: the step depends on the bound name only (same name, different kind -> identical step);
    # signal: same-kind steps align more than different-kind steps
    for kind_strength in [0.0, 1.0]:
        ds = _dataset(kind_strength)
        batch = ds.batch()
        ent = _entries(ds, batch, _code(ds, batch))
        kinds = np.array([k for k, *_ in ent]); names = np.array([n for _, n, *_ in ent])
        U = np.array([v for *_, v in ent]); U /= np.linalg.norm(U, axis=1, keepdims=True)
        C = U @ U.T
        same_k, same_n = kinds[:, None] == kinds[None], names[:, None] == names[None]
        if kind_strength == 0.0:
            assert np.allclose(C[~same_k & same_n], 1.0)
        else:
            assert C[same_k & ~same_n].mean() > C[~same_k & ~same_n].mean() + 0.1


def test_break_is_likelier_inside_a_loop_only_with_kind_effects():
    for kind_strength, expect in [(0.0, False), (1.0, True)]:
        ds = _dataset(kind_strength)
        base = ds.log_row([])
        inside = ds.log_row([("for", ds.names[0])])
        brk = ds.vocab.index("break")
        gain = (inside[brk] - inside.mean()) - (base[brk] - base.mean())
        assert (gain > 1.0) == expect


def test_blocks_parse_the_nesting_of_a_program():
    # def n0 { for n1 { n1 } if n2 { } }: the module (position 0) holds def, which holds for and if
    ds = _dataset()
    prog = ["def", "n0", "for", "n1", "n1", "end", "if", "n2", "end", "end"]
    ids = [ds.start_token] + [ds.vocab.index(t) for t in prog]
    blocks = ds.blocks(ids)
    assert blocks == [(0, -1, None, None), (2, 0, "def", "n0"), (4, 1, "for", "n1"), (8, 1, "if", "n2")]


def test_bound_names_are_never_reused_within_a_program():
    ds = _dataset()
    toks = ds.batch()["input_ids"].cpu().numpy()
    for s in toks:
        bound = [ds.vocab[s[t + 1]] for t in range(1, len(s) - 1) if ds.vocab[s[t]] in ds.kinds]
        assert len(bound) == len(set(bound))


def test_scopes_is_a_registered_setting():
    cfg = OmegaConf.create({"setting": "scopes", "random_seed": 0, "sample_mode": "sequence",
                            "scope_kwargs": {"n_names": 8, "name_bonus": 4.0, "kind_strength": 1.0, "pool_size": 16},
                            "training_kwargs": {"context_length": 16, "batch_size": 4}})
    assert isinstance(init_dataset(cfg), ScopeDataset)
