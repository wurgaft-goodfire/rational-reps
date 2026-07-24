"""Analysis helpers for the PCFG geometry experiments: interpretable prefix-state labels, canonical
constituent spans, and unsupervised constituent recovery from a representation (top-down
syntactic-distance bracketing). Pure functions, checked against hand-worked examples."""
import numpy as np

from src.code.pcfg_analysis import (anbn_count, dyck_stack, pl_stack_phase, pl_nesting,
                                     dyck_constituents, nl_constituents,
                                     syntactic_distance_brackets, constituent_recall,
                                     heldout_decode_r2)


# ── interpretable labels ─────────────────────────────────────────────────────
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


# ── canonical constituents ───────────────────────────────────────────────────
def test_dyck_constituents_are_matched_pairs():
    # ( [ ] ) with k=2: open0(0) open1(1) close1(3) close0(2)  -> tokens [0,1,3,2]
    assert dyck_constituents([0, 1, 3, 2], k=2) == {(0, 3), (1, 2)}
    # two siblings: ( ) ( )  -> [0,2,0,2]
    assert dyck_constituents([0, 2, 0, 2], k=2) == {(0, 1), (2, 3)}


def test_nl_constituents_finds_np_vp_pp():
    # det noun verb det noun prep det noun  = [0,2,3,0,2,4,0,2]
    #  subj NP [0,1]; obj NP [3,4]; PP-internal NP [6,7]; PP [5,7]; VP [2,7]
    spans = nl_constituents([0, 2, 3, 0, 2, 4, 0, 2])
    assert (0, 1) in spans and (3, 4) in spans and (6, 7) in spans
    assert (5, 7) in spans                                   # PP = prep + NP
    assert (2, 7) in spans                                   # VP spans verb..end


# ── unsupervised bracketing from a representation ────────────────────────────
def test_syntactic_distance_brackets_splits_at_largest_gap():
    # reps where token 2->3 is the big jump: [a a | b b], nested splits by next-largest gaps
    reps = np.array([[0.0], [0.1], [5.0], [5.1]])
    spans = syntactic_distance_brackets(reps)
    assert (0, 1) in spans and (2, 3) in spans              # the two low-gap pairs are constituents
    assert (0, 3) not in spans                               # root excluded


def test_constituent_recall_perfect_and_random():
    # a representation that jumps exactly at the gold boundaries recovers them all
    reps = np.array([[0.0], [0.1], [5.0], [5.1]])
    gold = {(0, 1), (2, 3)}
    assert constituent_recall(reps, gold) == 1.0
    # a representation whose only jump is after token 0 never brackets (0, 1)
    iso = np.array([[0.0], [5.0], [5.1], [5.2]])
    assert constituent_recall(iso, {(0, 1)}) == 0.0


# ── held-out linear decoding (the capacity-robust discriminator) ─────────────
def test_heldout_decode_r2_recovers_linear_and_rejects_random():
    rng = np.random.default_rng(0)
    G = 60                                                       # distinct prefix groups
    H = rng.standard_normal((G, 10))
    F = H @ rng.standard_normal((10, 3))                         # F linear in H -> decoder generalizes
    groups = np.arange(G)
    assert heldout_decode_r2(H, F, groups, train_frac=0.6, seed=0) > 0.95
    F_rand = rng.standard_normal((G, 3))                         # F independent of H -> it does not
    assert heldout_decode_r2(H, F_rand, groups, train_frac=0.6, seed=0) < 0.3


def test_heldout_decode_r2_respects_group_split():
    # rows of the same group share H and F (as a causal hidden does per prefix); splitting by group
    # keeps test prefixes fully unseen. A linear target is still recovered.
    rng = np.random.default_rng(1)
    G = 40; reps = 3
    Hg = rng.standard_normal((G, 8)); Fg = Hg @ rng.standard_normal((8, 2))
    H = np.repeat(Hg, reps, 0); F = np.repeat(Fg, reps, 0)
    groups = np.repeat(np.arange(G), reps)
    assert heldout_decode_r2(H, F, groups, train_frac=0.6, seed=0) > 0.95


# ── untrained-model control (shape only; the CCA gap is a notebook/slow check) ──
def test_untrained_twin_matches_architecture():
    import torch
    from omegaconf import OmegaConf
    from src.code.config import Config
    from src.code.data import PCFGDataset
    from src.code.pcfg_analysis import untrained_twin
    cfg = Config(OmegaConf.create({
        "setting": "pcfg", "grammar": "dyck", "grammar_kwargs": {"k": 2, "depth": 3},
        "max_len": 6, "sample_mode": "sequence", "random_seed": 0,
        "model_kwargs": {"model_type": "transformer", "model_config": {
            "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
            "intermediate_size": 64}},
        "training_kwargs": {"batch_size": 8, "context_length": 40},
        "eval_kwargs": {"num_eval_batches": 1, "output_hidden_states": True},
    }))
    ds = PCFGDataset(cfg)
    model = untrained_twin(cfg, ds)
    ids = torch.zeros(2, ds.seq_len, dtype=torch.long, device=ds.device)
    out = model(ids, output_hidden_states=True)
    assert out["hidden_states"][-1].shape == (2, ds.seq_len, 32)
