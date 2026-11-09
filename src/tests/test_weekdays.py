"""Uncertainty over weekdays: prompts that state a distribution over days, reads at the sentence's
final period (Spokes, Not a Ring) and at an appended answer slot, the ring's harmonic coordinates,
and the three hypotheses for where an uncertain prompt lands."""
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from src.code.weekdays import (DAYS, all_prompts, day_distribution, day_token_ids, fit_predict,
                               harmonic_basis, read_prompts, ring_coords, r2, spoke_predict)

LLAMA = "meta-llama/Llama-3.1-8B"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(LLAMA)


def test_prompts_state_their_weights_and_end_the_sentence_before_the_answer_slot():
    """Every prompt's sentence ends in a period, the answer slot ends in " on", and the stated
    weights are a distribution over the seven days: two days for the adverb and percent families
    (5 and 7 levels over the 42 ordered pairs one to three days apart), a contiguous set of 1-7
    equally likely days for the set family."""
    P = all_prompts()
    for p in P:
        assert p["pre"].endswith(".") and p["post"].endswith(" on") and p["post"].startswith(" ")
        assert np.isclose(p["weights"].sum(), 1) and (p["weights"] >= 0).all()
    adverb = [p for p in P if p["family"] == "adverb"]
    assert len({p["level"] for p in adverb}) == 5 and len({(p["A"], p["B"]) for p in adverb}) == 42
    assert {p["sep"] for p in adverb} == {1, 2, 3}
    p = next(p for p in adverb if p["A"] == 0 and p["B"] == 2 and p["level"] == 1 and p["cont"] == 0)
    assert p["pre"] == "Usually on Monday, sometimes on Wednesday, her flight leaves."
    assert p["post"] == " This week, her flight leaves on"
    assert np.allclose(p["weights"], [.75, 0, .25, 0, 0, 0, 0])
    sets = [p for p in P if p["family"] == "set"]
    s = next(p for p in sets if p["size"] == 3 and p["start"] == 5 and p["cont"] == 0)
    assert "Saturday, Sunday, or Monday" in s["pre"]
    assert np.allclose(s["weights"], np.array([1, 0, 0, 0, 0, 1, 1]) / 3)
    assert {p["size"] for p in sets} == set(range(1, 8))


def test_day_token_ids_are_the_single_token_spellings(tok):
    ids = day_token_ids(tok)
    assert len(ids) == 7 and ids[0] == [tok.encode(w, add_special_tokens=False)[0] for w in (" Monday", "Monday", " monday")]


def test_day_distribution_sums_each_days_spellings_and_keeps_the_rest_as_other():
    logits = torch.log(torch.tensor([[.1, .2, .3, .05, .35]]))
    q = day_distribution(logits, [[0, 1], [3]])
    assert np.allclose(q, [[.3, .05, .65]])


def test_reads_match_one_prompt_at_a_time(tok):
    """Batched, right-padded reads equal unbatched passes: the residual at the period (``mark``)
    and at the last token (``answer``) for every layer, and the day distribution at each."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2,
                                         intermediate_size=64)).eval()
    P = all_prompts()[::397][:5]
    out = read_prompts(model, tok, P, batch_size=3)
    days = day_token_ids(tok)
    for i, p in enumerate(P):
        pre = tok(p["pre"]).input_ids
        ids = torch.tensor([pre + tok(p["post"], add_special_tokens=False).input_ids])
        with torch.no_grad():
            res = model(ids, output_hidden_states=True)
        H = torch.stack(res["hidden_states"])[:, 0]
        assert np.allclose(out["mark"][:, i], H[:, len(pre) - 1].numpy(), atol=2e-3)
        assert np.allclose(out["answer"][:, i], H[:, -1].numpy(), atol=2e-3)
        assert np.allclose(out["q_mark"][i], day_distribution(res["logits"][0, [len(pre) - 1]], days)[0], atol=1e-5)
        assert np.allclose(out["q_answer"][i], day_distribution(res["logits"][0, [-1]], days)[0], atol=1e-5)


def test_harmonic_coordinates_put_any_seven_centroids_on_the_unit_circle_at_their_day():
    """The ring's coordinates are the centroids' discrete Fourier vectors: whatever the centroids'
    shape, day d sits at radius 1 and angle d (in days) in the first harmonic."""
    rng = np.random.default_rng(0)
    C = rng.normal(size=(7, 20))
    mu, B = harmonic_basis(C)
    angle, radius = ring_coords(C, mu, B)
    assert np.allclose(radius, 1) and np.allclose(angle, np.arange(7))
    angle, radius = ring_coords(mu[None], mu, B)
    assert np.allclose(radius, 0)


def test_a_planted_log_code_is_recovered_from_certain_prompts_and_beats_the_alternatives():
    """If the representation is a linear image of the centred log-probabilities, the map fitted on
    near-certain prompts predicts two-day mixtures exactly; a map fitted on probabilities, and the
    best one-spoke prediction, do worse. The certain answers' log-kernel must carry every harmonic
    (a peaked, Laplace-like falloff does; a von Mises, pure harmonic 1, would leave the map
    unidentified off the ring plane)."""
    rng = np.random.default_rng(1)
    dist = np.minimum(np.arange(7), 7 - np.arange(7))
    def vm(d, k=4):
        p = np.exp(-k * dist[(np.arange(7) - d) % 7] / 2); return np.r_[0.9 * p / p.sum(), 0.1]
    cert = np.stack([vm(d, k) for d in range(7) for k in (3, 4, 5)])
    mix = np.stack([w * vm(a) + (1 - w) * vm((a + s) % 7) for a in range(7) for s in (1, 2, 3) for w in (.5, .75)])
    M = rng.normal(size=(8, 30))
    clr = lambda q: np.log(q) - np.log(q).mean(1, keepdims=True)
    Hc, Hm = clr(cert) @ M, clr(mix) @ M
    assert r2(Hm, fit_predict(cert, Hc, mix, "log")) > 0.999
    assert r2(Hm, fit_predict(cert, Hc, mix, "prob")) < 0.99
    C = np.stack([Hc[3 * d:3 * d + 3].mean(0) for d in range(7)])
    assert r2(Hm, spoke_predict(Hm, mix, C, Hc.mean(0))) < 0.99


@pytest.mark.slow
def test_llama_holds_a_two_day_belief_at_the_answer_slot_but_not_at_the_period(tok):
    """Integration: for "Sometimes on Monday, sometimes on Thursday, ..." Llama-3.1-8B splits its
    answer between the two days at the answer slot, while at the period (where Spokes read) almost
    none of the next-token mass is on a day."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.bfloat16, device_map="cuda").eval()
    P = [p for p in all_prompts() if p["family"] == "adverb" and p["level"] == 2 and p["A"] == 0 and p["B"] == 3]
    out = read_prompts(model, tok, P, batch_size=8)
    qa = out["q_answer"].mean(0)
    assert qa[0] > 0.15 and qa[3] > 0.15
    assert out["q_mark"][:, :7].sum(1).mean() < 0.1


def test_pair_position_is_zero_on_As_spoke_one_on_Bs_and_follows_the_short_way_round():
    """``w`` measures the angle from A toward B along the short arc, in units of their separation,
    whichever naming order the prompt used."""
    from src.code.weekdays import pair_position
    A, B, sep = np.array([0, 2, 6]), np.array([2, 0, 1]), np.array([2, 2, 2])
    assert np.allclose(pair_position(np.array([1.0, 1.0, 0.5]), A, B, sep), [.5, .5, .75])
    assert np.allclose(pair_position(np.array([0.0, 0.0, 1.0]), A, B, sep), [0, 1, 1])
