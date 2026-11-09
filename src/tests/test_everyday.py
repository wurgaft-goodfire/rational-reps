"""Everyday predictions (Griffiths & Tenenbaum 2006) asked as remaining time: the world-prior kernel
log p(R | t), the prompts, and the reader that pulls Llama's states and answer distribution."""
import numpy as np
import pytest
import torch

from src.code.everyday import DOMAINS, encode, number_tokens, prompts, remaining_logcond
from src.code.llm import read_prompts
from src.code.geometry import predicted_geometry

LLAMA = "meta-llama/Llama-3.1-8B"


def _median(L, R):
    """Median of each row's distribution ``exp(L)`` over the grid ``R``."""
    cdf = np.cumsum(np.exp(L), 1)
    return R[(cdf < 0.5).sum(1)]


def test_erlang_kernel_is_the_same_for_every_t_so_the_code_collapses():
    """The Erlang prior's factor T cancels the 1/T likelihood: p(R | t) is one exponential for every t."""
    for d in ("pharaohs", "house"):
        t, R, L = remaining_logcond(DOMAINS[d])
        assert np.allclose(L, L[0], atol=1e-9)
        E = predicted_geometry(L)
        assert np.allclose(E, E[0], atol=1e-6)                             # every t at one point


def test_gaussian_and_power_law_codes_are_one_dimensional_in_t_and_log_t():
    """Gaussian: the t*R term of the log prior is bilinear, so the code is a line uniform in t.
    Power law: the posterior depends on R / t, so equal ratios of t sit equal distances apart."""
    t, R, L = remaining_logcond(DOMAINS["lifespan"])
    E = predicted_geometry(L)
    S = np.linalg.svd(E - E.mean(0), compute_uv=False)
    assert S[0] ** 2 / (S ** 2).sum() > 0.95
    step = np.linalg.norm(np.diff(E, axis=0), axis=1)
    assert step.max() / step.min() < 2.0

    t, R, L = remaining_logcond(DOMAINS["grosses"])
    E = predicted_geometry(L); at = {v: E[i] for i, v in enumerate(t)}
    assert np.linalg.norm(at[10] - at[20]) == pytest.approx(np.linalg.norm(at[20] - at[40]), rel=0.25)


def test_medians_follow_the_three_prediction_rules():
    """Erlang: R = beta ln 2 for every t. Power law: T = 2^(1/gamma) t. Gaussian: T about the same for a man
    of 18, 39 or 51, a little under the mean (the 1/T likelihood pulls it down by about sigma^2 / mu)."""
    t, R, L = remaining_logcond(DOMAINS["pharaohs"])
    assert np.all(np.abs(_median(L, R) - DOMAINS["pharaohs"]["prior"]["beta"] * np.log(2)) <= 1)
    t, R, L = remaining_logcond(DOMAINS["grosses"])
    big = t >= 20
    assert np.allclose((t + _median(L, R))[big] / t[big], 2 ** (1 / 1.71), rtol=0.05)
    t, R, L = remaining_logcond(DOMAINS["lifespan"])
    T = t + _median(L, R)
    young = T[np.isin(t, [18, 39, 51])]
    assert young.max() - young.min() <= 3 and np.all((70 <= young) & (young <= 75))


def test_prompts_end_before_the_answer_and_point_at_t():
    """With Llama's tokenizer: ``at`` is the token of t's digits, the prompt ends on the space token
    that precedes a number, and 0..999 are single tokens."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLAMA)
    nums = number_tokens(tok)
    assert len(nums) == 1000 and len(set(nums)) == 1000
    for d in DOMAINS:
        for q in ("remaining", "total"):
            for template in prompts(d, q):
                for t, text in list(template)[:: 10]:
                    ids, at = encode(tok, text)
                    assert ids[0] == tok.bos_token_id
                    assert tok.decode([ids[at]]) == str(t)
                    assert tok.decode([ids[-1]]) == " "


@pytest.mark.slow
def test_llama_expects_fewer_remaining_years_for_an_older_man():
    """Outer loop: real prompts through Llama-3.1-8B. Read shapes are (layers, prompts, d), and the
    median remaining years drops from age 20 to age 80."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLAMA)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.bfloat16, device_map="cuda").eval()
    texts = dict(prompts("lifespan", "remaining")[0])
    rows = [encode(tok, texts[a]) for a in (20, 80)]
    out = read_prompts(model, rows, number_tokens(tok), pad_id=tok.eos_token_id)
    assert out["last"].shape == (33, 2, 4096) and out["at_t"].shape == (33, 2, 4096)
    med = _median(out["logq"], np.arange(1000))
    assert med[0] > med[1] + 20
