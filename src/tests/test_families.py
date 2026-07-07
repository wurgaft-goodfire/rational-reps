"""Distribution families: factoring log p(x|theta) recovers code = natural parameters and
decoder = sufficient statistic (demo / demo_clean §2)."""
import numpy as np

from src.code.data import gaussian1d, gaussian2d, von_mises, categorical, mixture
from src.code.geometry import factor_logcond


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return abs((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def test_gaussian1d_code_is_natural_params_and_decoder_is_hermite():
    L, info = gaussian1d()
    _, E, D, S, _ = factor_logcond(L, drop_base_measure=True)
    assert S[2] < 0.01 * S[0]                                      # rank 2
    # code = natural params (mu/s^2, 1/s^2) up to a linear map
    X = E[:, :2] - E[:, :2].mean(0); Y = info["natural"] - info["natural"].mean(0)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    assert 1 - ((Y - X @ W) ** 2).sum() / (Y ** 2).sum() > 0.99
    # decoder = sufficient statistic (x, x^2) -> Hermite
    x = info["x"]
    assert _corr(D[0], x) > 0.95 and _corr(D[1], x ** 2) > 0.95


def test_von_mises_decoder_is_fourier():
    L, info = von_mises()
    _, E, D, S, _ = factor_logcond(L, drop_base_measure=True)
    assert S[2] < 0.01 * S[0]                                      # rank 2
    x = info["x"]
    for row in (D[0], D[1]):                                       # each row is a Fourier mode
        assert max(_corr(row, np.cos(x)), _corr(row, np.sin(x))) > 0.95


def test_categorical_dirichlet_is_flat_spectrum():
    L, _ = categorical(K=6, n_items=2000)
    _, _, _, S, _ = factor_logcond(L)
    s = S[S > 1e-9 * S[0]]
    assert (s / s.mean()).std() < 0.1                              # flat -> nothing to compress (null)


def test_gaussian2d_is_rank_three():
    L, _ = gaussian2d()
    _, _, _, S, _ = factor_logcond(L, drop_base_measure=True)
    assert S[2] > 0.3 * S[0] and S[3] < 0.01 * S[0]                # (x1, x2, |x|^2)


def test_mixture_stacks_two_families():
    L, _ = mixture()
    _, _, _, S, _ = factor_logcond(L, drop_base_measure=True)
    assert S[3] > 0.02 * S[0] and S[4] < 0.01 * S[0]               # rank 4 = 2 Gaussian + 2 von Mises
