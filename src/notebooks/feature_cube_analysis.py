# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: template-lm-training-codebase (3.13.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Superposition on the feature cube
#
# **The setting is the standard analogy case plus a sparse item measure.** Items are the
# $2^m$ vertices of the hypercube (one binary feature per axis); co-occurrence is the heat
# kernel at time $\gamma$ — literally `graph_conditional(structure=hypercube,
# cooccurrence="heat")`, the kernel that makes the log-conditional additive over features and
# produces linear analogies. Two deltas turn it into a superposition testbed:
#
# 1. **Sparse item measure**: feature $i$ is active with probability
#    $p_i = \text{sparsity} \cdot \text{p\_decay}^i$. Importance = frequency (the CE-native
#    loss weight); the kernel stays uniform diffusion. The measure only decides *which items'
#    errors matter*.
# 2. **A code budget** $d < m$.
#
# Because the kernel is symmetric, each feature of a neighbor is an independent copy that is
# wrong with probability $f = (1 - e^{-2\gamma})/2$, and in $\pm 1$ codes
#
# $$\log p(x' \mid x) = \text{const} + w \sum_i s_i s_i', \qquad w = \tfrac12\log\tfrac{1-f}{f}$$
#
# — **purely bilinear**: no outcome field, so *no bias is needed anywhere*, and every model
# in the comparison (oracles included) is a bias-free code $\times$ matrix readout. $\gamma$
# is the task temperature: small $\gamma$ = near-copy transitions = sharp conditionals
# (softmax-saturation regime); large $\gamma$ = diffuse. The analogy work runs at
# $\gamma = 0.8$ ($f \approx 0.40$) — the flat end.
#
# Two rungs at budget $d$ (`src/code/geometry.py`, both `fit_bias=False`):
# **closed form** (rung 3′, `factor_features`) — whitened truncation, the quadratic account
# that sees only second moments; **exact CE** (rung 4′, `als_features`) — alternating convex
# halves. The closed form can be a *stable local minimum* of the exact CE, so ALS runs with
# random restarts. The deviation between the rungs is the superposition signal, and the
# prediction is a diagonal phase boundary in (sharpness $\times$ sparsity).

# %%
import os, sys
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(sys.prefix).parent            # <repo>/.venv -> <repo>
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.data import feature_cube
from src.code.geometry import (factor_features, als_features, expected_ce, principal_angles,
                               _psd_sqrt)
from src.code import plot_utils
import matplotlib.pyplot as plt

QUICK = bool(int(os.environ.get("QUICK", "0")))   # QUICK=1: coarse grids + short fits/training
FIGDIR = Path("outputs/feature_cube") / ("figures_quick" if QUICK else "figures")
FIGDIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

M, D_BUDGET, P_DECAY = 6, 2, 0.7
RIDGE = 1e-4
N_RESTARTS, N_ROUNDS = (1, 20) if QUICK else (2, 40)


@lru_cache(maxsize=None)
def cube(sparsity, gamma, m=M):
    """The cube at heat time gamma with the frequency ladder p_i = sparsity * P_DECAY^i."""
    return feature_cube(sparsity * P_DECAY ** np.arange(m), gamma)


def best_als(Lf, Z, d, w, cf, fit_bias=False, n_random=N_RESTARTS, n_rounds=N_ROUNDS):
    """ALS from the closed-form warm start plus random restarts; lowest exact CE wins.
    Returns (fit, ce_warm, ce_best) — the warm-start CE is kept to expose the local minimum."""
    m, nx = Lf.shape
    kw = dict(w=w, ridge=RIDGE, n_rounds=n_rounds, fit_bias=fit_bias)
    fits = [als_features(Lf, Z, d, init=cf[:3], **kw)]
    for _ in range(n_random):
        init = (rng.normal(size=(d, m)), rng.normal(size=(d, nx)), np.zeros(nx))
        fits.append(als_features(Lf, Z, d, init=init, **kw))
    ces = [f[3][-1] for f in fits]                                # ridge-free CE at the solution
    return fits[int(np.argmin(ces))], ces[0], min(ces)


@lru_cache(maxsize=None)
def oracle(sparsity, gamma):
    """Both rungs at one phase-plane cell, memoized so every figure shares the same fits.
    Returns (pi, Z, Lf, cf, fit, ex_cf, ex_warm, ex_best) with excesses over Bayes."""
    logP, pi, Z, Lf = cube(sparsity, gamma)
    cf = factor_features(Lf, Z, D_BUDGET, w=pi, fit_bias=False)
    ce_cf, bayes = expected_ce(Z, Lf, *cf[:3], w=pi)
    fit, ce_warm, ce_best = best_als(Lf, Z, D_BUDGET, w=pi, cf=cf)
    return pi, Z, Lf, cf, fit, ce_cf - bayes, ce_warm - bayes, ce_best - bayes

# %% [markdown]
# ## Figure 1 — what the quadratic account sees
#
# Left: the whitened spectrum `factor_features` truncates. The frequency ladder is the only
# thing distinguishing features, and at second order sharpness ($\gamma$) and sparsity both
# just rescale this picture — the two knobs are indistinguishable to the closed form.
# Everything rung 4′ does beyond truncating this spectrum is invisible at second order.
# Right: the sparse-sharp log-conditional itself.

# %%
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for gamma, ls in [(0.6, "--"), (0.03, "-")]:
    for sp, color in [(0.5, plot_utils.FIG.colors["blue"]), (0.1, plot_utils.FIG.colors["orange"])]:
        logP, pi, Z, Lf = cube(sp, gamma)
        S = factor_features(Lf, Z, d=M, w=pi, fit_bias=False)[3]
        ax[0].semilogy(np.arange(1, M + 1), S, ls, marker="o", color=color,
                       label=rf"sparsity ${sp}$, $\gamma={gamma}$")
ax[0].set_xlabel("mode"); ax[0].set_ylabel("whitened singular value")
ax[0].legend(fontsize=9); ax[0].set_title("the spectrum rung 3' truncates")

logP, pi, Z, Lf = cube(0.1, 0.03)
plot_utils.conditional_heatmap(ax[1], logP, title=r"$\log p(x'\mid x)$, sparse-sharp cube",
                               xlabel="", ylabel="")
plot_utils.FIG.style_axes(ax[0])
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig1_spectrum.png")

# %% [markdown]
# ## Figure 2 — the phase plane: truncation → packing in (sharpness × sparsity)
#
# For each cell (everything bias-free): closed form, warm-started ALS, best-of-restarts ALS,
# and best-of-restarts ALS *granted an output bias*. Four readouts of the same object:
#
# - **best/cf** (top left): the phase signal. 1 = truncation optimal; small = packing wins.
# - **warm/cf** (top right): the local-minimum trap — where warm-started ALS fails to leave
#   the closed form even though the optimum has moved.
# - **max principal angle** (bottom left) between the Fisher-whitened row spaces of the
#   fitted logit maps.
# - **bias ablation** (bottom right): best-with-bias / best-bias-free. The target has no
#   constant component (the kernel is symmetric), so any gain from the bias here is the bias
#   acting as an *interference sink* for packing — a resource, not a correction.
#
# Sharpness increases rightward (decreasing $\gamma$); sparsity increases downward. Results
# cached to `phase_plane.npz`; delete to recompute.

# %%
GAMMA_GRID = [0.6, 0.03] if QUICK else [0.6, 0.25, 0.1, 0.03]     # flat -> sharp
SPARSITY_GRID = [0.5, 0.1] if QUICK else [0.5, 0.25, 0.1]
cache = FIGDIR.parent / ("phase_plane_quick.npz" if QUICK else "phase_plane.npz")
if cache.exists():
    dat = np.load(cache)
    ratio_best, ratio_warm, angle, bias_gain = (dat["ratio_best"], dat["ratio_warm"],
                                                dat["angle"], dat["bias_gain"])
else:
    shape = (len(SPARSITY_GRID), len(GAMMA_GRID))
    ratio_best, ratio_warm, angle, bias_gain = (np.zeros(shape) for _ in range(4))
    for i, sp in enumerate(SPARSITY_GRID):
        for j, gamma in enumerate(GAMMA_GRID):
            pi, Z, Lf, cf, fit, ex_cf, ex_warm, ex_best = oracle(sp, gamma)
            cf_bias = factor_features(Lf, Z, D_BUDGET, w=pi)
            _, _, ce_bias = best_als(Lf, Z, D_BUDGET, w=pi, cf=cf_bias, fit_bias=True)
            _, bayes = expected_ce(Z, Lf, *cf[:3], w=pi)
            q = pi @ np.exp(cube(sp, gamma)[0])
            Fh = _psd_sqrt(np.diag(q) - np.outer(q, q))
            ang = principal_angles((cf[0].T @ cf[1]) @ Fh, (fit[0].T @ fit[1]) @ Fh).max()
            ratio_best[i, j] = ex_best / ex_cf
            ratio_warm[i, j] = ex_warm / ex_cf
            angle[i, j] = np.degrees(ang)
            bias_gain[i, j] = (ce_bias - bayes) / ex_best
            print(f"sparsity={sp:4.2f} gamma={gamma:4.2f}  exCE cf={ex_cf:7.4f}  "
                  f"warm={ex_warm:7.4f}  best={ex_best:7.4f}  "
                  f"with-bias={ce_bias - bayes:7.4f}  ratio={ratio_best[i, j]:.3f}")
    np.savez(cache, ratio_best=ratio_best, ratio_warm=ratio_warm, angle=angle,
             bias_gain=bias_gain)

fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
panels = [(ratio_best, "excess-CE ratio  best ALS / closed form", 0, 1, "viridis_r"),
          (ratio_warm, "warm-started / closed form  (the local-minimum trap)", 0, 1, "viridis_r"),
          (angle, "max principal angle (deg)", 0, 90, "viridis"),
          (bias_gain, "bias ablation:  with-bias / bias-free  (interference sink)", 0, 1, "viridis_r")]
for a, (Mx, title, vmin, vmax, cmap) in zip(ax.ravel(), panels):
    im = a.imshow(Mx, cmap=cmap, vmin=vmin, vmax=vmax)
    a.set_xticks(range(len(GAMMA_GRID)), [f"{g:g}" for g in GAMMA_GRID])
    a.set_yticks(range(len(SPARSITY_GRID)), [f"{s:g}" for s in SPARSITY_GRID])
    a.set_xlabel(r"heat time  $\gamma$  (sharper $\to$)")
    a.set_ylabel(r"feature activity scale  (sparser $\downarrow$)")
    a.set_title(title, fontsize=10.5)
    for i in range(len(SPARSITY_GRID)):
        for j in range(len(GAMMA_GRID)):
            a.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=9,
                   color="white" if (Mx[i, j] - vmin) / (vmax - vmin) < 0.5 else "black")
    fig.colorbar(im, ax=a, shrink=0.85)
fig.suptitle(rf"phase plane at $m={M}$, $d={D_BUDGET}$ (frequency ladder decay {P_DECAY}; "
             "all fits bias-free unless noted)", fontsize=12.5)
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig2_phase_plane.png")

# %% [markdown]
# ## Figure 3 — the geometry at the corners of the plane
#
# Feature code vectors (columns of $A$, native $d=2$ coordinates — no projection) for the
# closed form and the best bias-free ALS at four corners. Truncation keeps the top of the
# frequency ladder everywhere; the CE optimum leaves it in the sharp column, spreading rare
# features into their own directions.

# %%
CORNERS = [(0.5, 0.6), (0.5, 0.03), (0.1, 0.6), (0.1, 0.03)]
fig, ax = plt.subplots(2, len(CORNERS), figsize=(3.4 * len(CORNERS), 7.4))
for col, (sp, gamma) in enumerate(CORNERS):
    pi, Z, Lf, cf, fit, ex_cf, _, ex_best = oracle(sp, gamma)
    for row, (A, name, ex) in enumerate([(cf[0], "closed form", ex_cf),
                                         (fit[0], "ALS (best)", ex_best)]):
        a = ax[row, col]
        plot_utils.feature_arrows(a, A.T, lw=2, fontsize=10.5)
        a.set_title(rf"{name}:  excess = {ex:.3f}", fontsize=10.5)
    ax[0, col].set_title(rf"sparsity ${sp}$, $\gamma={gamma}$" + "\n"
                         + ax[0, col].get_title(), fontsize=10.5)
fig.suptitle(rf"feature codes at $d={D_BUDGET}$  (1 = most frequent; native coordinates, "
             "no bias)", fontsize=13)
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig3_oracle_geometry.png")

# %% [markdown]
# ## Figure 4 — trained architectures on the sharp column
#
# The standard pipeline, unchanged: k-linear and MLP on pointwise item→neighbor pairs, GRU
# and transformer on short episodes from the sparse marginal — one token per item, every
# position on-task. **Every readout in the ladder is a pure matrix now** (klinear tensor,
# MLP final layer, GRU head, GPTNeoX unembedding — all bias-free), exactly the class the
# oracles fit. Codes are read off with the standard `evaluator.item_representations`; each
# model's excess CE is computed on every item under the item measure, exactly the oracle's
# objective (walk models renormalized over the item block — the start token is never a
# target).

# %%
import torch
from src.code.train import train_inline

D_ARCH = str(D_BUDGET)
ARCH = {
    "klinear": ["runner=klinear", f"model_kwargs.model_config.embed_dim={D_ARCH}",
                "training_kwargs.train_steps=6000"],
    "mlp": ["runner=mlp", f"model_kwargs.model_config.hidden_size={D_ARCH}",
            "model_kwargs.model_config.num_layers=1", "model_kwargs.model_config.init_scale=0.5",
            "training_kwargs.learning_rate=0.003", "training_kwargs.train_steps=8000"],
    "rnn": ["runner=rnn", f"model_kwargs.model_config.hidden_size={D_ARCH}",
            "training_kwargs.learning_rate=0.002", "training_kwargs.train_steps=12000"],
    "transformer": ["runner=transformer", f"model_kwargs.model_config.hidden_size={D_ARCH}",
                    "model_kwargs.model_config.num_attention_heads=1",
                    "model_kwargs.model_config.intermediate_size=16",
                    "training_kwargs.learning_rate=0.001", "training_kwargs.train_steps=12000"],
}
POINTS = [(0.5, 0.03), (0.1, 0.03)]                               # dense-sharp, sparse-sharp
if QUICK:                                                          # readout heads only, short runs
    ARCH = {"klinear": ARCH["klinear"][:-1] + ["training_kwargs.train_steps=2000"],
            "mlp": ARCH["mlp"][:-1] + ["training_kwargs.train_steps=3000"]}


def item_excess_ce(model, ds):
    """Excess CE (nats) over Bayes on every item, weighted by the item measure."""
    device = next(model.parameters()).device
    ids = torch.arange(ds.n_items, device=device)
    with torch.no_grad():
        if model.config.vocab_size == ds.n_items + 1:             # walk models: [start, item]
            start = torch.full((ds.n_items, 1), int(ds.start_token), device=device)
            logits = model(torch.cat([start, ids.unsqueeze(1)], dim=1))["logits"][:, -1]
            logits = logits[:, :ds.n_items]                       # start is never a target
        else:
            logits = model(ids.unsqueeze(1))["logits"]
    logq = torch.log_softmax(logits.double(), dim=-1).cpu().numpy()
    P = np.exp(ds.conditional)
    ce = -(ds.pi[:, None] * P * logq).sum()
    bayes = -(ds.pi[:, None] * P * ds.conditional).sum()
    return float(ce - bayes)


def feature_vectors(H, Z, w):
    """Weighted least-squares additive fit H ~ [Z, 1] W: effective feature vectors (m, d)."""
    X = np.column_stack([Z, np.ones(len(Z))]) * np.sqrt(w)[:, None]
    return np.linalg.lstsq(X, np.sqrt(w)[:, None] * np.asarray(H, float), rcond=None)[0][:-1]


arch_results = {}
for sp, gamma in POINTS:
    pi, Z, Lf, cf, fit, ex_cf, _, ex_best = oracle(sp, gamma)
    entry = {"closed form": (ex_cf, Z @ cf[0].T),
             "ALS": (ex_best, Z @ fit[0].T)}
    base = ["setting=feature_cube", f"n_features={M}", f"sparsity={sp}", f"gamma={gamma}",
            f"p_decay={P_DECAY}"]
    for name, ov in ARCH.items():
        model, ds, evaluator, *_ = train_inline(base + ov)
        H = evaluator.item_representations(model)
        entry[name] = (item_excess_ce(model, ds), np.asarray(H, float))
        print(f"sparsity={sp:4.2f} gamma={gamma:4.2f} {name:12s} "
              f"excess CE={entry[name][0]:.4f}  (cf {ex_cf:.4f} / ALS {ex_best:.4f})")
    arch_results[(sp, gamma)] = (entry, pi, Z)

# %% [markdown]
# ## Figure 5 — geometry over RDMs, per model
#
# The house comparison layout: each model's effective feature vectors (weighted additive fit
# of its item codes; exactly the $A$ columns for the oracles) over its item-code RDM (all
# $2^6$ items, lexicographic). Truncation pins the rare features at the origin; the packed
# solutions give every feature a direction.

# %%
names = ["closed form", "ALS"] + list(ARCH)
fig = plt.figure(figsize=(2.7 * len(names), 5.6 * len(POINTS)))
gs = fig.add_gridspec(2 * len(POINTS), len(names), hspace=0.45, wspace=0.3)
for prow, (sp, gamma) in enumerate(POINTS):
    entry, pi, Z = arch_results[(sp, gamma)]
    for col, n in enumerate(names):
        ex, H = entry[n]
        V = feature_vectors(H, Z, pi)
        V2 = V if V.shape[1] == 2 else V @ np.linalg.svd(V, full_matrices=False)[2][:2].T
        a = fig.add_subplot(gs[2 * prow, col])
        plot_utils.feature_arrows(a, V2, scale=1.3, fontsize=9.5)
        a.set_title(f"{n}\nexcess CE {ex:.3f}", fontsize=10)
        if col == 0:
            a.set_ylabel(rf"sparsity ${sp}$, $\gamma={gamma}$", fontsize=11)
        plot_utils.rdm_heatmap(fig.add_subplot(gs[2 * prow + 1, col]), entry[n][1], title="")
fig.suptitle("effective feature vectors (top plane) over item-code RDMs", fontsize=13)
plot_utils.save_figure(fig, FIGDIR / "fig5_geometry_rdms.png")

# %% [markdown]
# ## Summary
#
# - The setting is the analogy case + a sparse item measure + a budget: same hypercube, same
#   heat kernel (`gamma`, the knob the analogy work fixes at $0.8$), same conditional code
#   path. At full rank nothing changes (additive Walsh code, parallelograms); the phase
#   structure lives entirely in the gap the budget opens.
# - The log-conditional is purely bilinear in $\pm 1$ codes — no outcome field — so **no
#   model in the comparison has a bias**: klinear tensor, MLP/GRU heads, GPTNeoX unembedding,
#   and both oracle rungs are all pure matrix readouts of a code.
# - The phase plane (Figure 2): truncation optimal in the flat column, packing in the sharp
#   column — and within the *bias-free* class the ratio is set almost entirely by sharpness
#   (~0.99 → ~0.68 in every sparsity row). Sparsity does not move the truncation→packing
#   boundary here; its co-activation advantage (interference $\propto p_i p_j$, benefit
#   $\propto p_i$) is instead harvested by the **bias channel**: the with-bias/bias-free
#   ratio (bottom right) improves from 0.59 (dense-sharp) to 0.43 (sparse-sharp) — the
#   interference-sink value of an output bias grows with sparsity.
# - Warm-started ALS ≈ best-of-restarts everywhere in this class: the truncation trap seen
#   in earlier (bias-ful / asymmetric-kernel) formulations is absent.
# - Trained k-linear *beats* restart-ALS at both sharp points (Figures 4–5): SGD from small
#   init finds better bias-free optima than the L-BFGS alternation, so the trained linear
#   model is currently the tightest rung-4′ approximation and the ALS numbers are upper
#   bounds on the optimal excess, not the optimum.

# %%
