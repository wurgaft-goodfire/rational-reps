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
# # Word → attribute: frequency vs importance under a rank budget
#
# Words are the $2^m$ corners of the feature cube; outcomes are the $m$ attribute tokens
# plus a null ("nothing said") reference that pins the softmax gauge:
# $\log p(i|x)/p(\text{null}|x) = \alpha_i x_i + \tilde\beta_i$. The knobs are decoupled by
# construction — $\alpha_i = \alpha_0 \cdot 0.9^i$ (**reliability/importance**: the log-odds
# swing of emitting attribute $i$ between words that have vs lack it) and
# $\tilde\beta_i = -i\,\Delta\beta$ (**frequency/salience**: the attribute's base rate, blind
# to the word). The pinned-gauge table is rank $m$, so the budget question is $d < m$.
#
# Oracles, both in the trained model's own class (free per-item codes, bias-free readout):
# **truncation** (`factor_logcond`, rank-$d$ prefix) and **nuclear**
# (`nuclear_features(logP, I, d)`, the exact-CE optimum). Dropout is read from the fitted
# per-attribute swing $\hat\alpha_i$ (`attribute_stats`), not readout norms — at steep
# $\tilde\beta$ a rare attribute needs a large readout just to carry its intercept, so norms
# conflate signal with frequency.

# %%
import os, sys
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(sys.prefix).parent            # <repo>/.venv -> <repo>
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import torch
from src.code.data import word_attribute
from src.code.geometry import (factor_logcond, nuclear_features, expected_ce,
                               attribute_stats, feature_vectors, superposition_measure)
from src.code.train import train_inline
from src.code import plot_utils
import matplotlib.pyplot as plt

FIGDIR = Path("outputs/word_attribute/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

M, GAMMA, ALPHA_DECAY = 8, 0.03, 0.9
BETA_STEPS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
D_GRID = [2, 3, 4]
D_PANEL = 3


@lru_cache(maxsize=None)
def oracle(beta_step, d):
    """Truncation and nuclear fits at (beta_step, d). Returns a dict with the task, the two
    codes/readouts, excesses over Bayes, and fitted per-attribute (swing, intercept)."""
    logP, info = word_attribute(M, GAMMA, ALPHA_DECAY, beta_step)
    I = np.eye(len(logP))
    zeros = np.zeros(logP.shape[1])
    _, E, Df, S, _ = factor_logcond(logP)
    A_cf, D_cf = E[:, :d].T, Df[:d]
    ce_cf, bayes = expected_ce(I, logP, A_cf, D_cf, zeros)
    A, D, b, _ = nuclear_features(logP, I, d, fit_bias=False)
    ce_nuc, _ = expected_ce(I, logP, A, D, b)
    return dict(logP=logP, **info, bayes=bayes,
                cf=dict(E=A_cf.T, D=D_cf, ex=ce_cf - bayes,
                        stats=attribute_stats(A_cf.T @ D_cf, info["Z"])),
                nuc=dict(E=A.T, D=D, ex=ce_nuc - bayes,
                         stats=attribute_stats(A.T @ D, info["Z"])))

# %% [markdown]
# ## Figure 1 — excess CE across the frequency sweep
#
# Steepening the frequency ladder collapses the excess for every rung: rare attributes carry
# vanishing Fisher weight, so the effective task drops in rank. Nuclear ≤ truncation
# throughout; the packing gain is modest here (softmax over m+1 outcomes couples errors far
# less brutally than the cube's 2^m product kernel).

# %%
fig, ax = plt.subplots(figsize=(5.4, 3.8))
for d in D_GRID:
    ex_cf = [oracle(bs, d)["cf"]["ex"] for bs in BETA_STEPS]
    ex_nuc = [oracle(bs, d)["nuc"]["ex"] for bs in BETA_STEPS]
    (ln,) = ax.plot(BETA_STEPS, ex_nuc, marker="o", ms=4, label=f"nuclear d={d}")
    ax.plot(BETA_STEPS, ex_cf, marker="s", ms=4, ls="--", color=ln.get_color(),
            label=f"truncation d={d}")
ax.set_yscale("log")
ax.set_xlabel(r"frequency-ladder step $\Delta\beta$")
ax.set_ylabel("excess CE (nats)")
ax.legend(fontsize=8, ncol=2)
plot_utils.FIG.style_axes(ax)
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig1_excess_sweep.png")

# %% [markdown]
# ## Figure 2 — dropout: fitted swing per attribute across the sweep
#
# $\hat\alpha_i / \alpha_i$ (1 = fully represented, 0 = dropped). At $\Delta\beta = 0$ the
# tail is *graded* — partial superposed representation, no hard dropout (the feature-cube
# result reproduced in the asymmetric task). As $\Delta\beta$ steepens the front sharpens:
# rare attributes drop toward zero while the frequent ones stay exact. Note the mechanism
# (see fig 1 and the summary): this is Fisher-reweighted truncation, not features exiting
# superposition — $\tilde\beta$ prices attributes like an importance lever.

# %%
fig, ax = plt.subplots(2, len(D_GRID), figsize=(3.6 * len(D_GRID), 6.6),
                       sharex=True, sharey=True)
for col, d in enumerate(D_GRID):
    for row, which in enumerate(["cf", "nuc"]):
        rel = np.array([oracle(bs, d)[which]["stats"][0] / oracle(bs, d)["alpha"]
                        for bs in BETA_STEPS])
        for j in range(M):
            ax[row, col].plot(BETA_STEPS, rel[:, j], marker="o", ms=3, label=f"{j + 1}")
        ax[row, col].set_title(f"{'truncation' if which == 'cf' else 'nuclear'} d={d}",
                               fontsize=10)
        plot_utils.FIG.style_axes(ax[row, col])
    ax[1, col].set_xlabel(r"$\Delta\beta$")
for row in range(2):
    ax[row, 0].set_ylabel(r"fitted swing $\hat\alpha_i/\alpha_i$")
fig.legend(*ax[0, 0].get_legend_handles_labels(), title="attribute", fontsize=8,
           title_fontsize=9, loc="center right")
fig.tight_layout(rect=(0, 0, 0.93, 1))
plot_utils.save_figure(fig, FIGDIR / "fig2_dropout_curves.png")

# %% [markdown]
# ## Figure 3 — phase panel (Elhage-style), nuclear at $d = 3$
#
# Per $\Delta\beta$: signal-direction Gram $\hat V V^\top$ (feature vectors of the code,
# constant component removed), the fitted intercepts $\hat\beta_i$ (the frequency channel the
# code carries — no bias exists anywhere), and bars of the fitted swing colored by
# superposition (yellow = shares dimensions).

# %%
fig = plt.figure(figsize=(2.6 * len(BETA_STEPS), 6.4))
gs = fig.add_gridspec(2, len(BETA_STEPS), height_ratios=(1.15, 1.0), hspace=0.42, wspace=0.35)
cmap = plt.get_cmap("summer")                 # green = own dimension, yellow = shared
for col, bs in enumerate(BETA_STEPS):
    o = oracle(bs, D_PANEL)
    V, v0 = feature_vectors(o["nuc"]["E"], o["Z"])
    Vh = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    aG = fig.add_subplot(gs[0, col])
    aG.imshow(Vh @ V.T, cmap="RdBu_r", vmin=-np.abs(V).max(), vmax=np.abs(V).max())
    aG.set_title(rf"$\Delta\beta$ = {bs:g}" + f"\nexcess {o['nuc']['ex']:.3f}", fontsize=10)
    aG.set_xticks([]); aG.set_yticks([])
    a_hat, b_hat = o["nuc"]["stats"]
    sp = superposition_measure(V.T)
    aB = fig.add_subplot(gs[1, col])
    aB.barh(np.arange(M), a_hat / o["alpha"],
            color=cmap(np.clip(sp / max(sp.max(), 1e-12), 0, 1)))
    aB.invert_yaxis()
    aB.set_xlim(0, 1.05)
    aB.set_yticks(np.arange(M), [f"{j + 1}" for j in range(M)], fontsize=7)
    aB.set_xlabel(r"$\hat\alpha_i/\alpha_i$", fontsize=9)
    axt = aB.twiny()
    axt.plot(b_hat - o["beta"], np.arange(M), color="crimson", lw=1.2, marker=".", ms=4)
    axt.set_xlim(-3, 3); axt.tick_params(labelsize=7)
    axt.set_xlabel(r"$\hat\beta_i - \beta_i$", fontsize=8, color="crimson")
    plot_utils.FIG.style_axes(aB)
plot_utils.save_figure(fig, FIGDIR / "fig3_phase_panel.png")

# %% [markdown]
# ## Figure 4 — leakage audit
#
# Relative spurious mass per attribute, $\lambda_i = \mathbb{E}_x[\hat q_i - p_i]_+ /
# \mathbb{E}_x[p_i]$ (nuclear, $d = 3$): the fraction by which packing inflates an
# attribute, relative to its true rate. Packing aims its interference at the cheap
# (rare) outcomes.

# %%
from scipy.special import softmax

fig, ax = plt.subplots(figsize=(5.4, 3.6))
for bs in BETA_STEPS:
    o = oracle(bs, D_PANEL)
    q = softmax(o["nuc"]["E"] @ o["nuc"]["D"], axis=1)
    P = np.exp(o["logP"])
    lam = np.maximum(q - P, 0.0).mean(0)[:M] / P.mean(0)[:M]
    ax.plot(np.arange(1, M + 1), lam, marker="o", ms=4, label=f"{bs:g}")
ax.set_yscale("log")
ax.set_xlabel("attribute (frequent to rare)")
ax.set_ylabel(r"relative leaked mass $\lambda_i$")
ax.legend(title=r"$\Delta\beta$", fontsize=8, title_fontsize=9)
plot_utils.FIG.style_axes(ax)
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig4_leakage.png")

# %% [markdown]
# ## Figure 5 — trained k-linear vs the nuclear oracle
#
# Class-matched model (embedding + bias-free linear readout, $d = 3$) trained at the flat
# and steep ends of the sweep: excess CE, fitted swings, and code RSMs against the oracle.

# %%
def trained_run(beta_step):
    ov = ["setting=word_attribute", "runner=klinear",
          f"model_kwargs.model_config.embed_dim={D_PANEL}",
          "training_kwargs.train_steps=6000", f"family_kwargs.beta_step={beta_step}"]
    model, ds, evaluator, *_ = train_inline(ov)
    device = next(model.parameters()).device
    ids = torch.arange(ds.n_items, device=device)
    with torch.no_grad():
        logits = model(ids.unsqueeze(1))["logits"]
    G = torch.log_softmax(logits.double(), dim=-1).cpu().numpy()
    P = np.exp(ds.conditional)
    ce = -(P * G).sum() / ds.n_items
    bayes = -(P * ds.conditional).sum() / ds.n_items
    E = np.asarray(evaluator.item_representations(model), float)
    return dict(ex=ce - bayes, G=G, E=E)

TRAIN_BS = [0.0, 1.0]
runs = {bs: trained_run(bs) for bs in TRAIN_BS}
for bs in TRAIN_BS:
    o = oracle(bs, D_PANEL)
    print(f"beta_step={bs:g}  trained excess={runs[bs]['ex']:.4f}  "
          f"(truncation {o['cf']['ex']:.4f} / nuclear {o['nuc']['ex']:.4f})")

# %%
fig = plt.figure(figsize=(11.5, 6.4))
gs = fig.add_gridspec(2, 2 * len(TRAIN_BS), hspace=0.5, wspace=0.4)
for col, bs in enumerate(TRAIN_BS):
    o, r = oracle(bs, D_PANEL), runs[bs]
    a_tr, _ = attribute_stats(r["G"], o["Z"])
    a_nc, _ = o["nuc"]["stats"]
    aB = fig.add_subplot(gs[0, 2 * col:2 * col + 2])
    x = np.arange(1, M + 1)
    aB.bar(x - 0.18, a_nc / o["alpha"], width=0.36, label="nuclear")
    aB.bar(x + 0.18, a_tr / o["alpha"], width=0.36, label="trained")
    aB.set_title(rf"$\Delta\beta$ = {bs:g}: excess {r['ex']:.3f} vs nuclear "
                 f"{o['nuc']['ex']:.3f}", fontsize=10)
    aB.set_xlabel("attribute"); aB.set_ylabel(r"$\hat\alpha_i/\alpha_i$")
    aB.legend(fontsize=8)
    plot_utils.FIG.style_axes(aB)
    plot_utils.rdm_heatmap(fig.add_subplot(gs[1, 2 * col]), o["nuc"]["E"], title="nuclear code")
    plot_utils.rdm_heatmap(fig.add_subplot(gs[1, 2 * col + 1]), r["E"], title="trained code")
plot_utils.save_figure(fig, FIGDIR / "fig5_trained_vs_nuclear.png")

# %% [markdown]
# ## Summary
#
# - **The frequency lever drops features — by reweighted truncation, not superposition.**
#   At $\Delta\beta = 0$ the tail is partially packed into shared dimensions (the superposed
#   regime; graded swings). Steepening the ladder drops the rare tail entirely
#   ($\hat\alpha_i/\alpha_i \to 0$) while nuclear converges to truncation (fig 1): the
#   mechanism is Fisher pricing (errors on rare outcomes cost nothing), so $\tilde\beta$
#   acts as TMS's *importance* lever. It is not a sparsity/packing lever — a softmax's
#   outcome stream is always 1-hot, so there is no outcome-side co-activation to make
#   packing cheap. Packing pressure here comes only from sharpness $\alpha$ (~15% gain at
#   flat $\tilde\beta$); an input-side co-activation dial (sparse features) is the
#   phase-two rung that would restore the full TMS phase diagram.
# - **The trained model is the nuclear optimum.** Class-matched k-linear lands on the
#   convex program's excess to three decimals at both sweep ends (0.1529 vs 0.1526 flat;
#   0.0398 vs 0.0398 steep) and matches its per-attribute swings bar-for-bar and its code
#   RSM (fig 5) — the variational account transfers to the asymmetric task unchanged.
# - **Interference is aimed at the cheap outcomes** (fig 4): relative leaked mass is
#   redirected toward the rare attributes as the ladder steepens.
# - The intercepts of dropped attributes are under-represented ($\hat\beta_i - \beta_i > 0$,
#   fig 3 strips): once an attribute's swing is dropped, the model does not bother making it
#   as rare as it truly is — rarity beyond "negligible" is Fisher-free.
