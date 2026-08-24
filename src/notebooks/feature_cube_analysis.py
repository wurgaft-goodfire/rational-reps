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
# # Word → feature: sparsity vs importance under a rank budget
#
# $N$ words, each a bundle of $m$ features drawn i.i.d. Bernoulli($p$) in $\{0, 1\}$;
# **sparsity** $= 1 - p$. Outcomes are the $m$ feature tokens:
# $\log p(i \mid x) = \mathrm{log\,softmax}_i(\alpha_i Z_{xi})$, $\alpha_i = s \cdot r^i$.
# Knobs: $p$ = co-activation (the TMS sparsity dial), $r$ = importance ladder, $s$ =
# sharpness. No null token: the softmax gauge absorbs one dimension, so effective rank is
# $m - 1$ and the budget question is $d < m - 1$.
#
# The sweep stops at $p = 0.5$: complementing the features ($z \mapsto 1 - z$) maps
# $p > 0.5$ onto the sparse task plus a fixed outcome field $\alpha_i$ — a frequency
# ladder, not new sparsity physics.
#
# Oracles in the trained model's class (free per-item codes, bias-free readout):
# **truncation** (`factor_logcond`, rank-$d$ prefix) and **nuclear**
# (`nuclear_features(logP, I, d)`, the exact-CE optimum). Dropout is read from the fitted
# per-feature swing $\hat\alpha_i$ (`feature_stats`: gauge-free least squares
# $G_{xi} \sim a_i Z_{xi} + b_i + c_x$).

# %%
import os, sys
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(sys.prefix).parent            # <repo>/.venv -> <repo>
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.data import feature_cube
from src.code.geometry import (truncation_features, nuclear_features, expected_ce,
                               feature_stats, feature_vectors, superposition_measure)
from src.code.train import train_inline
from src.code import plot_utils
import matplotlib.pyplot as plt

FIGDIR = Path("outputs/feature_cube/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

N, M, DECAY, SCALE, SEED = 256, 8, 0.9, 8.0, 0
P_SWEEP = [float(round(p, 4)) for p in np.geomspace(0.5, 0.005, 10)]
D_GRID = [2, 3, 4]
D_PANEL = 3


SPARSITY = [1.0 - p for p in P_SWEEP]
COLORS = {2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a"}


def sparsity_ticks(ax):
    """Logit x-scale with plain floating-point tick labels."""
    from matplotlib.ticker import FixedLocator, NullLocator
    ticks = [0.5, 0.7, 0.9, 0.95, 0.98, 0.995]
    ax.set_xscale("logit")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])


@lru_cache(maxsize=None)
def oracle(p, d):
    """Truncation and nuclear fits at (p, d): task, codes/readouts, excesses over Bayes,
    and fitted per-feature (swing, intercept)."""
    logP, info = feature_cube(N, M, p, DECAY, SCALE, SEED)
    I = np.eye(len(logP))
    zeros = np.zeros(logP.shape[1])
    A_cf, D_cf = truncation_features(logP, d)
    ce_cf, bayes = expected_ce(I, logP, A_cf, D_cf, zeros)
    A, D, b, _ = nuclear_features(logP, I, d, fit_bias=False)
    ce_nuc, _ = expected_ce(I, logP, A, D, b)
    return dict(**info,
                cf=dict(E=A_cf.T, ex=ce_cf - bayes,
                        stats=feature_stats(A_cf.T @ D_cf, info["Z"])),
                nuc=dict(E=A.T, ex=ce_nuc - bayes,
                         stats=feature_stats(A.T @ D, info["Z"])))

# %% [markdown]
# ## Figure 1 — excess CE across the sparsity sweep
#
# For each sparsity $1 - p$ and budget $d$: excess CE over Bayes of the rank-$d$
# truncation (dashed) and the nuclear optimum (solid), uniform over the $N$ words.
# Color = budget, linestyle = oracle. Criterion: if sparsity is a packing lever, the
# nuclear-over-truncation gain should grow with sparsity (rare co-activation makes shared
# dimensions cheap); a constant ratio between the line styles falsifies it.

# %%
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(5.8, 3.8))
for d in D_GRID:
    ax.plot(SPARSITY, [oracle(p, d)["nuc"]["ex"] for p in P_SWEEP],
            marker="o", ms=4, color=COLORS[d], lw=2)
    ax.plot(SPARSITY, [oracle(p, d)["cf"]["ex"] for p in P_SWEEP],
            marker="o", ms=4, color=COLORS[d], lw=2, ls="--")
ax.set_yscale("log")
sparsity_ticks(ax)
ax.set_xlabel("sparsity")
ax.set_ylabel("excess CE")
handles = [Line2D([], [], color=COLORS[d], lw=2, label=f"d = {d}") for d in D_GRID] + \
          [Line2D([], [], color="0.25", lw=2, ls=ls, label=lab)
           for ls, lab in [("-", "nuclear"), ("--", "truncation")]]
ax.legend(handles=handles, fontsize=8, ncol=2)
plot_utils.FIG.style_axes(ax)
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig1_excess_sweep.png")

# %% [markdown]
# ## Figure 2 — trained architectures vs the oracles (code RDMs)
#
# Trained architectures at $d = 3$ (linear and MLP pointwise; GRU and transformer as
# causal LMs over [word, attribute_1 ... attribute_L] episodes: the word is the prefix and
# its features are drawn exchangeably from the same kernel row) at a dense and a sparse
# corner, next to the truncation and nuclear codes. Excess CE is the
# evaluator's KL against the Bayes targets, so every architecture is scored identically.
# Criterion: trained excess and RDM match nuclear, as in the feature cube.

# %%
ARCH = {
    "linear": ["runner=linear", f"model_kwargs.model_config.embed_dim={D_PANEL}",
                "training_kwargs.train_steps=6000"],
    "mlp": ["runner=mlp", f"model_kwargs.model_config.hidden_size={D_PANEL}",
            "model_kwargs.model_config.num_layers=1",
            "training_kwargs.learning_rate=0.003", "training_kwargs.train_steps=8000"],
    "rnn": ["runner=rnn", f"model_kwargs.model_config.hidden_size={D_PANEL}",
            "training_kwargs.learning_rate=0.002", "training_kwargs.train_steps=12000"],
    "transformer": ["runner=transformer", f"model_kwargs.model_config.hidden_size={D_PANEL}",
                    "model_kwargs.model_config.num_attention_heads=1",
                    "model_kwargs.model_config.intermediate_size=16",
                    "training_kwargs.learning_rate=0.001", "training_kwargs.train_steps=12000"],
}
P_CORNERS = [P_SWEEP[1], P_SWEEP[6]]              # on the sweep grid: dense / sparse


def trained_run(p, ov_arch):
    """Excess CE over Bayes (the evaluator's KL) and the item code, for one architecture."""
    model, ds, evaluator, *_ = train_inline(
        ["setting=feature_cube", f"structure_kwargs.p={p}"] + ov_arch)
    metrics, _ = evaluator.evaluate(model)
    return dict(ex=metrics["loss"], E=np.asarray(evaluator.item_representations(model), float))


arch_results = {}
for p in P_CORNERS:
    o = oracle(p, D_PANEL)
    entry = {"truncation": (o["cf"]["ex"], o["cf"]["E"]),
             "nuclear": (o["nuc"]["ex"], o["nuc"]["E"])}
    for name, ov in ARCH.items():
        r = trained_run(p, ov)
        entry[name] = (r["ex"], r["E"])
        print(f"p={p:g} {name:8s} excess CE={r['ex']:.4f}  "
              f"(truncation {o['cf']['ex']:.4f} / nuclear {o['nuc']['ex']:.4f})")
    arch_results[p] = entry

# %%
names = ["truncation", "nuclear"] + list(ARCH)
fig = plt.figure(figsize=(2.7 * len(names), 3.1 * len(P_CORNERS)))
gs = fig.add_gridspec(len(P_CORNERS), len(names), hspace=0.35, wspace=0.3)
for row, p in enumerate(P_CORNERS):
    for col, n in enumerate(names):
        ex, E = arch_results[p][n]
        a = fig.add_subplot(gs[row, col])
        plot_utils.rdm_heatmap(a, E, title=f"{n}\nexcess CE {ex:.3f}")
        if col == 0:
            a.set_ylabel(f"$p$ = {p:g}  (sparsity {1 - p:g})", fontsize=10)
plot_utils.save_figure(fig, FIGDIR / "fig2_geometry_rdms.png")

# %% [markdown]
# ## Figure 3 — feature regimes across sparsity
#
# How each point is computed: build the task at this sparsity, fit the oracle at
# $d = 3$, form its logits $\hat G = E D$ ($N \times m$), and run the gauge-free least
# squares $\hat G_{xi} \sim a_i Z_{xi} + b_i + c_x$; the plotted value is
# $\hat a_i / \alpha_i$ — the fraction of feature $i$'s true log-odds swing the code
# actually delivers (1 = fully represented, 0 = dropped). Criterion (TMS): sparsifying
# should let the unimportant tail back in (graded, superposed); densifying should force
# dropout. Swings above 1 are saturation overshoot: once predictions are near one-hot, CE
# no longer penalizes overdriving the kept logits. Caveat at the extreme sparse end: with
# $N = 256$, $p = 0.005$ leaves $\approx N p = 1.3$ carrier words per feature, so
# swings there reflect the sampled lexicon, not the ensemble.

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for a, which in zip(ax, ["cf", "nuc"]):
    rel = np.array([oracle(p, D_PANEL)[which]["stats"][0] / oracle(p, D_PANEL)["alpha"]
                    for p in P_SWEEP])
    for j in range(M):
        a.plot(SPARSITY, rel[:, j], marker="o", ms=3.5, label=f"{j + 1}")
    a.set_title("truncation" if which == "cf" else "nuclear", fontsize=11)
    a.set_xlabel("sparsity")
    plot_utils.FIG.style_axes(a)
sparsity_ticks(ax[0])                         # shared axis
ax[0].set_ylabel(r"fitted swing $\hat\alpha_i/\alpha_i$")
fig.legend(*ax[0].get_legend_handles_labels(), title="feature", fontsize=9,
           title_fontsize=9.5, loc="center right")
fig.tight_layout(rect=(0, 0, 0.92, 1))
plot_utils.save_figure(fig, FIGDIR / "fig3_attribute_regimes.png")

# %% [markdown]
# ## Figure 4 — phase panel across the sweep ($d = 3$)
#
# Per sparsity and oracle: signal-direction Gram $\hat V \hat V^\top$ (each row of
# $\hat V$ is feature $i$'s effective direction in code space; red diagonal = own
# dimension, off-diagonal structure = interference) and swing bars colored by
# superposition (yellow = shares dimensions); dotted line = full delivery.

# %%
P_PANEL = P_SWEEP[::2]                            # every other sweep point
BAR_MAX = 1.05 * max(float((oracle(p, D_PANEL)[w]["stats"][0] / oracle(p, D_PANEL)["alpha"]).max())
                     for p in P_PANEL for w in ("nuc", "cf"))
fig = plt.figure(figsize=(2.35 * len(P_PANEL), 11.6))
gs = fig.add_gridspec(4, len(P_PANEL), height_ratios=(1.1, 1.0, 1.1, 1.0), hspace=0.45,
                      wspace=0.35)
cmap = plt.get_cmap("summer")                 # green = own dimension, yellow = shared
for col, p in enumerate(P_PANEL):
    o = oracle(p, D_PANEL)
    for row, which in enumerate(["nuc", "cf"]):
        V, _ = feature_vectors(o[which]["E"], o["Z"])
        G = V @ V.T
        aG = fig.add_subplot(gs[2 * row, col])
        aG.imshow(G, cmap="RdBu_r", vmin=-np.abs(G).max(), vmax=np.abs(G).max())
        if row == 0:
            aG.set_title(f"sparsity {1 - p:.3g}", fontsize=10)
        aG.set_xticks([]); aG.set_yticks([])
        aG.set_xlabel(f"excess {o[which]['ex']:.3f}", fontsize=9)
        if col == 0:
            aG.set_ylabel("nuclear" if which == "nuc" else "truncation", fontsize=11)
        a_hat = o[which]["stats"][0]
        sp = superposition_measure(V.T)
        aB = fig.add_subplot(gs[2 * row + 1, col])
        aB.barh(np.arange(M), a_hat / o["alpha"],
                color=cmap(np.clip(sp / max(sp.max(), 1e-12), 0, 1)))
        aB.invert_yaxis()
        aB.set_xlim(0, BAR_MAX)
        aB.axvline(1.0, color="0.4", lw=0.8, ls=":")
        aB.set_yticks(np.arange(M), [f"{j + 1}" for j in range(M)], fontsize=7)
        aB.set_xlabel(r"$\hat\alpha_i/\alpha_i$", fontsize=9)
        plot_utils.FIG.style_axes(aB)
plot_utils.save_figure(fig, FIGDIR / "fig4_elhage_phase.png")

# %% [markdown]
# ## Summary
#
# - **The full TMS phase transition, driven by sparsity.** At $p = 0.5$ the $d = 3$ code
#   keeps the top 3–4 features and drops the tail; by $p \le 0.1$ all eight features
#   are represented, superposed into three dimensions (figs 3, 4). Sharpness makes packing
#   pay: nuclear beats truncation by ~64% at $p = 0.3$ and ~94% at $p = 0.02$ (fig 1) —
#   at the sparse end nuclear at $d$ even beats truncation at $d + 1$.
# - **Saturation overshoot**: fitted swings of kept features exceed $\alpha$ (up to
#   $\approx 2$, fig 3 right): once outputs are near one-hot, CE no longer penalizes
#   overdriving the logits, so the packed code exaggerates what it keeps.
# - **Trained k-linear tracks nuclear** (fig 2): 0.151 vs 0.142 at $p = 0.3$ (within 6%,
#   no longer to four decimals — the saturated landscape is harder) and exactly 0.0078 at
#   $p = 0.02$, with matching RDMs.
# - **The MLP fails twice over** (0.91 vs 0.14 at $p = 0.3$). Immediate cause: at the
#   notebook's budget (8k steps) two of its three bottleneck ReLU units are dead — the
#   code is effectively rank 1. Longer training (30–60k steps, lr 0.01–0.03) revives them
#   but plateaus at ~0.22, still 1.5x nuclear. The residual gap is structural: the
#   post-ReLU code lives in the positive orthant with a bias-free readout, and emulating
#   the signed optimum by shifting the code positive welds an unremovable outcome field
#   $W c$ into every prediction (nuclear's own code shifted positive costs 0.51; the
#   signed original 0.14).
# - **Sequence models sit between linear and the MLP** (fig 2), both untuned, as in the
#   feature cube. They train as causal LMs on [word, attribute_1 ... attribute_L] episodes
#   — the word is the prefix, its features are exchangeable draws from the same kernel
#   row — and their softmax spans the full token vocabulary, so leakage onto word tokens is
#   priced too: a slightly harder objective than the pointwise soft-target CE.
# - Excess CE is non-monotone in $p$ (fig 1, peak near $p \approx 0.3$); both ends
#   trivialize. Below $p \approx 0.01$ the finite lexicon dominates ($N p \approx 1$–$3$
#   carrier words per feature); swings there describe the sample, not the ensemble.
