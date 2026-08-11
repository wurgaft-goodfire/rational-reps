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
# Two predictions at budget $d$ (`src/code/geometry.py`, both `fit_bias=False`):
# **closed form** (`factor_features`) — whitened truncation, the quadratic account that sees
# only second moments; **nuclear** (`nuclear_features`) — the exact-CE fit through its convex
# relaxation: CE is convex in the logit map $R = A^\top D$, so the nuclear-penalized program
# is solved globally along a regularization path, and a deterministic joint polish descends
# within the basin each path point proposes. No restarts, no init dependence. The deviation
# between the two is the superposition signal; a trained k-linear model per cell is the
# empirical reference both predictions are judged against.

# %%
import os, sys
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(sys.prefix).parent            # <repo>/.venv -> <repo>
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.data import feature_cube
from src.code.geometry import (factor_features, nuclear_features, expected_ce,
                               principal_angles, _psd_sqrt)
from src.code import plot_utils
import matplotlib.pyplot as plt

QUICK = bool(int(os.environ.get("QUICK", "0")))   # QUICK=1: coarse grids + short fits/training
FIGDIR = Path("outputs/feature_cube") / ("figures_quick" if QUICK else "figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

M, D_BUDGET, P_DECAY = 6, 2, 0.7


@lru_cache(maxsize=None)
def cube(sparsity, gamma, m=M):
    """The cube at heat time gamma with the frequency ladder p_i = sparsity * P_DECAY^i."""
    return feature_cube(sparsity * P_DECAY ** np.arange(m), gamma)


@lru_cache(maxsize=None)
def oracle(sparsity, gamma):
    """Closed-form and nuclear fits at one phase-plane cell, memoized so every figure shares
    the same fits. Returns (pi, Z, Lf, cf, nuc, ex_cf, ex_nuc) with excesses over Bayes."""
    logP, pi, Z, Lf = cube(sparsity, gamma)
    cf = factor_features(Lf, Z, D_BUDGET, w=pi, fit_bias=False)
    ce_cf, bayes = expected_ce(Z, Lf, *cf[:3], w=pi)
    nuc = nuclear_features(Lf, Z, D_BUDGET, w=pi, fit_bias=False)
    ce_nuc, _ = expected_ce(Z, Lf, *nuc[:3], w=pi)
    return pi, Z, Lf, cf, nuc, ce_cf - bayes, ce_nuc - bayes

# %%
import torch
from src.code.train import train_inline


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


KLINEAR_STEPS = 2000 if QUICK else 6000


@lru_cache(maxsize=None)
def trained_klinear(sparsity, gamma):
    """The empirical reference at one cell: k-linear at the oracle budget, trained from
    scratch. Returns (excess CE, item codes H)."""
    model, ds, evaluator, *_ = train_inline(
        ["setting=feature_cube", f"n_features={M}", f"sparsity={sparsity}", f"gamma={gamma}",
         f"p_decay={P_DECAY}", "runner=klinear", f"model_kwargs.model_config.embed_dim={D_BUDGET}",
         f"training_kwargs.train_steps={KLINEAR_STEPS}"])
    return item_excess_ce(model, ds), np.asarray(evaluator.item_representations(model), float)

# %% [markdown]
# ## Figure 1 — what the quadratic account sees
#
# Left: the whitened spectrum `factor_features` truncates. The frequency ladder is the only
# thing distinguishing features, and at second order sharpness ($\gamma$) and sparsity both
# just rescale this picture — the two knobs are indistinguishable to the closed form.
# Everything exact CE does beyond truncating this spectrum is invisible at second order.
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
ax[0].legend(fontsize=9); ax[0].set_title("the spectrum the closed form truncates")

logP, pi, Z, Lf = cube(0.1, 0.03)
plot_utils.conditional_heatmap(ax[1], logP, title=r"$\log p(x'\mid x)$, sparse-sharp cube",
                               xlabel="", ylabel="")
plot_utils.FIG.style_axes(ax[0])
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig1_spectrum.png")

# %% [markdown]
# ## Figure 2 — the phase plane: truncation → packing in (sharpness × sparsity)
#
# For each cell (everything bias-free unless noted): closed form, nuclear, nuclear *granted
# an output bias*, and a trained k-linear at the same budget. Four readouts:
#
# - **nuclear/cf** (top left): the phase signal. 1 = truncation optimal; small = packing wins.
# - **k-linear/nuclear** (top right): reality vs prediction — ≈1 means the deterministic
#   convex-path fit accounts for what training finds.
# - **max principal angle** (bottom left) between the Fisher-whitened row spaces of the
#   fitted logit maps.
# - **bias ablation** (bottom right): nuclear-with-bias / nuclear-bias-free. The target has no
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
    ratio_nuc, ratio_klin, angle, bias_gain = (dat["ratio_nuc"], dat["ratio_klin"],
                                               dat["angle"], dat["bias_gain"])
else:
    shape = (len(SPARSITY_GRID), len(GAMMA_GRID))
    ratio_nuc, ratio_klin, angle, bias_gain = (np.zeros(shape) for _ in range(4))
    for i, sp in enumerate(SPARSITY_GRID):
        for j, gamma in enumerate(GAMMA_GRID):
            pi, Z, Lf, cf, nuc, ex_cf, ex_nuc = oracle(sp, gamma)
            nuc_bias = nuclear_features(Lf, Z, D_BUDGET, w=pi)
            ce_bias, bayes = expected_ce(Z, Lf, *nuc_bias[:3], w=pi)
            ex_klin, _ = trained_klinear(sp, gamma)
            q = pi @ np.exp(cube(sp, gamma)[0])
            Fh = _psd_sqrt(np.diag(q) - np.outer(q, q))
            ang = principal_angles((cf[0].T @ cf[1]) @ Fh, (nuc[0].T @ nuc[1]) @ Fh).max()
            ratio_nuc[i, j] = ex_nuc / ex_cf
            ratio_klin[i, j] = ex_klin / ex_nuc
            angle[i, j] = np.degrees(ang)
            bias_gain[i, j] = (ce_bias - bayes) / ex_nuc
            print(f"sparsity={sp:4.2f} gamma={gamma:4.2f}  exCE cf={ex_cf:7.4f}  "
                  f"nuclear={ex_nuc:7.4f}  klinear={ex_klin:7.4f}  "
                  f"with-bias={ce_bias - bayes:7.4f}  ratio={ratio_nuc[i, j]:.3f}")
    np.savez(cache, ratio_nuc=ratio_nuc, ratio_klin=ratio_klin, angle=angle,
             bias_gain=bias_gain)

fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
panels = [(ratio_nuc, "excess-CE ratio  nuclear / closed form", 0, 1, "viridis_r"),
          (ratio_klin, "trained k-linear / nuclear  (reality vs prediction)", 0, 1.5, "viridis_r"),
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
# closed form and the bias-free nuclear fit at four corners. Truncation keeps the top of the
# frequency ladder everywhere; the CE optimum leaves it in the sharp column, spreading rare
# features into their own directions.

# %%
CORNERS = [(0.5, 0.6), (0.5, 0.03), (0.1, 0.6), (0.1, 0.03)]
fig, ax = plt.subplots(2, len(CORNERS), figsize=(3.4 * len(CORNERS), 7.4))
for col, (sp, gamma) in enumerate(CORNERS):
    pi, Z, Lf, cf, nuc, ex_cf, ex_nuc = oracle(sp, gamma)
    for row, (A, name, ex) in enumerate([(cf[0], "closed form", ex_cf),
                                         (nuc[0], "nuclear", ex_nuc)]):
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

arch_results = {}
for sp, gamma in POINTS:
    pi, Z, Lf, cf, nuc, ex_cf, ex_nuc = oracle(sp, gamma)
    entry = {"closed form": (ex_cf, Z @ cf[0].T),
             "nuclear": (ex_nuc, Z @ nuc[0].T)}
    base = ["setting=feature_cube", f"n_features={M}", f"sparsity={sp}", f"gamma={gamma}",
            f"p_decay={P_DECAY}"]
    for name, ov in ARCH.items():
        model, ds, evaluator, *_ = train_inline(base + ov)
        H = evaluator.item_representations(model)
        entry[name] = (item_excess_ce(model, ds), np.asarray(H, float))
        print(f"sparsity={sp:4.2f} gamma={gamma:4.2f} {name:12s} "
              f"excess CE={entry[name][0]:.4f}  (cf {ex_cf:.4f} / nuclear {ex_nuc:.4f})")
    arch_results[(sp, gamma)] = (entry, pi, Z)

# %% [markdown]
# ## Figure 5 — geometry over RDMs, per model
#
# The house comparison layout: each model's effective feature vectors (weighted additive fit
# of its item codes; exactly the $A$ columns for the oracles) over its item-code RDM (all
# $2^6$ items, lexicographic). Truncation pins the rare features at the origin; the packed
# solutions give every feature a direction.

# %%
names = ["closed form", "nuclear"] + list(ARCH)
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
#   and both oracles are pure matrix readouts of a code.
# - The phase plane (Figure 2): truncation optimal in the flat column, packing in the sharp
#   column — and within the *bias-free* class the nuclear/cf ratio is set almost entirely by
#   sharpness (~0.99 → ~0.67 as $\gamma$: 0.6 → 0.03, nearly identical in every sparsity
#   row). Sparsity does not move the truncation→packing boundary here; its co-activation
#   advantage (interference $\propto p_i p_j$, benefit $\propto p_i$) is instead harvested by
#   the **bias channel** (bottom right): with-bias/bias-free 0.84 (flat) → 0.55–0.58 (sharp).
# - The exact-CE branch is `nuclear_features`: CE is convex in the logit map $R = A^\top D$
#   (only the rank constraint is nonconvex), so the nuclear-penalized program is solved
#   globally along a regularization path and polished within each proposed basin —
#   deterministic, no restarts. It matches the closed form in the flat column and beats the
#   old restart-ALS numbers at the sharp points (1.072 vs 1.099 dense-sharp).
# - Trained k-linear equals nuclear through the flat and mid columns (ratio 1.00–0.99) but
#   still sits 9–14% *below* it in the sharp column (0.917 vs 1.072; 0.230 vs 0.254). The
#   remaining gap is the relaxation's truncation looseness and/or a genuine class gap —
#   k-linear's free per-item codes need not stay additive in the features, and its sharp-cell
#   RDM (Figure 5) visibly departs from the additive checker pattern. The additive-residual
#   diagnostic (regress the trained embedding on $Z$) would separate the two.
# - A pure max-margin packing is *not* the sharp limit of this problem: dropping a feature
#   gives its flip-pairs margin exactly 0, while any full-support rank-2 packing of six
#   features has a negative worst-case margin, so max-min-margin prefers truncation. CE
#   instead charges ties a bounded log-degeneracy cost, which is why packing wins under CE.
#   The equally-spaced polygon of the MSE superposition story does not appear here.

# %%
