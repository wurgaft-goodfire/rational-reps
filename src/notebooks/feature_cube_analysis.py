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
# Items are the $2^m$ vertices of the hypercube; the conditional is a product kernel with
# **per-feature couplings** $w_i = w_0(\gamma) \cdot 0.9^i$ — the importance spectrum, fixed
# throughout ($\gamma = 0.03$, sharp: the softmax-saturation regime where packing can beat
# truncation). Every feature is active with the same probability $1 - s$, where $s$ is the
# **sparsity** — the phase knob, controlling only co-activation. Importance and frequency
# are thereby decoupled, as in toy models of superposition: sparsifying cheapens
# interference without cheapening the features. The log-conditional is purely bilinear in
# $\pm 1$ codes (no outcome field), so every model is a bias-free code $\times$ matrix
# readout. (The config field `sparsity` is the feature probability $1 - s$.)
#
# Two predictions at budget $d$, both in the model's own class (free per-item codes):
# **closed form** (`factor_logcond`, rank-$d$ prefix) — whitened truncation, the quadratic
# account; **nuclear** (`nuclear_features(logP, I, d)`) — the exact-CE optimum via the
# convex relaxation, deterministic. Trained models are judged against the nuclear fit.

# %%
import os, sys
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(sys.prefix).parent            # <repo>/.venv -> <repo>
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import torch
from src.code.data import feature_cube
from src.code.geometry import factor_logcond, nuclear_features, expected_ce
from src.code.geometry import feature_vectors as _feature_vectors
from src.code.train import train_inline
from src.code import plot_utils
import matplotlib.pyplot as plt

FIGDIR = Path("outputs/feature_cube/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

M, D_BUDGET, W_DECAY, GAMMA = 6, 2, 0.9, 0.03


@lru_cache(maxsize=None)
def oracle(p):
    """Whitened truncation and free-row nuclear at feature probability ``p`` (sparsity 1 - p).
    Returns (pi, Z, Lf, H_cf, H_nuc, ex_cf, ex_nuc): item codes (N, d), excesses over Bayes."""
    logP, pi, Z, Lf = feature_cube(np.full(M, p), GAMMA, W_DECAY)
    I = np.eye(len(Z))
    _, E, Dfull, _, _ = factor_logcond(logP, w=pi)
    A_cf, D_cf = E[:, :D_BUDGET].T, Dfull[:D_BUDGET]
    ce_cf, bayes = expected_ce(I, logP, A_cf, D_cf, np.zeros(logP.shape[1]), w=pi)
    nuc = nuclear_features(logP, I, D_BUDGET, w=pi, fit_bias=False)
    ce_nuc, _ = expected_ce(I, logP, *nuc[:3], w=pi)
    return pi, Z, Lf, A_cf.T, nuc[0].T, ce_cf - bayes, ce_nuc - bayes


def feature_vectors(H, Z, w):
    """Effective feature vectors (m, d) of the additive fit (constant component dropped)."""
    return _feature_vectors(H, Z, w)[0]


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

# %% [markdown]
# ## Figure 1 — oracle geometry along the sparsity sweep
#
# Effective feature vectors of the truncation and nuclear codes (1 = most important). At low sparsity the exact optimum is a packed fan (five features with real
# directions) while truncation keeps only the top pair; sparsifying *concentrates* the
# optimum onto the most important features even as its advantage over truncation widens.

# %%
P_CORNERS = [0.5, 0.25, 0.1, 0.05]
fig, ax = plt.subplots(2, len(P_CORNERS), figsize=(3.4 * len(P_CORNERS), 7.4))
for col, p in enumerate(P_CORNERS):
    pi, Z, Lf, H_cf, H_nuc, ex_cf, ex_nuc = oracle(p)
    for row, (V, name, ex) in enumerate([(feature_vectors(H_cf, Z, pi), "closed form", ex_cf),
                                         (feature_vectors(H_nuc, Z, pi), "nuclear", ex_nuc)]):
        plot_utils.feature_arrows(ax[row, col], V, lw=2, fontsize=10.5)
        ax[row, col].set_title(f"{name}:  excess = {ex:.3f}", fontsize=10.5)
    ax[0, col].set_title(f"sparsity {1 - p:g}\n" + ax[0, col].get_title(), fontsize=10.5)
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig1_oracle_geometry.png")

# %% [markdown]
# ## Figure 2 — trained architectures vs the oracles
#
# k-linear and MLP train on pointwise item→neighbor pairs, GRU and transformer on short
# episodes; every readout is bias-free. Effective feature vectors over item-code RDMs.

# %%
ARCH = {
    "klinear": ["runner=klinear", f"model_kwargs.model_config.embed_dim={D_BUDGET}",
                "training_kwargs.train_steps=6000"],
    "mlp": ["runner=mlp", f"model_kwargs.model_config.hidden_size={D_BUDGET}",
            "model_kwargs.model_config.num_layers=1", "model_kwargs.model_config.init_scale=0.5",
            "training_kwargs.learning_rate=0.003", "training_kwargs.train_steps=8000"],
    "rnn": ["runner=rnn", f"model_kwargs.model_config.hidden_size={D_BUDGET}",
            "training_kwargs.learning_rate=0.002", "training_kwargs.train_steps=12000"],
    "transformer": ["runner=transformer", f"model_kwargs.model_config.hidden_size={D_BUDGET}",
                    "model_kwargs.model_config.num_attention_heads=1",
                    "model_kwargs.model_config.intermediate_size=16",
                    "training_kwargs.learning_rate=0.001", "training_kwargs.train_steps=12000"],
}
P_POINTS = [0.5, 0.1]

arch_results = {}
for p in P_POINTS:
    pi, Z, Lf, H_cf, H_nuc, ex_cf, ex_nuc = oracle(p)
    entry = {"closed form": (ex_cf, H_cf), "nuclear": (ex_nuc, H_nuc)}
    base = ["setting=feature_cube", f"n_features={M}", f"sparsity={p}", f"gamma={GAMMA}",
            "p_decay=1.0", f"w_decay={W_DECAY}"]
    for name, ov in ARCH.items():
        model, ds, evaluator, *_ = train_inline(base + ov)
        entry[name] = (item_excess_ce(model, ds),
                       np.asarray(evaluator.item_representations(model), float))
        print(f"sparsity={1 - p:g} {name:12s} excess CE={entry[name][0]:.4f}  "
              f"(cf {ex_cf:.4f} / nuclear {ex_nuc:.4f})")
    arch_results[p] = (entry, pi, Z)

# %%
names = ["closed form", "nuclear"] + list(ARCH)
fig = plt.figure(figsize=(2.7 * len(names), 5.6 * len(P_POINTS)))
gs = fig.add_gridspec(2 * len(P_POINTS), len(names), hspace=0.45, wspace=0.3)
for prow, p in enumerate(P_POINTS):
    entry, pi, Z = arch_results[p]
    for col, n in enumerate(names):
        ex, H = entry[n]
        V = feature_vectors(H, Z, pi)
        V2 = V if V.shape[1] == 2 else V @ np.linalg.svd(V, full_matrices=False)[2][:2].T
        a = fig.add_subplot(gs[2 * prow, col])
        plot_utils.feature_arrows(a, V2, scale=1.3, fontsize=9.5)
        a.set_title(f"{n}\nexcess CE {ex:.3f}", fontsize=10)
        if col == 0:
            a.set_ylabel(f"sparsity {1 - p:g}", fontsize=11)
        plot_utils.rdm_heatmap(fig.add_subplot(gs[2 * prow + 1, col]), entry[n][1], title="")
plot_utils.save_figure(fig, FIGDIR / "fig2_geometry_rdms.png")

# %% [markdown]
# ## Figure 3 — feature regimes across sparsity
#
# Per-feature effective vector length, relative to the longest (1 = most important). The
# closed form's tail modes rise from zero with sparsity; the exact optimum starts fully
# packed at low sparsity and concentrates toward the important features as the measure
# sparsifies. The toy-models entering-staircase does not appear under CE: input-side
# co-activation gets rarer, but the softmax couples every outcome in every row, so
# readout-side interference never sparsifies away.

# %%
P_DENSE = np.geomspace(0.5, 0.05, 9)
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
rel = {"closed form": [], "nuclear": []}
for p in P_DENSE:
    pi, Z, Lf, H_cf, H_nuc, *_ = oracle(float(p))
    rel["closed form"].append(np.linalg.norm(feature_vectors(H_cf, Z, pi), axis=1))
    rel["nuclear"].append(np.linalg.norm(feature_vectors(H_nuc, Z, pi), axis=1))
for a, (name, L) in zip(ax, rel.items()):
    L = np.array(L)
    L = L / L.max(axis=1, keepdims=True)
    for j in range(M):
        a.plot(1 - P_DENSE, L[:, j], marker="o", ms=3.5, label=f"{j + 1}")
    a.set_title(name, fontsize=11)
    a.set_xlabel("sparsity")
    plot_utils.FIG.style_axes(a)
ax[0].set_ylabel("relative feature length")
fig.legend(*ax[0].get_legend_handles_labels(), title="feature", fontsize=9,
           title_fontsize=9.5, loc="center right")
fig.tight_layout(rect=(0, 0, 0.92, 1))
plot_utils.save_figure(fig, FIGDIR / "fig3_feature_regimes.png")

# %% [markdown]
# ## Summary
#
# - With importance (couplings) decoupled from frequency (uniform activation probability),
#   the packing gap widens monotonically with sparsity (nuclear/cf ratio 0.84 → 0.67), but
#   the geometry runs opposite to toy models of superposition: the exact optimum is a packed
#   fan at low sparsity and *concentrates* onto the important features as the measure
#   sparsifies. The TMS mechanism (interference needs co-activation) only governs the input
#   side; under CE the softmax couples every outcome in every row, and that interference
#   channel does not sparsify away.
# - The nuclear fit is the CE optimum of the model class itself, computed without training;
#   trained k-linear lands on it (1.397 vs 1.388 dense; 0.441 vs 0.439 sparse). The lambda
#   path alone missed the dense packed basin (it proposed only truncations — trained
#   k-linear exposed the trap); `nuclear_features` now also polishes from a small fixed
#   init, mirroring training's own dynamics. The width-2 MLP and transformer stay badly
#   tuned.
