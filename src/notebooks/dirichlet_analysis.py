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
# # In-context belief dynamics: the Dirichlet-categorical task
#
# The simplest case where the belief state lives over the next-token distribution itself: each
# sequence draws `theta ~ Dir(alpha)` (flat prior) and emits iid tokens. The Bayes-optimal output
# is the posterior predictive `(alpha + counts_t)/(alpha_0 + t)` — a running average that starts
# at uniform and settles on `theta` as draws accumulate — and the predicted representation is its
# **centered log** (`ds.logit_code`): the logits a softmax readout must produce, drawn in the
# coordinate frame the task's SVD provides.

# %%
import os, sys
from pathlib import Path
import numpy as np, torch

# Project root, independent of the kernel's cwd (headless `jupytext --execute` can launch it outside
# the repo): the uv venv sits at the repo root. chdir so Hydra resolves conf/, and add it to sys.path.
ROOT = Path(sys.prefix).parent            # <repo>/.venv -> <repo>
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.data import categorical
from src.code.geometry import factor_logcond
from src.code.eval import cka, cca
from src.code.analysis_utils import align_representations, heldout_decode_r2
from src.code.train import train_inline
from src.code import plot_utils
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. The predicted geometry (closed form)
#
# The triptych of the geometry notebook, for this task: **structure** — the latent space is the
# simplex of categoricals drawn in log space, the coordinates a softmax readout uses: the
# triangle's corners stretch outward (a dominant category), its edges push away (a missing
# category means an unboundedly negative logit); **log-conditional** `log p(x|theta)`
# over prior samples — the matrix we SVD; **embedding** — the SVD recovers that same shape from
# the matrix alone, and its column side gives one direction per category (arrows) — the
# prototypical objects each context will combine.

# %%
K, ALPHA = 3, 1.0                                                     # = conf/setting/dirichlet.yaml
Lp, info = categorical(K=K, n_items=3000, alpha=ALPHA, seed=1)        # log p(x|theta), theta ~ prior
_, E_rows, D, S_p, _ = factor_logcond(Lp)
r = S_p > 1e-9 * S_p[0]
E_rows = E_rows[:, r]                                                 # a code per sampled theta
cat_emb = np.linalg.pinv(D[r])                                        # (K, 2): one embedding per category

verts = np.stack([np.cos(2 * np.pi * np.arange(K) / K), np.sin(2 * np.pi * np.arange(K) / K)], 1)
order = np.argsort(np.arctan2(*(info["P"] @ verts).T[::-1]))          # sort rows for a readable heatmap
fig_pred, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))
# uniform grid over the log (logit) plane, masked to the simplex with every category >= 1%:
# each grid point is a categorical (softmax), so the filled shape IS the space of logit codes
g = np.linspace(-4.7, 4.7, 140)
Yg = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
lg = (2 / 3) * Yg @ verts.T                                           # sum-zero logits at display coords Yg
Th = np.exp(lg); Th /= Th.sum(1, keepdims=True)
m = Th.min(1) >= 0.01
ax[0].scatter(Yg[m, 0], Yg[m, 1], s=4, c=Th[m])
ax[0].set_xlim(-5.2, 5.2); ax[0].set_ylim(-5.2, 5.2)
ax[0].set_title('structure: the simplex in log space\n(color = theta, RGB)')
plot_utils.conditional_heatmap(ax[1], np.clip(Lp[order], -6, 0), title='log-conditional log p(x|theta)\n(the matrix we SVD)',
                               xlabel='category x', ylabel=r'sampled $\theta$ (sorted)', aspect='auto')
# display gauge: rotate the SVD plane so the category directions match panel (a)'s layout
from scipy.linalg import orthogonal_procrustes
Rrot, _ = orthogonal_procrustes(cat_emb, verts)
E_show, cat_show = E_rows @ Rrot, cat_emb @ Rrot
arrows = cat_show / np.linalg.norm(cat_show, axis=1, keepdims=True) * 0.5 * np.abs(E_show).max()
ax[2].scatter(E_show[:, 0], E_show[:, 1], s=4, c=info["P"])
for k in range(K):
    ax[2].annotate('', arrows[k], (0, 0), arrowprops=dict(arrowstyle='->', lw=1.8, color='0.15'))
    ax[2].annotate(f'category {k}', arrows[k] * 1.12, ha='center')
ax[2].set_title('embedding: theta codes + the 3\ncategory directions (SVD)')
[a.set_aspect('equal') for a in (ax[0], ax[2])]
[plot_utils.FIG.clean(a) for a in (ax[0], ax[2])]
fig_pred.tight_layout()                                               # saved once we have a run config (below)

# %% [markdown]
# ## 2. Train inline; behavior reaches Bayes
#
# `runner=dirichlet_transformer`: one layer, one head — the Bayes update is a running average of
# token indicators, within reach of one attention layer. We train once and sample a fixed set of
# sequences everything below reuses.

# %%
model, ds, ev, hist, config = train_inline(['runner=dirichlet_transformer'])
plot_utils.save_figure(fig_pred, plot_utils.figure_path(config, "predicted_geometry"))

B, L = 128, 64                                                        # fixed eval sequences
ds.rng.manual_seed(0); ds.np_rng = np.random.default_rng(0)   # reseed: training consumes the generators
theta = torch.tensor(ds.np_rng.dirichlet(ds.alpha.cpu().numpy(), size=B),
                     dtype=torch.float32, device=ds.device)
x = torch.multinomial(theta, L, replacement=True, generator=ds.rng)
ids = torch.cat([torch.full((B, 1), ds.start_token, dtype=torch.long, device=ds.device), x], 1)
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
H = out['hidden_states'][-1][:, 1:].reshape(-1, model.config.hidden_size).cpu().numpy()
p = ds.posterior_predictive(x); pf = p.reshape(-1, K).cpu().numpy()   # exact posterior predictive

p_model = torch.softmax(out['logits'][:, :-1][..., ds.item_tokens], -1).cpu().numpy()
prior = (ds.alpha / ds.alpha.sum())
p_bayes = torch.cat([prior.expand(B, 1, -1), p[:, :-1]], 1).cpu().numpy()
kl = lambda a, b: float((a * (np.log(a + 1e-9) - np.log(b + 1e-9))).sum(-1).mean())
print(f'KL(bayes || model) = {kl(p_bayes, p_model):.4f}   '
      f'(flat-prior baseline {kl(p_bayes, prior.cpu().numpy()[None, None]):.3f})')

# %% [markdown]
# ## 3. Belief dynamics over the context
#
# Each context is a weighted sum of the 3 category embeddings, weight = the centered log of the
# Bayes output (`ds.logit_code`). Here the output *is* the belief, so the logit code is pinned in
# basis (CKA), not just up to a linear map.
#
# First the full dynamics across all sequences: the logit code's own cloud beside the
# transformer's hidden state aligned into that plane (CCA-level linear map), every position of
# every sequence, colored by dominant category x evidence quartile. Then two sequences side by
# side, Bayes vs network.

# %%
lc = ds.logit_code(x).reshape(-1, K).cpu().numpy()                    # centered log of the Bayes outputs
code = lc @ cat_emb                                                   # each context: weighted sum of 3 embeddings
tpos = np.tile(np.arange(1, L + 1), B)                                # evidence count t per position

R2_p = heldout_decode_r2(H, pf, groups=np.arange(len(H)))
print(f'predictive linearly represented:  probe R2 = {R2_p:.3f}')
print(f'hidden vs logit code:             CCA = {cca(H, lc):.3f}   CKA = {cka(H, lc):.3f}')

labels = pf.argmax(1) * 4 + np.searchsorted([8, 16, 32], tpos)        # dominant category x t-quartile
fig = plot_utils.plot_prediction_pair(H, lc, 'logit code', colors=labels)
plot_utils.save_figure(fig, plot_utils.figure_path(config, "belief_dynamics"))

ncode, _ = align_representations(H, code, method='linear')            # network hidden, mapped
fig = plot_utils.plot_trajectory_overlay(
    code.reshape(B, L, 2), ncode.reshape(B, L, 2), seqs=(0, 1),
    labels=('logit code (Bayes)', 'network hidden'),
    suptitle=f'the network hidden tracks the posterior over the context '
             f'(CCA = {cca(H, lc):.2f}; color = time)')
plot_utils.save_figure(fig, plot_utils.figure_path(config, "belief_trajectory"))

# %% [markdown]
# ## 4. Memorizing vs generalizing: one dimension per task, or the family rank?
#
# Train the same architecture on a prior of **8 fixed categoricals** (`runner=
# categorical_transformer`): exact Bayes is now filtering over 8 memorized tasks — the
# "memorizing predictor" (compare the context to each task, output the posterior-weighted
# average). Two hypotheses about its hidden state:
#
# - **Rational compression**: the posterior over the 8 atoms is a function of the same 2-D
#   sufficient statistic (counts), and mixtures of categoricals stay categorical — so the hidden
#   spectrum should match the continuous-prior model's (~2 dims + corrections), with "which
#   atom" read off *position* in the plane.
# - **Per-task lookup** (task-vector intuition): the network allocates ~one direction per
#   memorized task, so the M=8 spectrum should carry ~7 healthy extra singular values that the
#   continuous-prior model lacks.

# %%
model8, ds8, ev8, hist8, config8 = train_inline(['runner=categorical_transformer'])

ds8.rng.manual_seed(0)
theta8 = ds8.sample_emission(B)
x8 = torch.multinomial(theta8, L, replacement=True, generator=ds8.rng)
ids8 = torch.cat([torch.full((B, 1), ds8.start_token, dtype=torch.long, device=ds8.device), x8], 1)
with torch.no_grad():
    out8 = model8(ids8, output_hidden_states=True)
H8 = out8['hidden_states'][-1][:, 1:].reshape(-1, model8.config.hidden_size).cpu().numpy()

# behavior: the memorizing net should match filtering over the 8 atoms, NOT the conjugate rule
p8 = ds8.posterior_predictive(x8)
prior8 = ds8.prior_predictive
bayes8 = torch.cat([prior8.expand(B, 1, -1), p8[:, :-1]], 1).cpu().numpy()
p_model8 = torch.softmax(out8['logits'][:, :-1][..., ds8.item_tokens], -1).cpu().numpy()
counts8 = torch.cumsum(torch.nn.functional.one_hot(x8, ds8.K).float(), dim=1)
tvec = torch.arange(1, L + 1, device=x8.device, dtype=torch.float32).view(1, -1, 1)
conj = ((1.0 + counts8) / (float(ds8.K) + tvec))
conj = torch.cat([torch.full((B, 1, ds8.K), 1 / ds8.K, device=x8.device), conj[:, :-1]], 1).cpu().numpy()
print(f'KL(filtering bayes || model)    = {kl(bayes8, p_model8):.4f}   '
      f'(prior baseline {kl(bayes8, prior8.cpu().numpy()[None, None]):.3f})')
print(f'KL(filtering bayes || conjugate) = {kl(bayes8, conj):.4f}   '
      f'(what a generalizing predictor would output)')

# %% [markdown]
# The verdict is in the hidden-state spectra: same architecture, same context length, only the
# prior differs. The Bayes logit code is rank 2 in both cases (mixture closure), so any extra
# dimensions in the M=8 model are representational choices, not task requirements.

# %%
def spectrum(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    pr = float((s ** 2).sum() ** 2 / (s ** 4).sum())
    return s / s[0], pr

s_dense, pr_dense = spectrum(H)
s_mem, pr_mem = spectrum(H8)
s_code, _ = spectrum(lc)
print(f'participation ratio:  continuous prior {pr_dense:.2f}   8-atom prior {pr_mem:.2f}')
print(f'dims above 5% of top: continuous {int((s_dense > .05).sum())}   '
      f'8-atom {int((s_mem > .05).sum())}   (Bayes logit code: {int((s_code > .05).sum())})')

w8 = ds8.posterior_weights(x8).cpu().numpy()
map_atom = w8.argmax(-1).reshape(-1)
U8, S8, _ = np.linalg.svd(H8 - H8.mean(0), full_matrices=False)
proj8 = (U8 * S8)[:, :2]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].semilogy(np.arange(1, 21), s_dense[:20], 'o-', ms=4, label='continuous prior (generalizing)')
ax[0].semilogy(np.arange(1, 21), s_mem[:20], 's-', ms=4, label='8-atom prior (memorizing)')
ax[0].axvline(2.5, ls='--', color='0.7', lw=1)
ax[0].annotate('Bayes code rank', (2.6, 0.5), fontsize=8, color='0.4')
ax[0].set_xlabel('component'); ax[0].set_ylabel(r'$\sigma_i / \sigma_1$')
ax[0].legend(fontsize=8)
ax[0].set_title('hidden-state spectra: does memorization\nbuy one dimension per task?')
plot_utils.FIG.style_axes(ax[0])
ax[1].scatter(proj8[:, 0], proj8[:, 1], c=map_atom, cmap='tab10', vmax=9, s=5)
ax[1].set_title('memorizing model, hidden PCA\n(color = Bayes-MAP atom)')
plot_utils.FIG.clean(ax[1])
fig.tight_layout()
plot_utils.save_figure(fig, plot_utils.figure_path(config8, "memorizing_spectrum"))

# %%
