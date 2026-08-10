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
# # In-context belief dynamics for any family: the exchangeable task
#
# The Dirichlet-categorical task generalized: each sequence draws an emission distribution from
# **any family's atom pool** (here von Mises on a binned circle) and emits iid tokens. The
# Bayes-optimal output is the posterior predictive under exact filtering over the atoms, and the
# predicted representation is its **centered log** (`ds.logit_code`) — the logits a softmax
# readout must produce. Late in the context the posterior concentrates on one atom, so the logit
# code collapses onto the family's natural-parameter surface (for von Mises,
# `kappa cos(x - mu) = (kappa cos mu) cos x + (kappa sin mu) sin x`: circles of radius kappa in
# the plane spanned by the sufficient statistic `(cos x, sin x)`). Early in the context it is the
# log of a *mixture* of atoms, off that surface — the belief flows from the prior-predictive
# point outward to the natural-parameter circles.

# %%
import os, sys
from pathlib import Path
import numpy as np, torch

# Project root, independent of the kernel's cwd (headless `jupytext --execute` can launch it outside
# the repo): the uv venv sits at the repo root. chdir so Hydra resolves conf/, and add it to sys.path.
ROOT = Path(sys.prefix).parent            # <repo>/.venv -> <repo>
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.geometry import factor_logcond
from src.code.eval import cka, cca
from src.code.analysis_utils import align_representations, heldout_decode_r2
from src.code.train import train_inline
from src.code import plot_utils
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Train inline; behavior reaches Bayes
#
# `runner=exchangeable_transformer`: von Mises atoms (96 means x 3 concentrations) on 36 support
# bins — the mean grid is finer than the end-of-context posterior width, so the finite pool is
# indistinguishable from the continuous prior over the whole context. Two layers: filtering over
# atoms needs a smoothed histogram of the context, not just a running average. We train once and
# sample a fixed set of sequences everything below reuses.

# %%
model, ds, ev, hist, config = train_inline(['runner=exchangeable_transformer'])
K = ds.num_dims

B, L = 128, 64                                                        # fixed eval sequences
ds.rng.manual_seed(0)                                                 # reseed: training consumed it
theta = ds.sample_emission(B)
x = torch.multinomial(theta, L, replacement=True, generator=ds.rng)
ids = torch.cat([torch.full((B, 1), ds.start_token, dtype=torch.long, device=ds.device), x], 1)
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
H = out['hidden_states'][-1][:, 1:].reshape(-1, model.config.hidden_size).cpu().numpy()
p = ds.posterior_predictive(x); pf = p.reshape(-1, K).cpu().numpy()   # exact posterior predictive

p_model = torch.softmax(out['logits'][:, :-1][..., ds.item_tokens], -1).cpu().numpy()
prior = ds.prior_predictive
p_bayes = torch.cat([prior.expand(B, 1, -1), p[:, :-1]], 1).cpu().numpy()
kl = lambda a, b: float((a * (np.log(a + 1e-9) - np.log(b + 1e-9))).sum(-1).mean())
print(f'KL(bayes || model) = {kl(p_bayes, p_model):.4f}   '
      f'(prior-predictive baseline {kl(p_bayes, prior.cpu().numpy()[None, None]):.3f})')

# %% [markdown]
# ## 2. The predicted geometry (closed form)
#
# The triptych, for this task: **structure** — the natural parameters
# `(kappa cos mu, kappa sin mu)` of the atom pool, one circle per concentration; **log-conditional**
# `log p(x|theta)` over the pool (the matrix we SVD); **embedding** — the SVD recovers those same
# circles from the matrix alone, and its column side recovers the sufficient statistic: the top two
# decoder rows are `cos x` and `sin x` over the support.

# %%
Lp = ds.atom_logcond                                                  # log p(x|theta), the atom pool
_, E_rows, D, S_p, _ = factor_logcond(Lp)
r = S_p > 1e-6 * S_p[0]
E_rows = E_rows[:, r]                                                 # a code per atom
suff_emb = np.linalg.pinv(D[r])                                       # (K, rank): one embedding per bin
print(f'rank of the centered log-conditional: {r.sum()}   (predicted: 2, the sufficient statistic)')

mu, kappa = ds.info["params"].T
nat = ds.info["natural"]                                              # (kappa cos mu, kappa sin mu)
fig_pred, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))
ax[0].scatter(nat[:, 0], nat[:, 1], s=18, c=mu, cmap='hsv')
ax[0].set_title('structure: natural parameters\n(one circle per kappa; color = mu)')
plot_utils.conditional_heatmap(ax[1], Lp, title='log-conditional log p(x|theta)\n(the matrix we SVD)',
                               xlabel='support bin x', ylabel=r'atom $\theta$', aspect='auto')
# display gauge: rotate the SVD plane so the embedded circles match panel (a)'s layout
from scipy.linalg import orthogonal_procrustes
Rrot, _ = orthogonal_procrustes(E_rows[:, :2], nat)
E_show = E_rows[:, :2] @ Rrot
ax[2].scatter(E_show[:, 0], E_show[:, 1], s=18, c=mu, cmap='hsv')
ax[2].set_title('embedding: atom codes (SVD)\nrecover the kappa circles')
[a.set_aspect('equal') for a in (ax[0], ax[2])]
[plot_utils.FIG.clean(a) for a in (ax[0], ax[2])]
fig_pred.tight_layout()
plot_utils.save_figure(fig_pred, plot_utils.figure_path(config, "predicted_geometry"))

fig_suff, ax = plt.subplots(figsize=(5, 3))
for j in range(2):
    ax.plot(D[j], label=f'decoder row {j + 1}')
ax.set_xlabel('support bin x'); ax.legend()
ax.set_title('the decoder is the sufficient statistic:\ncos x and sin x over the support')
plot_utils.FIG.clean(ax); fig_suff.tight_layout()
plot_utils.save_figure(fig_suff, plot_utils.figure_path(config, "sufficient_statistic"))

# %% [markdown]
# ## 3. Belief dynamics over the context
#
# The full dynamics across all sequences: the logit code's own cloud beside the transformer's
# hidden state aligned into that plane (CCA-level linear map), every position of every sequence,
# colored by posterior mean angle x evidence quartile. Then two sequences side by side, Bayes vs
# network, in the natural-parameter plane: trajectories start at the prior-predictive point and
# flow out to the kappa circles as evidence accumulates.

# %%
lc = ds.logit_code(x).reshape(-1, K).cpu().numpy()                    # centered log of the Bayes outputs
code = lc @ suff_emb[:, :2]                                           # projected into the (cos, sin) plane
tpos = np.tile(np.arange(1, L + 1), B)                                # evidence count t per position

R2_p = heldout_decode_r2(H, pf, groups=np.arange(len(H)))
print(f'predictive linearly represented:  probe R2 = {R2_p:.3f}')
print(f'hidden vs logit code:             CCA = {cca(H, lc):.3f}   CKA = {cka(H, lc):.3f}')

angle_bin = ((np.arctan2(code[:, 1], code[:, 0]) / (2 * np.pi) + 0.5) * 8).astype(int) % 8
labels = angle_bin * 4 + np.searchsorted([8, 16, 32], tpos)           # mean angle x t-quartile
fig = plot_utils.plot_prediction_pair(H, lc, 'logit code', colors=labels)
plot_utils.save_figure(fig, plot_utils.figure_path(config, "belief_dynamics"))

ncode, _ = align_representations(H, code, method='linear')            # network hidden, mapped
fig = plot_utils.plot_trajectory_overlay(
    code.reshape(B, L, 2), ncode.reshape(B, L, 2), seqs=(0, 1),
    labels=('logit code (Bayes)', 'network hidden'),
    suptitle=f'the network hidden tracks the posterior over the context '
             f'(CKA = {cka(H, lc):.2f}; color = time)')
plot_utils.save_figure(fig, plot_utils.figure_path(config, "belief_trajectory"))

# %%
