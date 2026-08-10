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
# # In-context belief dynamics, Gaussian case: when mixtures leave the family
#
# The exchangeable task with `N(mu, sigma)` hypotheses on a binned line. Two things are new here
# relative to the categorical/von Mises cases, both because Gaussians are **not closed under
# mixing** (a mixture of two distant Gaussians is bimodal, not Gaussian):
#
# 1. **Off-sheet (uncertainty) dimensions.** The logit code is the centered log of the posterior
#    *predictive* — a log-of-mixture. Early in context the mixture is genuinely multi-hypothesis,
#    so its log leaves the family's 2D span `{x, x^2}`; late in context the posterior concentrates,
#    the mixture collapses to one Gaussian, and the code lands on the sheet. Predicted: the
#    off-sheet variance fraction of the code decays with position.
# 2. **Curved trajectories.** The evidence direction for bin x (its prototype `phi(x) = (x, x^2)`)
#    differs from its certainty ray (`(2x, -1)`, the sigma -> 0 limit) because the prototypes have
#    unequal norms — unlike the von Mises circle. Belief trajectories must bend from the evidence
#    direction onto the natural-parameter half-plane `(mu/sigma^2, -1/(2*sigma^2))`.

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
# `runner=gaussian_transformer`: 64 means x 16 sigmas on 36 support bins — the grid is finer than
# the end-of-context posterior width, so the finite pool is indistinguishable from the continuous
# prior over the whole context.

# %%
model, ds, ev, hist, config = train_inline(['runner=gaussian_transformer'])
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
# The triptych: **structure** — the natural parameters `(mu/sigma^2, -1/(2*sigma^2))`, a fan in
# the lower half-plane (certainty = sigma -> 0 is off to infinity along the `(2x, -1)` rays);
# **log-conditional** over the atom pool; **embedding** — the SVD (base measure dropped) recovers
# the fan, and the decoder rows come out as the sufficient statistic: a linear and a quadratic
# function of the support.

# %%
Lp = ds.atom_logcond                                                  # log p(x|theta), the atom pool
# threshold at 1e-4: the float32 tail bins (log-probs ~ -100) leave quantization crumbs ~1e-5
_, E_fan, D_suff, S_d, _ = factor_logcond(Lp, drop_base_measure=True)
rd = S_d > 1e-4 * S_d[0]
print(f'rank with base measure dropped: {rd.sum()}   (predicted: 2, the sufficient statistic)')

mu, sig = ds.info["params"].T
nat = ds.info["natural"]                                              # (mu/sigma^2, -1/(2 sigma^2)) up to sign
nat = np.stack([nat[:, 0], -0.5 * nat[:, 1]], 1)                      # builder stores 1/sigma^2; flip to -1/(2s^2)
fig_pred, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))
sc = ax[0].scatter(nat[:, 0], nat[:, 1], s=8, c=mu, cmap='coolwarm')
fig_pred.colorbar(sc, ax=ax[0], label='mu')
ax[0].set_title('structure: natural parameters\n(half-plane fan; certainty at infinity)')
plot_utils.conditional_heatmap(ax[1], np.clip(Lp, -12, 0), title='log-conditional log p(x|theta)\n(the matrix we SVD)',
                               xlabel='support bin x', ylabel=r'atom $\theta$', aspect='auto')
from scipy.linalg import orthogonal_procrustes
E2 = E_fan[:, :2]
Rrot, _ = orthogonal_procrustes(E2 / np.abs(E2).max(), nat / np.abs(nat).max())
E_show = (E2 / np.abs(E2).max()) @ Rrot
ax[2].scatter(E_show[:, 0], E_show[:, 1], s=8, c=mu, cmap='coolwarm')
ax[2].set_title('embedding: atom codes (SVD)\nrecover the fan')
[plot_utils.FIG.clean(a) for a in (ax[0], ax[2])]
fig_pred.tight_layout()
plot_utils.save_figure(fig_pred, plot_utils.figure_path(config, "predicted_geometry"))

fig_suff, ax = plt.subplots(figsize=(5, 3))
grid = ds.info["x"]
for j in range(2):
    ax.plot(grid, D_suff[j], label=f'decoder row {j + 1}')
ax.set_xlabel('support value x'); ax.legend()
ax.set_title('the decoder is the sufficient statistic:\nlinear and quadratic in x')
plot_utils.FIG.clean(ax); fig_suff.tight_layout()
plot_utils.save_figure(fig_suff, plot_utils.figure_path(config, "sufficient_statistic"))

# %% [markdown]
# ## 3. Belief dynamics; the evidence accumulators
#
# The logit code cloud vs the transformer's hidden state aligned into it, plus the linear
# concepts: the hidden state should carry the running sufficient statistics
# `(sum x_s, sum x_s^2, t)` — the family's evidence accumulators.

# %%
lc = ds.logit_code(x).reshape(-1, K).cpu().numpy()                    # centered log of the Bayes outputs
tpos = np.tile(np.arange(1, L + 1), B)

vals = torch.tensor(grid, dtype=torch.float32, device=ds.device)[x]   # bin values of the samples
m1 = torch.cumsum(vals, 1).reshape(-1).cpu().numpy()                  # running sum x
m2 = torch.cumsum(vals ** 2, 1).reshape(-1).cpu().numpy()             # running sum x^2
moments = np.stack([m1, m2, tpos], 1)

R2_p = heldout_decode_r2(H, pf, groups=np.arange(len(H)))
R2_m = heldout_decode_r2(H, moments, groups=np.arange(len(H)))
print(f'predictive linearly represented:   probe R2 = {R2_p:.3f}')
print(f'evidence accumulators (sum x, sum x^2, t): probe R2 = {R2_m:.3f}')
print(f'hidden vs logit code:              CCA = {cca(H, lc):.3f}   CKA = {cka(H, lc):.3f}')

pmean = pf @ grid                                                     # posterior-predictive mean
labels = np.digitize(pmean, np.quantile(pmean, [.25, .5, .75])) * 4 + np.searchsorted([8, 16, 32], tpos)
fig = plot_utils.plot_prediction_pair(H, lc, 'logit code', colors=labels)
plot_utils.save_figure(fig, plot_utils.figure_path(config, "belief_dynamics"))

# %% [markdown]
# ## 4. Off-sheet (uncertainty) dimensions decay with evidence
#
# The family sheet is the span of the log-conditional's own modes (base measure + the two
# sufficient-statistic directions). Early-context codes are logs of multi-hypothesis mixtures and
# leave that span; as the posterior concentrates the mixture collapses to a single Gaussian and
# the residual goes to zero. The categorical/von Mises tasks cannot show this: there the
# predictive never leaves the family.

# %%
_, _, D_fam, S_f, _ = factor_logcond(Lp)                              # base measure kept: the full sheet
rf = S_f > 1e-4 * S_f[0]
basis = D_fam[rf] - D_fam[rf].mean(1, keepdims=True)                  # centered, to match the centered code
proj = basis.T @ np.linalg.solve(basis @ basis.T, basis @ lc.T)       # least-squares projection onto the sheet
off = ((lc.T - proj) ** 2).sum(0) / (lc ** 2).sum(1)                  # off-sheet variance fraction per point
off_t = off.reshape(B, L).mean(0)
print(f'off-sheet variance fraction: {off_t[0]:.3f} at t=1  ->  {off_t[-1]:.3f} at t={L}')

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(np.arange(1, L + 1), off_t)
ax.set_ylim(0, None)
ax.set_xlabel('position t (evidence count)'); ax.set_ylabel('off-sheet variance fraction')
ax.set_title(f'uncertainty dimensions: the logit code starts off the\n'
             f'family sheet (rank {rf.sum()}) and collapses onto it')
plot_utils.FIG.style_axes(ax); fig.tight_layout()
plot_utils.save_figure(fig, plot_utils.figure_path(config, "off_sheet_decay"))

# %% [markdown]
# ## 5. Curved trajectories onto the half-plane
#
# Trajectories in the natural-parameter plane, atoms in the background: Bayes (dots) and the
# network's hidden state aligned into the same plane (crosses). Because the prototypes `(x, x^2)`
# have unequal norms, evidence pushes and certainty rays disagree — the trajectories bend as they
# descend onto the fan, where the von Mises ones flew straight at their atom.

# %%
suff_emb = np.linalg.pinv(D_suff[:2])                                 # (K, 2): code-plane coords
code = lc @ suff_emb
ncode, _ = align_representations(H, code, method='linear')            # network hidden, mapped

E_bg = E_fan[:, :2]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for a, seq in zip(axes, (0, 1)):
    a.scatter(E_bg[:, 0], E_bg[:, 1], s=6, c='0.85', label='atoms (hypotheses)')
    c, n = code.reshape(B, L, 2)[seq], ncode.reshape(B, L, 2)[seq]
    a.plot(c[:, 0], c[:, 1], '-', c='0.6', lw=0.8, zorder=1)
    a.scatter(c[:, 0], c[:, 1], c=np.arange(L), cmap='viridis', s=26, zorder=2, label='logit code (Bayes)')
    a.scatter(n[:, 0], n[:, 1], c=np.arange(L), cmap='viridis', s=26, marker='x', zorder=2,
              label='network hidden')
    # zoom to the trajectory (the full atom fan is much wider and would crush it to a squiggle)
    pts = np.vstack([c, n])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.35 * (hi - lo)
    a.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); a.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    a.set_title(f'sequence {seq}')
    plot_utils.FIG.clean(a)
axes[0].legend(loc='lower left', fontsize=8)
fig.suptitle(f'belief trajectories bend onto the natural-parameter fan '
             f'(CKA = {cka(H, lc):.2f}; color = time)')
fig.tight_layout()
plot_utils.save_figure(fig, plot_utils.figure_path(config, "belief_trajectory"))

# %%
