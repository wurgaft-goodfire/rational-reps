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
# # In-context belief dynamics: the hidden-Markov task
#
# Latent states walk a ring and emit observable tokens; the network sees only the tokens. The
# predicted geometry is read off the latent transition in closed form, exactly as in the geometry
# notebook — but a single token is uninformative about the state, so nothing lands on that
# geometry until context accumulates: the optimal predictor's belief `p(s_t | x_{1:t})` (the
# forward algorithm) walks the predicted ring, and the network's hidden state should walk it too.

# %%
import os, sys
from pathlib import Path
import numpy as np, torch

# Project root, independent of the kernel's cwd (headless `jupytext --execute` can launch it outside
# the repo): the uv venv sits at the repo root. chdir so Hydra resolves conf/, and add it to sys.path.
ROOT = Path(sys.prefix).parent            # <repo>/.venv -> <repo>
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.data import adjacency_conditional, structure_adjacency
from src.code.geometry import predicted_geometry
from src.code.eval import cka, cca
from src.code.analysis_utils import align_representations, heldout_decode_r2
from src.code.train import train_inline
from src.code import plot_utils
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. The predicted geometry (closed form)
#
# Factor the latent transition's log-conditional — no training. The ring's translation symmetry
# gives a degenerate top Fourier pair, and the top-2 code lands on a circle. In context, the
# belief will walk this ring.

# %%
spec = {"structure": "ring", "n": 24, "gamma": 0.8, "cooccurrence": "diffusion"}  # = conf/setting/hmm.yaml
L_lat = adjacency_conditional(structure_adjacency(spec), spec["gamma"], spec["cooccurrence"])
E_lat = predicted_geometry(L_lat)
fig_pred = plot_utils.plot_predicted_geometry(spec, L_lat, E_lat)     # saved once we have a run config (below)

# %% [markdown]
# ## 2. Train inline; behavior reaches Bayes
#
# `runner=hmm_transformer`: one layer, one head, trained on emissions only. We widen the bump
# emissions (`emission_sigma=2.5`) so states alias: the belief then genuinely differs from the
# output logit code (CCA between them ~0.7, vs ~0.96 at the default width), making §3's
# comparison discriminative. We train once and sample a fixed set of sequences everything below
# reuses.

# %%
model, ds, ev, hist, config = train_inline(['runner=hmm_transformer', 'emission_sigma=2.5'])
plot_utils.save_figure(fig_pred, plot_utils.figure_path(config, "predicted_geometry"))
n = ds.n_states

B, L = 128, 40                                                        # fixed eval sequences
ds.rng.manual_seed(0)                       # reseed: training consumes the generator on a fresh run
s = torch.empty(B, L, dtype=torch.long, device=ds.device)
s[:, 0] = torch.multinomial(ds.prior, B, replacement=True, generator=ds.rng)
for t in range(1, L):
    s[:, t] = torch.multinomial(ds.transition[s[:, t-1]], 1, generator=ds.rng).squeeze(1)
x = torch.multinomial(ds.P[s.reshape(-1)], 1, generator=ds.rng).reshape(B, L)
ids = torch.cat([torch.full((B, 1), ds.start_token, dtype=torch.long, device=ds.device), x], 1)
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
H = out['hidden_states'][-1][:, 1:].reshape(-1, model.config.hidden_size).cpu().numpy()
bel = ds.forward_algorithm(x); belf = bel.reshape(-1, n).cpu().numpy()  # exact beliefs

p_model = torch.softmax(out['logits'][:, :-1][..., ds.item_tokens], -1).cpu().numpy()
p_bayes = torch.cat([(ds.prior @ ds.P).expand(B, 1, -1), (bel[:, :-1] @ ds.transition) @ ds.P], 1).cpu().numpy()
kl = lambda p, q: float((p * (np.log(p + 1e-9) - np.log(q + 1e-9))).sum(-1).mean())
print(f'KL(bayes || model) = {kl(p_bayes, p_model):.4f}   (marginal baseline '
      f'{kl(p_bayes, (ds.prior @ ds.P).cpu().numpy()[None, None]):.3f})')

# %% [markdown]
# ## 3. Belief dynamics over the context
#
# Each context is the belief-weighted combination of the latent-state embeddings,
# `code_t = belief_t @ E_lat` — the forward algorithm walking the predicted ring. The network's
# hidden state tracks the same path through one linear readout.
#
# The latent belief is never output, so no basis is pinned for it — linear equivalence (probe,
# CCA) is all the task demands, and CKA against the raw simplex is expectedly lower. What *is*
# pinned in basis is the centered log of the Bayes **output** `log(belief @ T @ Em)` (the logits
# the softmax readout must produce, as in the Dirichlet notebook) — and there the hidden matches
# at the basis level.
#
# First the full dynamics across all sequences: for each prediction (belief, output logit code),
# its own cloud beside the transformer's hidden state aligned into that plane (CCA-level linear
# map), every position of every sequence, colored by the true latent state. Then two sequences
# side by side, Bayes vs network.

# %%
q = ((bel @ ds.transition) @ ds.P).double().reshape(-1, ds.n_symbols).cpu().numpy()   # Bayes output
lq = np.log(q); lq -= lq.mean(1, keepdims=True)                       # output logit code
st = s.reshape(-1).cpu().numpy()                                      # true latent state per position

def cca_var(Y, frac=0.99):
    """Mean canonical correlation on the top PCs of Y carrying ``frac`` of its variance — plain
    cca() whitens every direction equally, so the aliased belief's near-empty tail directions
    (which the network has no reason to encode) would otherwise drag the mean down."""
    Yc = Y - Y.mean(0); _, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), frac)) + 1
    return cca(H, Yc @ Vt[:k].T), k

R2 = heldout_decode_r2(H, belf, groups=np.arange(len(H)))
(c_b, k_b), (c_q, k_q) = cca_var(belf), cca_var(lq)
print(f'belief linearly represented:  probe R2 = {R2:.3f}')
print(f'hidden vs belief:             CCA = {c_b:.3f} (top-{k_b} PCs, 99% var; all dims {cca(H, belf):.3f})   '
      f'CKA = {cka(H, belf):.3f}   (basis unconstrained)')
print(f'hidden vs output logit code:  CCA = {c_q:.3f} (top-{k_q} PCs, 99% var; all dims {cca(H, lq):.3f})   '
      f'CKA = {cka(H, lq):.3f}   (basis pinned by softmax)')

fig = plot_utils.plot_prediction_pair(H, belf, 'belief (forward algorithm)', colors=st)
plot_utils.save_figure(fig, plot_utils.figure_path(config, "belief_dynamics_belief"))
fig = plot_utils.plot_prediction_pair(H, lq, 'output logit code', colors=st)
plot_utils.save_figure(fig, plot_utils.figure_path(config, "belief_dynamics_logit"))

code = (belf @ E_lat[:, :2])                                          # belief-weighted ring position
ncode, _ = align_representations(H, code, method='linear')            # network hidden, mapped
fig = plot_utils.plot_trajectory_overlay(
    code.reshape(B, L, 2), ncode.reshape(B, L, 2), seqs=(0, 1),
    labels=('belief code (forward algorithm)', 'network hidden'),
    suptitle=f'the network hidden tracks the belief over the context '
             f'(CCA = {cca(H, belf):.2f}; color = time)')
plot_utils.save_figure(fig, plot_utils.figure_path(config, "belief_trajectory"))

# %%
