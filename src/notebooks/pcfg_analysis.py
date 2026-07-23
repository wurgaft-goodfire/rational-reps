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
# # PCFG: the belief lives on the latent tree geometry
#
# The tree generalization of the HMM notebook. Latent states diffuse *down a tree* (branching
# diffusion on a ring); each leaf emits an observable terminal. The transformer sees only the flat
# string and must parse it left-to-right. The Bayes-optimal per-position belief — the distribution
# over the latent state that emits the *next* terminal (`pcfg_next_token`, the left-to-right prefix
# parser) — lives on the same predicted ring geometry, but its trajectory now reflects the parse:
# it drifts locally inside a constituent and *jumps* at constituent boundaries (a stack pop).

# %%
import os, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(sys.prefix).parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.geometry import predicted_geometry
from src.code.data import pcfg_next_token
from src.code.eval import cka, cca, rsa
from src.code.train import train_inline
from src.code import plot_utils
import matplotlib.pyplot as plt

# %% [markdown]
# ## Setup: train a (4-layer) transformer on ring branching-diffusion PCFG emissions
# A 1-layer transformer cannot parse; belief-tracking here needs depth.

# %%
model, ds, ev, hist, config = train_inline(
    ['runner=pcfg_transformer', 'name=pcfg_smoke', 'training_kwargs.train_steps=2500',
     'wandb.enabled=false', 'eval_kwargs.output_hidden_states=true'])
m = ds.m
E_lat = predicted_geometry(ds.latent_conditional)                      # predicted latent ring

# Fixed eval sequences (sample once; reuse everywhere below).
B = 96
seqs = [ds._sample_sentence()[0] for _ in range(B)]
maxk = max(len(s) for s in seqs)
dists = np.zeros((B, maxk + 1, ds.num_dims)); beliefs = np.zeros((B, maxk + 1, m))
lens = np.array([len(s) for s in seqs])
for b, s in enumerate(seqs):
    d, be = pcfg_next_token(np.array(s, dtype=int), ds.g)
    dists[b, :len(s) + 1] = d; beliefs[b, :len(s) + 1] = be

# Build the token stream fed to the model, and read hidden states at the terminal positions.
Lin = ds.seq_len
ids = np.full((B, Lin), ds.pad_token); ids[:, 0] = ds.start_token
for b, s in enumerate(seqs):
    ids[b, 1:1 + len(s)] = s; ids[b, 1 + len(s)] = ds.eos_token
ids = torch.tensor(ids, dtype=torch.long, device=ds.device)
with torch.no_grad():
    out = model(ids, output_hidden_states=True)

# Flatten to per-position rows over VALID prediction positions (start..last terminal => predicts
# next token / EOS). Hidden at position p predicts token p+1; belief[t] aligns with hidden[t].
H, BEL, TT, PREDIST = [], [], [], []
for b, s in enumerate(seqs):
    k = len(s)
    for t in range(k + 1):                                             # t = prefix length 0..k
        H.append(out['hidden_states'][-1][b, t].cpu().numpy())         # hidden after reading t tokens
        BEL.append(beliefs[b, t]); PREDIST.append(dists[b, t])
        TT.append(t)
H = np.array(H); BEL = np.array(BEL); PREDIST = np.array(PREDIST); TT = np.array(TT)
print(f"trained. {len(seqs)} sequences, mean len {lens.mean():.1f}, {len(H)} prediction positions")

# %% [markdown]
# ## 1. The belief geometry (closed form)
# The predicted latent geometry is the ring; the exact belief (the next-emitter posterior from the
# prefix parser) lives on that ring.

# %%
code = BEL @ E_lat
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
plot_utils.scatter_geometry(ax[0], E_lat[:, :2], 'predicted latent ring')
ax[1].scatter(code[:, 0], code[:, 1], s=6, c='0.55'); ax[1].set_aspect('equal')
ax[1].set_title('belief trajectory on the ring'); plot_utils.FIG.clean(ax[1])
fig.tight_layout()
plot_utils.save_figure(fig, plot_utils.figure_path(config, "belief_geometry"))

# %% [markdown]
# ## 2. Behavior reaches Bayes
# KL(Bayes || model) over the next-token distribution (terminals + EOS), vs the marginal baseline.

# %%
logits = out['logits'][:, :-1][..., ds.item_tokens]                    # map to terminals + EOS
pm = torch.softmax(logits, -1).cpu().numpy()
kl = lambda p, q: float((p * (np.log(p + 1e-9) - np.log(q + 1e-9))).sum(-1))
kls = []
marg = PREDIST.mean(0)
kl_marg = []
for b, s in enumerate(seqs):
    for t in range(len(s) + 1):
        kls.append(kl(dists[b, t], pm[b, t]))
        kl_marg.append(kl(dists[b, t], marg))
print(f'KL(bayes || model) = {np.mean(kls):.4f}   (marginal baseline {np.mean(kl_marg):.3f})')

# %% [markdown]
# ## 3. The belief is recovered in the deep representation
# Linear probe + CCA of the hidden states against the exact belief; the trajectory overlay shows the
# network's hidden tracking the belief — including the jumps at constituent boundaries (stack pops).

# %%
ntr = int(0.8 * len(H)); lam = 1e-2
Xtr = np.c_[H[:ntr], np.ones(ntr)]; Xte = np.c_[H[ntr:], np.ones(len(H) - ntr)]
W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ BEL[:ntr]); pred = Xte @ W
R2 = 1 - ((BEL[ntr:] - pred) ** 2).sum() / ((BEL[ntr:] - BEL[ntr:].mean(0)) ** 2).sum()
print(f'belief linearly represented:  probe R2 = {R2:.3f}   CCA(hidden, belief) = {cca(H, BEL):.3f}')

bcode = BEL @ E_lat[:, :2]
Wc = np.linalg.lstsq(np.c_[H, np.ones(len(H))], bcode, rcond=None)[0]
ncode = np.c_[H, np.ones(len(H))] @ Wc
# plot a few whole sequences as paths (belief vs network), colored by time.
fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
starts = np.concatenate([[0], np.cumsum(lens + 1)])
for a, si in zip(ax, [0, 1]):
    sl = slice(starts[si], starts[si + 1]); L = lens[si] + 1
    a.plot(bcode[sl, 0], bcode[sl, 1], '-', color='0.8', lw=1, zorder=0)
    a.scatter(bcode[sl, 0], bcode[sl, 1], c=range(L), cmap='viridis', s=34, zorder=1, label='belief (prefix parser)')
    a.scatter(ncode[sl, 0], ncode[sl, 1], c=range(L), cmap='viridis', marker='x', s=30, lw=1.4, zorder=2, label='network hidden')
    a.set_aspect('equal'); plot_utils.FIG.clean(a); a.set_title(f'sequence {si}')
ax[0].legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f'the network hidden tracks the belief trajectory (CCA = {cca(H, BEL):.2f}; color = time)', y=1.02)
fig.tight_layout()
plot_utils.save_figure(fig, plot_utils.figure_path(config, "belief_trajectory"))

# %% [markdown]
# ## 4. The output readout recovers the latent geometry
# With identity emissions each terminal is a latent state, so the unembedding over terminals should
# recover the predicted latent ring.

# %%
W_out = model.get_output_embeddings().weight.detach().cpu().numpy()[:ds.V]
print(f'unembedding vs predicted latent geometry:  CKA = {cka(W_out, E_lat):.3f}  '
      f'RSA = {rsa(W_out, E_lat):.3f}  CCA = {cca(W_out, E_lat):.3f}')
sym_spec = {"structure": "ring", "n": ds.V}
fig_dec = plot_utils.plot_decoder_panels(E_lat, {"transformer": W_out}, spec=sym_spec)
plot_utils.save_figure(fig_dec, plot_utils.figure_path(config, "readout_geometry"))
