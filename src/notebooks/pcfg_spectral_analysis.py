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
# # Transformer predictive-state representations for PCFGs
#
# A transformer trained on strings from a PCFG must maintain, left-to-right, the **predictive state** —
# the minimal sufficient statistic of the future given the prefix (the parser stack / counter). The
# canonical basis for that state is the **spectral (Hankel) basis**: build the Hankel matrix
# `H[u,v] = p(string = u·v)` and take its SVD (this is the string-level spectral method — HKZ for HMMs
# is the length-1 special case). We ask, per grammar:
#
# 1. **Rank** — how many predictive states does the grammar require? (rank of `H`)
# 2. **Presence** — is the predictive state linearly in the transformer's hidden? (`CCA`)
# 3. **Basis** — does the transformer use the *canonical spectral basis specifically*, i.e. match it
#    under rotation (`CKA`) better than the raw one-hot sufficient-statistic basis?

# %%
import os, sys
from pathlib import Path
from collections import Counter
import numpy as np, torch
ROOT = Path(sys.prefix).parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.data import pcfg_next_token
from src.code.geometry import hankel_forward_states
from src.code.eval import cca, cka
from src.code.train import train_inline
from src.code import plot_utils
import matplotlib.pyplot as plt


def analyze(overrides, name, label_fn, label_name, cmap="viridis", n_seq=250, basis_cap=120):
    model, ds, ev, hist, config = train_inline(
        list(overrides) + [f"name={name}", "wandb.enabled=false", "eval_kwargs.output_hidden_states=true"])
    # sample eval sequences and read the transformer hidden at every prefix position
    seqs = [ds._sample_sentence()[0] for _ in range(n_seq)]
    ids = np.full((n_seq, ds.seq_len), ds.pad_token); ids[:, 0] = ds.start_token
    for b, s in enumerate(seqs):
        ids[b, 1:1 + len(s)] = s; ids[b, 1 + len(s)] = ds.eos_token
    ids = torch.tensor(ids, dtype=torch.long, device=ds.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    pm = torch.softmax(out["logits"][:, :-1][..., ds.item_tokens], -1).cpu().numpy()

    H, hists, labels, kls = [], [], [], []
    for b, s in enumerate(seqs):
        d, _ = pcfg_next_token(np.array(s, dtype=int), ds.g)
        for t in range(len(s) + 1):
            H.append(out["hidden_states"][-1][b, t].cpu().numpy())
            hists.append(tuple(s[:t])); labels.append(label_fn(s[:t]))
            kls.append(float((d[t] * (np.log(d[t] + 1e-9) - np.log(pm[b, t] + 1e-9))).sum()))
    H = np.array(H)

    # spectral (Hankel) predictive-state basis, from the exact string probabilities. The joint
    # H[u,v]=p(uv) is already implicitly measure-weighted (like the HMM's joint P21); we also build a
    # version reweighted by empirical prefix/suffix frequency (the loss's own weighting).
    Pc = Counter(hists); Sc = Counter(tuple(s[t:]) for s in seqs for t in range(len(s) + 1))
    P = [u for u, _ in Pc.most_common(basis_cap)]; S = [v for v, _ in Sc.most_common(basis_cap)]
    fwd, info = hankel_forward_states(ds.g, P, S)
    F = np.array([fwd(h) for h in hists])                            # canonical predictive state per position
    fwd_w, _ = hankel_forward_states(ds.g, P, S, w=np.array([Pc[u] for u in P], float),
                                     nu=np.array([Sc[v] for v in S], float))
    Fw = np.array([fwd_w(h) for h in hists])                         # frequency-weighted basis

    # raw baseline basis = one-hot of the hand sufficient statistic (counter / stack)
    labs = sorted(set(labels)); lid = {k: i for i, k in enumerate(labs)}
    Braw = np.eye(len(labs))[[lid[l] for l in labels]]

    print(f"[{name}] Hankel rank = {info['rank']:3d}   KL(bayes||model) = {np.mean(kls):.4f}   "
          f"CCA(hidden, spectral) = {cca(H, F):.3f}")
    print(f"        CKA vs: spectral(joint) = {cka(H, F):.3f}   spectral(freq-weighted) = {cka(H, Fw):.3f}"
          f"   raw one-hot = {cka(H, Braw):.3f}")

    # geometry: top-2 CCA canonical variates (whitened, paired) of spectral state vs transformer —
    # if the hidden holds the predictive state the two panels coincide.
    def basis(A, tol=1e-9):
        Ac = A - A.mean(0); U, s, _ = np.linalg.svd(Ac, full_matrices=False); return U[:, s > tol * s[0]]
    Bf, Bh = basis(F), basis(H)
    Uu, _, Vt = np.linalg.svd(Bf.T @ Bh, full_matrices=False)
    Fp = Bf @ Uu[:, :2]; Hp = Bh @ Vt[:2].T
    lab_arr = np.array([lid[l] for l in labels])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    for a, (C, ttl) in zip(ax, [(Fp, "spectral predictive state"), (Hp, "transformer hidden (CCA-aligned)")]):
        sc = a.scatter(C[:, 0], C[:, 1], c=lab_arr, cmap=cmap, s=14, alpha=0.6)
        a.set_aspect("equal"); plot_utils.FIG.clean(a); a.set_title(ttl)
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02); cb.set_label(label_name)
    fig.suptitle(f"{name}: transformer = spectral predictive state up to a linear map "
                 f"(rank {info['rank']}, CCA {cca(H, F):.2f})", y=1.0)
    plot_utils.save_figure(fig, plot_utils.figure_path(config, "spectral_predictive_state"))
    return dict(rank=info["rank"], cca=cca(H, F), cka_spec=cka(H, F), cka_w=cka(H, Fw),
                cka_raw=cka(H, Braw), kl=np.mean(kls))


# %% [markdown]
# ## Dyck-2 (depth 3): predictive state = the bracket stack (+ ascending/descending phase)
# %%
def dyck_stack(pre, k=2):
    st = []
    for t in pre:
        st.append(t) if t < k else st.pop()
    return len(st)
r_dyck = analyze(["runner=pcfg_dyck"], "pcfg_dyck", dyck_stack, "stack depth")

# %% [markdown]
# ## a^n b^n: predictive state = the unmatched-a counter (rank grows with the depth bound)
# %%
def anbn_count(pre):
    return sum(t == 0 for t in pre) - sum(t == 1 for t in pre)
r_anbn = analyze(["runner=pcfg_anbn", "grammar_kwargs.depth=5", "max_len=12",
                  "training_kwargs.train_steps=6000"], "pcfg_anbn_d5", anbn_count, "unmatched a")

# %% [markdown]
# ## a^n T b^n: the trigger makes the task easy; same counter predictive state
# %%
r_antbn = analyze(["runner=pcfg_antbn", "grammar_kwargs.depth=6", "max_len=14",
                   "training_kwargs.train_steps=6000"], "pcfg_antbn_d6", anbn_count, "unmatched a")

# %% [markdown]
# ## Summary
# %%
print(f"{'grammar':16s} {'rank':>5s} {'KL':>8s} {'CCA':>6s} {'CKA_joint':>10s} {'CKA_freqw':>10s} {'CKA_raw':>8s}")
for nm, r in [("dyck", r_dyck), ("anbn", r_anbn), ("antbn", r_antbn)]:
    print(f"{nm:16s} {r['rank']:5d} {r['kl']:8.4f} {r['cca']:6.3f} {r['cka_spec']:10.3f} "
          f"{r['cka_w']:10.3f} {r['cka_raw']:8.3f}")


# %% [markdown]
# ## What differs: same info (CCA=1), different variance geometry
# CCA=1 but CKA is low because the transformer and the spectral SVD spread their variance over the
# shared predictive-state subspace differently. Left/middle: the states in each representation's own
# top-2 PCA (not aligned). Right: the variance spectrum — the spectral basis is balanced across all
# predictive states, the transformer concentrates variance in a few directions (a compressed,
# magnitude-dominated code). A rotation cannot fix a variance rescaling, so the rotational match is
# near zero even though the information is identical.
# %%
from scipy.linalg import orthogonal_procrustes

def geometry_diagnostic(overrides, name, label_fn, label_name, cmap="viridis"):
    model, ds, ev, hist, config = train_inline(
        list(overrides) + [f"name={name}", "wandb.enabled=false", "eval_kwargs.output_hidden_states=true"])
    seqs = [ds._sample_sentence()[0] for _ in range(250)]
    ids = np.full((len(seqs), ds.seq_len), ds.pad_token); ids[:, 0] = ds.start_token
    for b, s in enumerate(seqs):
        ids[b, 1:1 + len(s)] = s; ids[b, 1 + len(s)] = ds.eos_token
    ids = torch.tensor(ids, dtype=torch.long, device=ds.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    H, hists, labels = [], [], []
    for b, s in enumerate(seqs):
        for t in range(len(s) + 1):
            H.append(out["hidden_states"][-1][b, t].cpu().numpy()); hists.append(tuple(s[:t]))
            labels.append(label_fn(s[:t]))
    H = np.array(H); labels = np.array(labels)
    P = [u for u, _ in Counter(hists).most_common(120)]
    S = [v for v, _ in Counter(tuple(s[t:]) for s in seqs for t in range(len(s) + 1)).most_common(120)]
    fwd, info = hankel_forward_states(ds.g, P, S); F = np.array([fwd(h) for h in hists]); r = info["rank"]

    def unit(X): Xc = X - X.mean(0); return Xc / np.linalg.norm(Xc)
    def top(X, k): return (X - X.mean(0)) @ np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:k].T
    Hr, Fr = unit(top(H, r)), unit(top(F, r))
    R, _ = orthogonal_procrustes(Hr, Fr)
    r2_rot = 1 - np.linalg.norm(Hr @ R - Fr) ** 2 / np.linalg.norm(Fr) ** 2
    svF = np.linalg.svd(F - F.mean(0), compute_uv=False)[:r]; svF /= svF.sum()
    svH = np.linalg.svd(H - H.mean(0), compute_uv=False)[:r]; svH /= svH.sum()
    print(f"[{name}] rank={r}  CCA={cca(H, F):.3f}  CKA={cka(H, F):.3f}  rotation-align R2={r2_rot:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    for a, (C, ttl) in zip(ax[:2], [(top(F, 2), "spectral (own PCA)"), (top(H, 2), "transformer (own PCA)")]):
        sc = a.scatter(C[:, 0], C[:, 1], c=labels, cmap=cmap, s=16, alpha=0.6)
        a.set_aspect("equal"); plot_utils.FIG.clean(a); a.set_title(ttl)
    ax[2].plot(range(1, r + 1), svF, "-o", ms=4, label="spectral")
    ax[2].plot(range(1, r + 1), svH, "-s", ms=4, label="transformer")
    ax[2].set_xlabel("component"); ax[2].set_ylabel("variance fraction")
    ax[2].set_title("variance spectrum"); ax[2].legend(frameon=False); plot_utils.FIG.style_axes(ax[2])
    fig.colorbar(sc, ax=ax[:2], fraction=0.02, pad=0.02, label=label_name)
    fig.suptitle(f"{name}: same info (CCA={cca(H, F):.2f}) but different variance geometry "
                 f"(CKA={cka(H, F):.2f}, rotation-R2={r2_rot:.2f})", y=1.02)
    plot_utils.save_figure(fig, plot_utils.figure_path(config, "spectral_vs_transformer_geometry"))

geometry_diagnostic(["runner=pcfg_antbn", "grammar_kwargs.depth=6", "max_len=14",
                     "training_kwargs.train_steps=6000"], "pcfg_antbn_d6", anbn_count, "unmatched a")
