# %% [markdown]
# # Classic CF grammars: the network activations track the exact belief trajectory
#
# As in the HMM notebook (forward-algorithm belief vs transformer activations, compared by CCA), we
# read the geometry off the *belief trajectory*, not off `factor_logcond` of a conditional. For a
# genuine context-free language the exact Bayes belief is the posterior over the latent sufficient
# statistic — the count of unmatched a's (a^n b^n) or the stack of open bracket types (Dyck), both
# deterministic given the prefix here. We show (i) the transformer reaches the exact Bayes oracle,
# and (ii) its hidden state is the belief trajectory up to a linear map (CCA ~ 1), the same test that
# singled out the belief in the HMM.

# %%
import os, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(sys.prefix).parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.data import lpcfg_next_token, tree_adjacency, walk_cooccur, logcond_from_sim
from src.code.geometry import predicted_geometry
from src.code.eval import cka, cca, rsa
from src.code.train import train_inline
from src.code import plot_utils
import matplotlib.pyplot as plt


def load_and_probe(runner, name, key_fn):
    """Train/load, sample sequences, and return the per-position hidden states + the exact belief
    (a one-hot over the latent sufficient statistic ``key_fn(prefix)`) + the KL to the Bayes oracle."""
    model, ds, ev, hist, config = train_inline(
        [f'runner={runner}', f'name={name}', 'wandb.enabled=false',
         'eval_kwargs.output_hidden_states=true'])
    B = 300
    seqs = [ds._sample_sentence()[0] for _ in range(B)]
    ids = np.full((B, ds.seq_len), ds.pad_token); ids[:, 0] = ds.start_token
    for b, s in enumerate(seqs):
        ids[b, 1:1 + len(s)] = s; ids[b, 1 + len(s)] = ds.eos_token
    ids = torch.tensor(ids, dtype=torch.long, device=ds.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    pm = torch.softmax(out['logits'][:, :-1][..., ds.item_tokens], -1).cpu().numpy()
    kl = lambda p, q: float((p * (np.log(p + 1e-9) - np.log(q + 1e-9))).sum(-1))

    keys = sorted({key_fn(s[:t]) for s in seqs for t in range(len(s) + 1)})
    kid = {k: i for i, k in enumerate(keys)}
    H, B1hot, KEYS, SEQIDX, TT, kls, kl_marg_rows = [], [], [], [], [], [], []
    for b, s in enumerate(seqs):
        d, _ = lpcfg_next_token(np.array(s, dtype=int), ds.g)
        for t in range(len(s) + 1):
            H.append(out['hidden_states'][-1][b, t].cpu().numpy())
            oneh = np.zeros(len(keys)); oneh[kid[key_fn(s[:t])]] = 1.0
            B1hot.append(oneh); KEYS.append(key_fn(s[:t])); SEQIDX.append(b); TT.append(t)
            kls.append(kl(d[t], pm[b, t])); kl_marg_rows.append(d[t])
    H = np.array(H); B1hot = np.array(B1hot); marg = np.array(kl_marg_rows).mean(0)
    klm = np.mean([kl(r, marg) for r in kl_marg_rows])
    print(f'[{name}] KL(bayes||model) = {np.mean(kls):.4f}   (marginal baseline {klm:.3f})')
    print(f'[{name}] CCA(hidden, belief trajectory) = {cca(H, B1hot):.3f}   '
          f'(CKA {cka(H, B1hot):.3f}; rotation under-reports as in the HMM)')
    return dict(model=model, ds=ds, config=config, H=H, belief=B1hot, seqs=seqs,
                keys=keys, KEYS=KEYS, SEQIDX=np.array(SEQIDX), TT=np.array(TT))


def cca_variates(X, Y, k=2, tol=1e-9):
    """Top-k CCA canonical variates: paired coordinates ``(Xv, Yv)`` where column i of Xv and Yv are
    maximally correlated (correlation ``S[i]``). This gives the belief its geometry *from its
    correlation with the hidden* (a one-hot belief has none on its own), and puts both sides in the
    same, directly comparable frame — the literal "both under CCA"."""
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    Ux, Sx, _ = np.linalg.svd(Xc, full_matrices=False); Ux = Ux[:, Sx > tol * Sx[0]]
    Uy, Sy, _ = np.linalg.svd(Yc, full_matrices=False); Uy = Uy[:, Sy > tol * Sy[0]]
    P, S, Qt = np.linalg.svd(Ux.T @ Uy, full_matrices=False)
    return Ux @ P[:, :k], Uy @ Qt[:k].T, S                          # Xv (model), Yv (belief), corrs


def belief_vs_model(r, title, fname, color=None, cmap='viridis', legend=None,
                    color_desc='position', single_traj=False, nshow=80):
    """The HMM figure: the exact Bayes belief trajectory in its own (optimal) geometry, beside the
    model's, both under CCA. The two panels are the paired CCA canonical variates — belief (left) and
    model hidden (right) — so if the hidden holds the belief up to a linear map the panels coincide.
    ``single_traj`` plots only the longest sequence (clearest for a^n T b^n); else up to ``nshow``
    trajectories. Points colored by ``color`` (default position, or a phase-label array)."""
    H, Bel, SEQIDX, TT = r['H'], r['belief'], r['SEQIDX'], r['TT']
    c_all = TT if color is None else np.asarray(color)
    Xv, Yv, S = cca_variates(H, Bel, 2)                             # model / belief canonical variates
    if single_traj:
        sel = max(np.unique(SEQIDX), key=lambda b: (SEQIDX == b).sum())
        which = [sel]; sz, lw = 110, 1.3
    else:
        which = np.unique(SEQIDX)[:nshow]; sz, lw = 14, 0.6
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.6))
    for a, (coords, lab) in zip(ax, [(Yv, 'belief (optimal geometry)'),
                                     (Xv, 'model hidden (CCA-aligned)')]):
        for b in which:
            m = np.where(SEQIDX == b)[0]; o = m[np.argsort(TT[m])]
            a.plot(coords[o, 0], coords[o, 1], '-', color='0.8', lw=lw, zorder=0)
            a.scatter(coords[o, 0], coords[o, 1], c=c_all[o], cmap=cmap, s=sz, zorder=1,
                      vmin=c_all.min(), vmax=c_all.max(),
                      edgecolors='white' if single_traj else 'none', linewidths=0.8)
        a.set_aspect('equal'); plot_utils.FIG.clean(a); a.set_title(lab)
    if legend is not None:
        cm = plt.get_cmap(cmap); n = len(legend)
        handles = [plt.Line2D([0], [0], marker='o', ls='', ms=8,
                              color=cm(i / max(n - 1, 1)), label=lab) for i, lab in enumerate(legend)]
        ax[1].legend(handles=handles, frameon=False, fontsize=9, loc='center left',
                     bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f'{title}   (CCA = {cca(H, Bel):.2f}; color = {color_desc})', y=1.01)
    fig.tight_layout()
    plot_utils.save_figure(fig, plot_utils.figure_path(r['config'], fname))


def phase_colors(r, trigger=None):
    """Per-position phase label for a^n b^n / a^n T b^n: 0 = opening (a-run), 1 = trigger, 2 = closing
    (b-run). Read off the last terminal of each prefix."""
    lab = []
    for b in range(len(r['seqs'])):
        s = r['seqs'][b]
        for t in range(len(s) + 1):
            last = s[t - 1] if t > 0 else 0
            lab.append(1 if last == trigger else (2 if last == 1 else 0))
    return np.array(lab)


# %% [markdown]
# ## a^n b^n : the belief is the count of unmatched a's (colored by a-run vs b-run)
# %%
def anbn_key(prefix):                                               # exact belief = (phase, count)
    count = sum(1 for t in prefix if t == 0) - sum(1 for t in prefix if t == 1)
    return (1 if any(t == 1 for t in prefix) else 0, count)         # phase 1 once a b has appeared
r = load_and_probe('lpcfg_anbn', 'lpcfg_anbn', anbn_key)
belief_vs_model(r, 'a^n b^n: hidden = belief trajectory up to a linear map', "anbn_belief_cca",
                color=phase_colors(r, trigger=None), cmap='coolwarm', color_desc='phase',
                legend=['opening (a-run)', 'closing (b-run)'])

# %% [markdown]
# ## a^n T b^n : an explicit trigger terminal marks the phase boundary
# The counter belief is the same, but the trigger T is a single observable event at the turn. Shown as
# a single trajectory: the a-run climbs, T sits at the turn, the b-run descends.
# %%
def antbn_key(prefix):                                              # exact belief = (phase, count)
    count = sum(1 for t in prefix if t == 0) - sum(1 for t in prefix if t == 1)
    return (1 if any(t == 2 for t in prefix) else 0, count)         # phase 1 once T has appeared
r = load_and_probe('lpcfg_antbn', 'lpcfg_antbn', antbn_key)
belief_vs_model(r, 'a^n T b^n: one belief trajectory, hidden = belief up to a linear map',
                "antbn_belief_cca", color=phase_colors(r, trigger=2), cmap='viridis',
                color_desc='phase', single_traj=True,
                legend=['opening (a-run)', 'trigger (T)', 'closing (b-run)'])

# %% [markdown]
# ## Dyck-k : the belief is the parser stack
# %%
K = 2
def dyck_key(prefix):
    st = []
    for t in prefix:
        st.append(t) if t < K else st.pop()
    return tuple(st)
r = load_and_probe('lpcfg_dyck', 'lpcfg_dyck', dyck_key)
belief_vs_model(r, f'Dyck-{K}: hidden = stack belief trajectory up to a linear map', "dyck_belief_cca")
