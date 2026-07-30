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
# ## Spectral emergence of a latent category: when does "gender" start to exist?
#
# The gender/noun-class DGP: every noun carries a hidden binary class, invisible in the token
# itself, revealed only through *marked* agreement paradigms — pairs of alternating forms
# (la/el, -a/-o, ...) seen with probability `p` per token and biased by `delta` toward the
# class's preferred form. In the noun x context count matrix the class is a planted **rank-1
# spike** in the correspondence-analysis residuals, with population singular value
# `delta * sqrt(p)` per token against a Marchenko-Pastur noise bulk. So the abstraction has a
# *critical corpus size* — the BBP transition `N* = sqrt(V*C) / (p * delta^2)`:
#
# - **below N***: the class direction is statistically invisible — at any rank, for any method
#   reading this matrix — and generalization to held-out nouns is chance;
# - **above N***: the direction snaps into alignment, and with it comes the abstraction's payoff:
#   a *novel* noun is classified from a **single** marked token ("la blicket" -> "blicket blanca"),
#   i.e. per-noun sample complexity for agreement drops from many attestations to one.
#
# "Usefulness of the compression" is literally the singular value: frequency of the revealing
# contexts (`sqrt(p)`) x their predictive impact (`delta`). With Zipfian noun frequencies the
# global snap smears into a per-noun *staircase* — the age-of-acquisition figure at the end.

# %%
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(sys.prefix).parent            # <repo>/.venv -> <repo>
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.emergence import (acquisition_N, bbp_threshold, ca_residuals, class_context_joint,
                                identification_curve, mp_edge, payoff_curve, sample_counts,
                                threshold_N)
from src.code import plot_utils
from src.code.plot_utils import FIG, SERIES
import matplotlib.pyplot as plt

FIGDIR = Path(os.environ.get("CACHE_DIR", "outputs")) / "class-emergence" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

V, N_NEUTRAL, N_MARKED = 400, 10, 8       # 16 marked + 10 neutral contexts
P, DELTA = 0.1, 0.8
C = 2 * N_MARKED + N_NEUTRAL
NSTAR = bbp_threshold(V, C, P, DELTA)
print(f"V={V} nouns, C={C} contexts, p={P}, delta={DELTA}  ->  predicted N* = {NSTAR:.0f} tokens")

# %% [markdown]
# ## 1. The spectrum snap: a singular value crosses the noise floor
#
# Sorted singular values of the residual matrix at four corpus sizes (uniform noun frequencies —
# the case the BBP theory describes exactly). Below `N*` the class spike is swallowed by the
# Marchenko-Pastur bulk (dashed edge); above it, one singular value detaches. The moment of
# detachment is the moment the category becomes *detectable at all* — no rank choice can recover
# it earlier, because the empirical singular vector is orthogonal to the truth below threshold.

# %%
Qu, cls_u, _ = class_context_joint(V, N_NEUTRAL, p=P, delta=DELTA, n_marked=N_MARKED,
                                   zipf_a=0.0, seed=0)
rng = np.random.default_rng(0)
Ns_show = [200, 1_000, 5_000, 50_000]
fig, ax = plt.subplots(1, len(Ns_show), figsize=(14, 3.4), sharey=True)
edge = mp_edge((V, C))
for a, N in zip(ax, Ns_show):
    S = np.linalg.svd(ca_residuals(sample_counts(Qu, N, rng)), compute_uv=False)
    a.bar(np.arange(len(S)), S / edge, color=[SERIES[3]] + [FIG.colors['gray']] * (len(S) - 1))
    a.axhline(1.0, ls='--', color='0.3', lw=1)
    a.set_title(f"N = {N:,}" + ("  (< N*)" if N < NSTAR else "  (> N*)"), fontsize=13)
    a.set_xlabel("singular value rank")
    FIG.style_axes(a)
ax[0].set_ylabel("singular value / MP edge")
fig.suptitle(f"the class spike detaches from the noise bulk near N* $\\approx$ {NSTAR:.0f}", y=1.06)
fig.tight_layout()
plot_utils.save_figure(fig, str(FIGDIR / "spectrum_snap.png"))

# %% [markdown]
# ## 2. The BBP transition: identification of the class direction
#
# Overlap between the empirical top singular vector and the planted class direction, as the same
# corpus grows (8 corpus seeds). Below threshold the overlap hovers near zero — the direction is
# not "weakly known", it is *unknown* — then rises sharply past `N*`. This is the spectral picture
# of sudden category emergence: not gradual refinement but a phase transition in what the data
# statistically supports.

# %%
Ns = np.unique(np.geomspace(50, 200_000, 16).astype(int))
curves = identification_curve(Qu, cls_u, Ns, n_seeds=8, seed=0)
n_meas = threshold_N(Ns, curves, level=0.5)
print(f"measured N* (median overlap crosses 0.5): {n_meas:.0f};  BBP prediction: {NSTAR:.0f}")

fig, ax = plt.subplots(figsize=(7, 4.2))
for c in curves:
    ax.plot(Ns, c, color=SERIES[0], alpha=0.25, lw=1)
ax.plot(Ns, np.median(curves, 0), color=SERIES[0], lw=2.5, label="identification (median)")
ax.axvline(NSTAR, ls='--', color='0.3', lw=1.2)
ax.text(NSTAR * 1.15, 0.05, "predicted $N^*$", fontsize=12, color='0.3')
ax.set(xscale='log', xlabel="corpus size N (tokens)", ylabel="overlap with class direction",
       ylim=(-0.02, 1.02))
ax.legend(frameon=False); FIG.style_axes(ax)
ax.set_title("the class direction snaps into alignment at $N^*$")
fig.tight_layout()
plot_utils.save_figure(fig, str(FIGDIR / "bbp_identification.png"))

# %% [markdown]
# ## 3. The payoff: one-shot generalization to novel nouns
#
# The reason to call the direction an *abstraction*: once it exists, a held-out noun is classified
# from a **single** marked token (which side of the learned context axis its one observed form
# falls on, within its paradigm). What must be learned is the *sign pattern across all 8
# paradigms* — a single pair's polarity is one bit and can be guessed lucky, which is why the DGP
# has many paradigms. Below `N*` the pattern is noise and accuracy hovers near chance; above `N*`
# it approaches the Bayes ceiling `(1+delta)/2` (set by the token's own informativeness).
# Identification and payoff transition *together*: detecting the direction and being able to use
# it are the same event.

# %%
acc = payoff_curve(Qu, cls_u, Ns, delta=DELTA, n_marked=N_MARKED, n_seeds=8, seed=0)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(Ns, np.median(curves, 0), color=SERIES[0], lw=2.5, label="identification (overlap)")
for a_ in acc:
    ax.plot(Ns, a_, color=SERIES[1], alpha=0.2, lw=1)
ax.plot(Ns, np.median(acc, 0), color=SERIES[1], lw=2.5, label="one-shot accuracy (novel nouns)")
ax.axhline(0.5, ls=':', color='0.5', lw=1)
ax.text(Ns[-1], 0.485, "chance", fontsize=11, color='0.5', ha='right', va='top')
ax.axhline((1 + DELTA) / 2, ls=':', color='0.5', lw=1)
ax.text(Ns[-1], (1 + DELTA) / 2 - 0.015, "Bayes ceiling", fontsize=11, color='0.5',
        ha='right', va='top')
ax.axvline(NSTAR, ls='--', color='0.3', lw=1.2)
ax.set(xscale='log', xlabel="corpus size N (tokens)", ylim=(-0.02, 1.05))
ax.legend(frameon=False, loc="center left"); FIG.style_axes(ax)
ax.set_title("crossing $N^*$ is what buys one-shot generalization")
fig.tight_layout()
plot_utils.save_figure(fig, str(FIGDIR / "one_shot_payoff.png"))

# %% [markdown]
# ## 4. The scaling law: $N^* \propto 1/(p\,\delta^2)$
#
# "Usefulness = frequency x impact", quantitatively. Sweeping the frequency of revealing contexts
# `p` (at fixed `delta`) and their strength `delta` (at fixed `p`), the measured emergence
# threshold should fall on the BBP line `N* = sqrt(V*C)/(p * delta^2)`: log-log slope **-1 in p**
# and **-2 in delta**. This is the calibration that later lets us *predict* the emergence corpus
# size for natural data (UD treebank gender) from its measured `(p, delta)` — zero-shot.

# %%
Ns_sw = np.unique(np.geomspace(30, 60_000, 14).astype(int))

def measured_nstar(p, delta, seed):
    Q, cls, _ = class_context_joint(V, N_NEUTRAL, p=p, delta=delta, n_marked=N_MARKED,
                                    zipf_a=0.0, seed=seed)
    return threshold_N(Ns_sw, identification_curve(Q, cls, Ns_sw, n_seeds=6, seed=seed))

ps = np.array([0.05, 0.1, 0.2, 0.4])
deltas = np.array([0.25, 0.35, 0.5, 0.7, 1.0])
nstar_p = np.array([measured_nstar(p, 0.8, seed=10 + i) for i, p in enumerate(ps)])
nstar_d = np.array([measured_nstar(0.2, d, seed=20 + i) for i, d in enumerate(deltas)])
slope_p = np.polyfit(np.log(ps), np.log(nstar_p), 1)[0]
slope_d = np.polyfit(np.log(deltas), np.log(nstar_d), 1)[0]
print(f"measured slopes: N* ~ p^{slope_p:.2f} (theory -1),  N* ~ delta^{slope_d:.2f} (theory -2)")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for a, x, y, sl, th, lab in [(ax[0], ps, nstar_p, slope_p, -1, "$p$ (marked-context frequency)"),
                             (ax[1], deltas, nstar_d, slope_d, -2, "$\\delta$ (agreement strength)")]:
    a.plot(x, y, 'o-', color=SERIES[2], lw=2, ms=7, label=f"measured (slope {sl:.2f})")
    theory = (bbp_threshold(V, C, x, 0.8) if th == -1 else bbp_threshold(V, C, 0.2, x))
    a.plot(x, theory, '--', color='0.3', lw=1.2, label=f"BBP: slope {th}")
    a.set(xscale='log', yscale='log', xlabel=lab, ylabel="emergence threshold $N^*$")
    a.legend(frameon=False); FIG.style_axes(a)
fig.suptitle("usefulness = frequency $\\times$ impact: $N^* = \\sqrt{VC}\\,/\\,(p\\,\\delta^2)$", y=1.04)
fig.tight_layout()
plot_utils.save_figure(fig, str(FIGDIR / "scaling_law.png"))

# %% [markdown]
# ## 5. Zipf and the age-of-acquisition hockey stick
#
# With Zipfian noun frequencies (the natural case) the global snap smears into a per-noun
# staircase, and this is the figure with the cogsci shape. A noun's **age of acquisition** — the
# corpus size from which it stays correctly classified — is bounded by two sloped guides:
#
# - **brute attestation** (upper): with no abstraction, classifying a noun needs ~`4/delta^2`
#   marked tokens of its own, so AoA ~ `4 / (f p delta^2)`;
# - **single attestation** (lower): once the class direction exists, *one* marked token places
#   the noun — the wait for that token, AoA ~ `1 / (f p)`.
#
# The abstraction converts the per-noun requirement from many attestations to one. For frequent
# nouns both waits are short (they're learned brute-force before the class even exists — the
# pre-abstraction regime); for the Zipf tail the wait *is* the cost, and the cloud hugging the
# lower line instead of the upper one is the compression paying out. The same per-noun signature
# — AoA collapsing off the brute line — is what to look for across checkpoints of a neural LM.

# %%
Qz, cls_z, fz = class_context_joint(V, N_NEUTRAL, p=P, delta=DELTA, n_marked=N_MARKED,
                                    zipf_a=1.2, seed=42)
Ns_aoa = np.unique(np.geomspace(100, 1_000_000, 24).astype(int))
aoa = acquisition_N(Qz, cls_z, Ns_aoa, seed=42)
never = ~np.isfinite(aoa)
print(f"acquired within 1M tokens: {(~never).sum()}/{V} nouns"
      + (f" (dropping {never.sum()} never-acquired)" if never.any() else ""))

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.scatter(fz[~never], aoa[~never], s=18, alpha=0.6, color=SERIES[0], edgecolors='none')
fgrid = np.geomspace(fz.min(), fz.max(), 50)
ax.plot(fgrid, 4.0 / (fgrid * P * DELTA ** 2), '--', color='0.3', lw=1.4,
        label="brute attestation: $4/(f\\,p\\,\\delta^2)$")
ax.plot(fgrid, 1.0 / (fgrid * P), '--', color=SERIES[1], lw=1.4,
        label="single attestation: $1/(f\\,p)$")
ax.set(xscale='log', yscale='log', xlabel="noun frequency $f$ (Zipf)",
       ylabel="age of acquisition (tokens)")
ax.legend(frameon=False, loc="lower left"); FIG.style_axes(ax)
ax.set_title("one marked token suffices once the class exists")
fig.tight_layout()
plot_utils.save_figure(fig, str(FIGDIR / "age_of_acquisition.png"))

# %% [markdown]
# ## What this sets up
#
# The synthetic calibration fixes everything needed for the confirmatory natural-data experiment:
# measure `(p, delta, V, C)` in a gendered UD treebank (Spanish-AnCora / French-GSD nouns with
# `Gender=` annotations, adjacent determiner/adjective forms as contexts), pre-register the
# predicted `N*` from the scaling law, then subsample tokens and watch for the snap — with the
# semantically arbitrary irregulars (mano, dia, problema) as the ecologically valid one-shot test.
# The same axes (identification + one-shot vs N) are then ready for a neural learner: does a
# trained LM's per-noun agreement accuracy show the same hockey stick, at the same N*?
