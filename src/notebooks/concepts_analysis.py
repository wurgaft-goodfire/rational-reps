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
# # What is a concept? Instructive figures on the ring
#
# The static symmetric case: an environment is a conditional over discrete alternatives, and
# every notion of "concept" is a reading of its centered log $L_c \approx E\,D$. One space, one
# invariant:
# $$ \langle e(i),\, d(x) \rangle = L_c[i, x] = \log p(x \mid i)\ \text{(centered)}, $$
# with concepts at positions $e(i)$ (rows of $E$), outcomes at positions $d(x)$ (columns of
# $D$), and the axes of the space indexed by the rows of $D$. In the symmetric case
# $E \approx D^\top$: concept points and outcome prototypes coincide on one circle.
#
# Environment: a 60-node ring with the **heat** kernel, so each concept's output is a wrapped
# Gaussian bump (log-kernel parabolic in distance; the diffusion kernel would instead give
# peaked Laplace bumps — exponential in distance, triangular log, odd harmonics only).

# %%
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(sys.prefix).parent            # <repo>/.venv -> <repo>
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.data import graph_conditional
from src.code.geometry import factor_logcond
from src.code import plot_utils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGDIR = Path("outputs/graph-cooccurrence/concepts_math/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

N, GAMMA = 60, 8.0
L = graph_conditional({"structure": "ring", "n": N, "gamma": GAMMA, "cooccurrence": "heat"})
Lc, E, D, S, nu = factor_logcond(L)
theta = 2 * np.pi * np.arange(N) / N
hue = plt.cm.twilight(theta / (2 * np.pi))                          # house cyclic colormap
P = np.exp(L); P /= P.sum(1, keepdims=True)

def decode(code):
    """Softmax readout of a code point: the output distribution it encodes. (..., k) -> (..., N)."""
    z = code @ D
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)

KDIMS = 4
KSTAR = 0                                                          # highlighted row (dimension)
XSTAR = int(np.argmax(np.minimum(D[0], D[1])))                     # highlighted column: a concept at a
                                                                   # generic angle (both coords large)

# %% [markdown]
# ## Figure 1 — the factorization, and the two readings of the decoder
#
# The manuscript schematic, $L_c \approx E \times D$ (top $k{=}4$ dimensions), with the
# concepts/dimensions distinction overlaid on the decoder: the grey band is one **row** — a
# conceptual dimension, an axis of the space with its expression profile over outcomes; the
# black band is one **column** — the decoder vector $d(x)$ of a single outcome, a point in the
# space (and, in the symmetric case, a concept: $e(x) = d(x)$). The companion panels unpack
# each reading, plus the spectrum that prices the dimensions.

# %%
fig, _ = plot_utils.plot_factorization(L, k=KDIMS, item_label="items  $i$",
                                        highlight_row=KSTAR, highlight_col=XSTAR, caption=None)
plot_utils.save_figure(fig, FIGDIR / "fig1a_factorization.png")

lim = np.percentile(np.abs(D[:KDIMS]), 98)

def strip(ax, M, edge):
    ax.imshow(M, cmap=plot_utils.FIG.diverging_cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(edge); sp.set_linewidth(2.2)

fig = plt.figure(figsize=(11.5, 3.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.3)

# the red row, unrolled: a conceptual dimension = a profile over outcomes
gsA = gs[0, 0].subgridspec(2, 1, height_ratios=[0.13, 1], hspace=0.15)
axs = fig.add_subplot(gsA[0]); strip(axs, D[KSTAR][None], "0.45")
axs.set_title("row of D: conceptual dimension", fontsize=12)
axc = fig.add_subplot(gsA[1])
for k in range(KDIMS):
    if k != KSTAR:
        axc.plot(D[k], color="0.85", lw=1.1)
axc.plot(D[KSTAR], color="0.3", lw=2.4)
axc.set_xlim(-0.5, N - 0.5)
axc.set_xlabel("outcomes x")
plot_utils.FIG.style_axes(axc)

# the blue column, plotted: an outcome's position = a point in the space
gsB = gs[0, 1].subgridspec(1, 2, width_ratios=[0.09, 1], wspace=0.12)
axv = fig.add_subplot(gsB[0]); strip(axv, D[:KDIMS, XSTAR][:, None], "black")
axv.set_xlabel(r"$d(x)$", fontsize=13)
axp = fig.add_subplot(gsB[1])
r = np.abs(D[:2]).max() * 1.35
# the top two rows of D, traced out as the axes of the conceptual (coefficient) space
axp.annotate("", (r, 0), (-r, 0), arrowprops=dict(arrowstyle="->", color="0.45", lw=1.6))
axp.annotate("", (0, r), (0, -r), arrowprops=dict(arrowstyle="->", color="0.7", lw=1.6))
axp.scatter(D[0], D[1], c=hue, s=16, zorder=3)
px, py = D[:2, XSTAR]
axp.scatter(px, py, s=190, facecolor="none", edgecolor="black", lw=2.4, zorder=4)
# read the concept's coordinates off the axes
axp.plot([px, px], [0, py], ls="--", color="0.15", lw=1.1, zorder=2)
axp.plot([0, px], [py, py], ls="--", color="0.15", lw=1.1, zorder=2)
axp.scatter([px, 0], [0, py], marker="|", s=0)
axp.set_xlim(-r * 1.15, r * 1.15); axp.set_ylim(-r * 1.15, r * 1.15)
axp.set_aspect("equal")
axp.set_title("column of D: concept", fontsize=12)
axp.set_xlabel("coefficient space", fontsize=11, color="0.5")
plot_utils.FIG.clean(axp)
plot_utils.save_figure(fig, FIGDIR / "fig1b_dual_reading.png")

# %% [markdown]
# ## Figure 3 — moving through the hull: what the outputs do
#
# Two ways to leave a vertex, and what each does to the decoded/predicted output:
#
# - **Pull between two concepts (the chord).** Position $\lambda$ along the chord is the
#   posterior-weighted point for belief $(1{-}\lambda, \lambda)$ over $\{A, B\}$; the Bayes
#   output is the *mixture* $(1{-}\lambda)p_A + \lambda p_B$ — two peaks trading mass. Being
#   torn between two concepts is not being an intermediate concept.
# - **Broaden one concept (the radius).** Scaling a code toward the center, $\tau\, e(C)$ with
#   $\tau: 1 \to 0$, decodes to the tempered output $p_C^{\tau}$ — the same concept at falling
#   confidence, sharp to flat. Radius is confidence; angle is content.

# %%
fig = plot_utils.plot_uncertainty(E)
plot_utils.save_figure(fig, FIGDIR / "fig3_uncertainty.png")

# %% [markdown]
# ## Figure 4 — concept vs prototype: an identity that asymmetry breaks
#
# A node plays two roles in the conditional. Its **row** is the node as a source — what it
# emits — giving the concept point $e(i)$: where belief must *settle* when the source is $i$.
# Its **column** is the node as evidence — what *seeing* that token implies about the source —
# giving the prototype $d(i)$: the direction one observation *pushes* the belief. On the ring,
# homogeneity makes the two coincide exactly; at the chain's boundary they split (the end token
# is more diagnostic than the end source is distinctive, so its prototype is ~30% longer and
# rotated). In practice the split bends inference: from a middle source, the evidence path and
# the belief path run together; from an end source, evidence pushes along the prototypes while
# the belief settles at the concept — the two trajectories visibly disagree.

# %%
Lch = graph_conditional({"structure": "chain", "n": N, "gamma": GAMMA, "cooccurrence": "heat"})
Lch_c, Ech, Dch, Sch, _ = factor_logcond(Lch)
mid, end = N // 2, 0
c_e, c_d = "0.2", plot_utils.FIG.colors["orange"]

# gauge-fix the per-mode reflections so rows and columns live in one coordinate frame
sgn = np.sign((Ech[:, :2] * Dch[:2].T).sum(0))
E2, D2 = Ech[:, :2], (Dch[:2].T * sgn)
Pch = np.exp(Lch); Pch /= Pch.sum(1, keepdims=True)

def paths(src, T=60, seed=1):
    """One sequence of iid tokens from a source: the running-evidence trajectory (mean of
    prototype vectors so far) and the belief trajectory (posterior-weighted concept point)."""
    rng = np.random.default_rng(seed)
    xs = rng.choice(N, size=T, p=Pch[src])
    ev = np.cumsum(D2[xs], axis=0) / np.arange(1, T + 1)[:, None]
    ll = np.cumsum(Lch[:, xs], axis=1)                             # (N, T)
    w = np.exp(ll - ll.max(0)); w /= w.sum(0)
    return ev, w.T @ E2

fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))
ax[0].scatter(*E2.T, c=c_e, s=16, zorder=3, label="concepts  e(i)")
ax[0].scatter(*D2.T, facecolor="none", edgecolor=c_d, s=26, lw=1.4, zorder=3,
              label="prototypes  d(i)")
for i in range(N):
    ax[0].plot([E2[i, 0], D2[i, 0]], [E2[i, 1], D2[i, 1]], color="0.8", lw=0.7, zorder=1)
ax[0].annotate("end", D2[end], D2[end] + np.array([0.55, -0.1]), fontsize=11, color="0.4")
ax[0].annotate("middle", D2[mid], D2[mid] + np.array([0, 0.6]), ha="center", fontsize=11,
               color="0.4")
ax[0].legend(fontsize=11, loc="center")
ax[0].set_title("chain: the two roles, one space")
ax[0].set_aspect("equal"); plot_utils.FIG.clean(ax[0])

for a, src, name in ((ax[1], mid, "middle source"), (ax[2], end, "end source")):
    a.scatter(*E2.T, c="0.88", s=10, zorder=1)
    a.scatter(*D2.T, facecolor="none", edgecolor="0.85", s=16, lw=1, zorder=1)
    ev, po = paths(src)
    a.plot(*ev.T, color=c_d, lw=1.8, zorder=3, label="running evidence (mean of d's)")
    a.plot(*po.T, color=c_e, lw=1.8, zorder=3, label="belief (posterior-weighted e)")
    a.scatter(*D2[src], marker="o", s=110, facecolor="none", edgecolor=c_d, lw=2.2, zorder=4)
    a.scatter(*E2[src], marker="*", s=260, facecolor="gold", edgecolor="k", lw=0.8, zorder=5)
    a.set_title(f"inference from the {name}")
    a.set_aspect("equal"); plot_utils.FIG.clean(a)
ax[1].legend(fontsize=10.5, loc="center")
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig4_concept_vs_prototype.png")

# %% [markdown]
# ## Figure 5 — a property is a direction, not a point
#
# The word-analogy world, derived rather than assumed: eight words = three binary factors
# (royal, female, adult) with unequal saliences; outcomes are the factor assertions. The
# factorization is exact (rank 3) and its dimensions align with the factors. Marked on the
# decomposition itself: a **concept is a row of E** (queen: its three coordinates) and the
# **decoder vector for an outcome is a column of D** — the point the readout dots the code
# against for that outcome's logit. No single column is "female": the vector that operationally
# means the property is the **difference of two decoder columns**,
# $u = d(\mathrm{fem}_+) - d(\mathrm{fem}_-)$, whose projection $\langle e_i, u\rangle$ is the
# log-odds of the answer (differences are forced — a softmax leaves columns defined only up to
# per-concept shifts, so only contrasts between outcomes are honest vectors). In the 3-D
# geometry the same $u$ carries man to woman, king to queen, boy to girl, prince to princess:
# a property is a direction (a contrast between decoder vectors), never a location.

# %%
an_items = ["man", "woman", "king", "queen", "boy", "girl", "prince", "princess"]
an_F = {"man": (-1, -1, 1), "woman": (-1, 1, 1), "king": (1, -1, 1), "queen": (1, 1, 1),
        "boy": (-1, -1, -1), "girl": (-1, 1, -1), "prince": (1, -1, -1), "princess": (1, 1, -1)}
an_props = [r"royal$_+$", r"royal$_-$", r"female$_+$", r"female$_-$", r"adult$_+$", r"adult$_-$"]
sal = np.array([1.2, 1.0, 0.8])                                    # factor saliences (break degeneracy)
Xa = np.array([an_F[i] for i in an_items], float)
La = 0.5 * np.concatenate([np.stack([sal[j] * Xa[:, j], -sal[j] * Xa[:, j]], 1)
                           for j in range(3)], 1)
Lac, Ea, Da, Sa, _ = factor_logcond(La)
orange = plot_utils.FIG.colors["orange"]
Qi = an_items.index("queen")

fig, axes = plot_utils.plot_factorization(La, k=3, item_label="", caption=None)
axes["L"].set_yticks(range(8)); axes["L"].set_yticklabels(an_items, fontsize=11)
axes["L"].set_xticks(range(6)); axes["L"].set_xticklabels(an_props, fontsize=10, rotation=45)
axes["D"].set_xticks(range(6)); axes["D"].set_xticklabels(an_props, fontsize=10, rotation=45)
axes["E"].add_patch(mpatches.Rectangle((-0.5, Qi - 0.5), 3, 1, fill=False, ec="black", lw=2.2,
                                       clip_on=False))
axes["E"].annotate("a concept:\n'queen' = a row of E", (-0.6, Qi), (-5.8, Qi + 3.2), fontsize=12,
                   ha="center", annotation_clip=False,
                   arrowprops=dict(arrowstyle="->", color="0.2"))
axes["D"].add_patch(mpatches.Rectangle((1.5, -0.5), 1, 3, fill=False, ec="0.45", lw=2.2,
                                       clip_on=False))
axes["D"].add_patch(mpatches.Rectangle((2.5, -0.5), 1, 3, fill=False, ec="0.45", lw=2.2,
                                       clip_on=False))
axes["D"].annotate("a property:\n'female' = d(fem$_+$) $-$ d(fem$_-$)\n"
                   "(difference of decoder columns)", (3.6, 1), (6.3, 2.6), fontsize=12,
                   ha="left", color="0.35", annotation_clip=False,
                   arrowprops=dict(arrowstyle="->", color="0.45"))
plot_utils.save_figure(fig, FIGDIR / "fig5a_decomposition.png")

# %%
E3 = Ea[:, :3]
sgn = np.sign([np.corrcoef(E3[:, k], Xa[:, k])[0, 1] for k in range(3)])
E3 = E3 * sgn
d_fem = (Da[:3, 2] - Da[:3, 3]) * sgn
fig = plt.figure(figsize=(9, 6.2))
ax = fig.add_subplot(111, projection="3d")
for i in range(8):
    for j in range(i + 1, 8):
        if sum(x != y for x, y in zip(an_F[an_items[i]], an_F[an_items[j]])) == 1:
            fem = an_F[an_items[i]][1] != an_F[an_items[j]][1]
            ax.plot(*zip(E3[i], E3[j]), color=orange if fem else "0.8",
                    lw=2.2 if fem else 1.0, zorder=1)
ax.scatter(*E3.T, c="0.2", s=55, zorder=5)
for i, name in enumerate(an_items):
    ax.text(*(E3[i] * 1.17), name, fontsize=13, ha="center", zorder=10)
start = np.array([1.35, -1.5, 0.1])
v = d_fem / np.linalg.norm(d_fem) * 1.2
ax.quiver(*start, *v, color=orange, lw=2.8, arrow_length_ratio=0.14)
ax.text(*(start + v * 0.2 + np.array([0, 0, -0.42])), "decoder 'female'", fontsize=13, color=orange)
ax.set_axis_off()
ax.set_title("the 'female' dimension carries every concept to its counterpart",
             fontsize=15, y=1.02)
ax.view_init(elev=16, azim=-62)
plot_utils.save_figure(fig, FIGDIR / "fig5b_property_direction_3d.png")

# %% [markdown]
# ## Figure 6 — the decoder as geometry: binary, then a continuous line
#
# Drawn constructions (the SVD check comes later): the one law
# $\log p(x \mid i) = \langle e_i, d_x\rangle$, read as geometry. **Left**, the standard
# binary picture: concepts fall in two groups, the property's decoder contrast
# $u = d_{\mathrm{yes}} - d_{\mathrm{no}}$ is one direction, and a concept's log-odds is its
# projection onto that direction. The prototypes are drawn astride a shared component (a gauge
# the softmax ignores — only the contrast survives the readout), so the chord between them *is*
# the log-odds axis: the line inference travels in fig 7. **Right**, the simplest continuous extension: temperature
# $\theta$ on a line, adjective outputs with Gaussian tuning
# $\log p(x \mid \theta) = -(\theta - \mu_x)^2/2 + \mathrm{const}(\theta)$. This factorizes
# *exactly* in 2-D — no curved embedding needed: $e_\theta = (\theta,\, 1)$ stays a straight
# line (offset from the origin; the constant second coordinate is forced), and the curvature
# the bumps require moves into the **decoder**, $d_x = (\mu_x,\, -\mu_x^2/2)$ — a parabola
# whose drop is each adjective's norm penalty. The logit
# $\langle e_\theta, d_x\rangle = \theta\mu_x - \mu_x^2/2$ is then squared distance on the
# line up to a per-$\theta$ constant: each adjective is a tuning bump centered at its
# $\mu_x$, and a warm day scores positively on $d_{\mathrm{hot}}$, negatively on
# $d_{\mathrm{cold}}$. (Horseshoe/arc embeddings appear only when tuning saturates away from
# Gaussian — that is what a chain's SVD produces; we verify both regimes by SVD later.)
# **Far right**, the circular latent: time of week, the seven days as decoders, von Mises
# tuning $\log p(x \mid \varphi) = \kappa\cos(\varphi - \varphi_x) + \mathrm{const}$. Here the
# symmetry does what the line could not: the normalizer is constant, so no offset coordinate
# and no curvature asymmetry — embeddings and decoders sit on concentric circles ($e = d$ up
# to scale, the self-legible symmetric world), and every day's logit is again angle read
# through a projection.

# %%
fig, ax = plt.subplots(1, 3, figsize=(19.5, 6))

def _code_axes(a, rx, y0, y1):
    """Light coordinate axes of the code space."""
    a.annotate("", xy=(rx, 0), xytext=(-rx, 0), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.8"))
    a.annotate("", xy=(0, y1), xytext=(0, y0), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.8"))
    a.text(rx, -0.1, "dim 1", fontsize=10, color="0.55", ha="right", va="top")
    a.text(0.05, y1, "dim 2", fontsize=10, color="0.55", ha="left", va="top")

c_blue = plt.get_cmap(plot_utils.FIG.diverging_cmap)(0.08)
c_red = plt.get_cmap(plot_utils.FIG.diverging_cmap)(0.92)

# left — binary property: the yes/no prototypes astride their contrast, one embedding
a = ax[0]
_code_axes(a, 1.55, -1.0, 1.15)
u = np.array([np.cos(np.deg2rad(15)), np.sin(np.deg2rad(15))])
mAB = 0.55 * np.array([-u[1], u[0]])                 # shared component — a gauge the softmax ignores
d_yes, d_no = mAB + 0.9 * u, mAB - 0.9 * u           # the two prototypes, astride the contrast
a.plot(*zip(mAB + 1.35 * u, mAB - 1.35 * u), ls=":", color="0.8", lw=1.2, zorder=0)   # the chord
a.text(*(mAB + 1.42 * u), r"$u = d_{\mathrm{yes}} - d_{\mathrm{no}}$", fontsize=11.5,
       color="0.55", ha="left", va="center")
ei = 1.1 * np.array([np.cos(np.deg2rad(62)), np.sin(np.deg2rad(62))])
a.annotate("", xy=ei, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2.6, color=c_blue))
a.annotate(r"$e_i$", ei + np.array([0.02, 0.14]), ha="center", va="bottom",
           fontsize=13, color=c_blue)
for d, lab, off in [(d_yes, r"$d_{\mathrm{yes}}$", (0.2, -0.18)),
                    (d_no, r"$d_{\mathrm{no}}$", (-0.2, -0.18))]:
    a.annotate("", xy=d, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.6, color="0.15"))
    a.annotate(f"{lab}\n{ei @ d:+.2f}", d, d + np.array(off), fontsize=11,
               ha="center", va="top", color="0.3")
for d, r in [(d_yes, 0.3), (d_no, 0.45)]:            # angle arcs, embedding to each prototype
    dang = np.degrees(np.arctan2(d[1], d[0]))
    a.add_patch(mpatches.Arc((0, 0), 2 * r, 2 * r, theta1=min(62, dang),
                             theta2=max(62, dang), color="0.78", lw=1.0))
a.set_title("binary property: log-odds = projection", fontsize=13, pad=14)
a.set_xlim(-1.75, 1.75); a.set_ylim(-1.15, 1.7)
a.set_aspect("equal"); plot_utils.FIG.clean(a)

# right — temperature on a line: e straight (offset), curvature lives in the decoder parabola
a = ax[1]
_code_axes(a, 1.7, -0.75, 1.45)
adjs = ["freezing", "cold", "moderate", "warm", "hot"]
mus = np.linspace(-1, 1, 5)                          # adjective centers on the latent
Dv = np.stack([mus, (np.mean(mus**2) - mus**2) / 2], 1) * 1.2    # centered decoder parabola
a.plot([-1.45, 1.45], [1, 1], ls=":", color="0.8", lw=1.2, zorder=0)   # the embedding line
a.text(1.52, 1.0, r"$e_\theta = (\theta,\ 1)$", fontsize=12, color="0.55", ha="left", va="center")
tstar = 0.55                                         # one embedding: a warm day
e = np.array([tstar, 1.0])
a.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2.6, color=c_red))
a.annotate(r"$e_\theta$  (warm day)", e + np.array([0.02, 0.14]), ha="center", va="bottom",
           fontsize=12.5, color=c_red)
e_ang = np.degrees(np.arctan2(e[1], e[0]))
lab_off = {0: (0, -0.14, "center", "top"), 1: (-0.12, 0.06, "right", "center"),
           2: (-0.04, 0.24, "right", "bottom"), 3: (0.12, 0.06, "left", "center"),
           4: (0, -0.14, "center", "top")}
d_angs = np.degrees(np.arctan2(Dv[:, 1], Dv[:, 0]))
d_angs += 360 * ((d_angs - e_ang) < -180) - 360 * ((d_angs - e_ang) > 180)   # shortest sweep
rank = np.argsort(np.argsort(np.abs(d_angs - e_ang)))
for j, (dv, lab) in enumerate(zip(Dv, adjs)):
    a.annotate("", xy=dv, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.6, color="0.15"))
    dx, dy, ha, va = lab_off[j]
    a.annotate(f"{lab}\n{e @ dv:+.2f}", dv, dv + np.array([dx, dy]), fontsize=11,
               ha=ha, va=va, color="0.3")
    r = 0.2 + 0.08 * rank[j]                         # angle arcs nested by angular distance
    a.add_patch(mpatches.Arc((0, 0), 2 * r, 2 * r, theta1=min(e_ang, d_angs[j]),
                             theta2=max(e_ang, d_angs[j]), color="0.78", lw=1.0))
a.set_title("continuous latent: embeddings stay a line, the decoder curves", fontsize=13, pad=18)
a.set_xlim(-1.9, 1.9); a.set_ylim(-0.95, 1.85)
a.set_aspect("equal"); plot_utils.FIG.clean(a)

# far right — circular latent (weektime): the symmetric case, e = d up to scale
a = ax[2]
_code_axes(a, 1.95, -1.5, 1.68)
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
day_angs = 90 - 360 * np.arange(7) / 7                            # Monday at the top, clockwise
Dw = 0.85 * np.stack([np.cos(np.deg2rad(day_angs)), np.sin(np.deg2rad(day_angs))], 1)
tt = np.linspace(0, 2 * np.pi, 200)
a.plot(np.cos(tt), np.sin(tt), ls=":", color="0.8", lw=1.2, zorder=0)   # the embedding circle
phi = 90 - 360 * 2.6 / 7                                          # one embedding: Wednesday night
e = np.array([np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))])
a.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2.6, color=c_red))
a.annotate(r"$e_\varphi$  (Wed night)", e * 1.85, ha="center", va="center",
           fontsize=12.5, color=c_red)
w_angs = day_angs + 360 * ((day_angs - phi) < -180) - 360 * ((day_angs - phi) > 180)
rank = np.argsort(np.argsort(np.abs(w_angs - phi)))
for j, (dv, lab) in enumerate(zip(Dw, days)):
    a.annotate("", xy=dv, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.6, color="0.15"))
    a.annotate(f"{lab}\n{e @ dv:+.2f}", dv, dv / 0.85 * 1.3, fontsize=11,
               ha="center", va="center", color="0.3")
    r = 0.16 + 0.06 * rank[j]                        # angle arcs nested by angular distance
    a.add_patch(mpatches.Arc((0, 0), 2 * r, 2 * r, theta1=min(phi, w_angs[j]),
                             theta2=max(phi, w_angs[j]), color="0.78", lw=1.0))
a.set_title("circular latent: the symmetric case, $e = d$ up to scale", fontsize=13, pad=18)
a.set_xlim(-2.25, 2.25); a.set_ylim(-1.85, 2.1)
a.set_aspect("equal"); plot_utils.FIG.clean(a)
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig6_continuous_decoder.png")

# %% [markdown]
# ## Figure 6b — the same three worlds, with the readout's decision regions
#
# Figure 6 redrawn with the **argmax of the output-token logits** shaded in. Logits are linear
# in the code with no bias, so every token's region is a **cone from the origin**: angle decides,
# radius only sharpens the softmax. **Binary**: the two half-planes of $d_{\mathrm{yes}}$ /
# $d_{\mathrm{no}}$, split by the line $\perp u$; the prototypes sit one per side and the chord
# between them crosses the boundary exactly at its midpoint — the shared component moves both
# decoders but never the boundary. **Line**: five cones of unequal width; the
# embedding line $e_\theta = (\theta, 1)$ crosses their boundaries at $\theta = (\mu_i+\mu_j)/2$
# — the decision thresholds on the latent. **Circle**: seven equal sectors, the Voronoi of the
# day decoders (the warm-day / Wed-night embeddings of fig 6 sit inside the region they decode to).

# %%
from matplotlib.colors import to_rgb

def argmax_regions(Dcols, r_x, r_y, res=500):
    """Argmax-outcome index over the code plane. Dcols (2, n) -> (res, res) image + extent."""
    gx, gy = np.linspace(*r_x, res), np.linspace(*r_y, res)
    C = np.stack(np.meshgrid(gx, gy), -1)                          # (res, res, 2), [i,j] = (x_j, y_i)
    return (C @ Dcols).argmax(-1), (*r_x, *r_y)

def region_bg(ax, win, ext, rgb, fade=0.5):
    """Decision regions as a white-blended background image (rgb: (n, 3-or-4) region colors)."""
    ax.imshow(1 - fade * (1 - np.asarray(rgb)[win][..., :3]), origin="lower", extent=ext, zorder=0)

fig, ax = plt.subplots(1, 3, figsize=(19.5, 6))

# left — binary property: two half-planes
a = ax[0]
yes_no = [to_rgb(plot_utils.FIG.colors["blue"]), to_rgb(plot_utils.FIG.colors["orange"])]
region_bg(a, *argmax_regions(np.stack([d_yes, d_no], 1), (-1.75, 1.75), (-1.15, 1.7)),
          yes_no, fade=0.16)
_code_axes(a, 1.55, -1.0, 1.15)
perp = np.array([-u[1], u[0]])                                     # the decision boundary, ⟂ u
a.plot(*zip(1.55 * perp, -1.1 * perp), ls="--", color="0.4", lw=1.3, zorder=2)
a.plot(*zip(mAB + 1.35 * u, mAB - 1.35 * u), ls=":", color="0.7", lw=1.2, zorder=1)   # the chord
a.text(*(mAB + 1.42 * u), r"$u$", fontsize=11.5, color="0.55", ha="left", va="center")
a.scatter(*mAB, s=30, facecolor="white", edgecolor="0.35", lw=1.2, zorder=4)   # midpoint, on the boundary
a.annotate("", xy=ei, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2.6, color=c_blue))
a.annotate(r"$e_i$", ei + np.array([0.02, 0.14]), ha="center", va="bottom",
           fontsize=13, color=c_blue)
for d, lab, off in [(d_yes, r"$d_{\mathrm{yes}}$", (0.2, -0.16)),
                    (d_no, r"$d_{\mathrm{no}}$", (-0.2, -0.16))]:
    a.annotate("", xy=d, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.6, color="0.15"))
    a.annotate(lab, d + np.array(off), fontsize=11.5, ha="center", va="top", color="0.3")
a.text(1.28, -0.95, "argmax: yes", fontsize=11.5, color="0.35", ha="center")
a.text(-1.28, -0.95, "argmax: no", fontsize=11.5, color="0.35", ha="center")
a.set_title("binary: two half-planes", fontsize=13, pad=14)
a.set_xlim(-1.75, 1.75); a.set_ylim(-1.15, 1.7)
a.set_aspect("equal"); plot_utils.FIG.clean(a)

# middle — temperature line: five cones, thresholds where the embedding line crosses them
a = ax[1]
cols5 = plt.get_cmap(plot_utils.FIG.diverging_cmap)(np.linspace(0.06, 0.94, 5))
region_bg(a, *argmax_regions(Dv.T, (-1.9, 1.9), (-0.95, 1.85), res=600), cols5, fade=0.55)
_code_axes(a, 1.7, -0.75, 1.45)
for dv in Dv:
    a.annotate("", xy=dv, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.6, color="0.15"))
for mu, lab in zip(mus, adjs):
    a.text(1.55 * mu, 1.58, lab, fontsize=11, ha="center", color="0.25")
a.plot([-1.45, 1.45], [1, 1], ls=":", color="0.3", lw=1.4, zorder=3)
a.text(1.52, 1.0, r"$e_\theta = (\theta,\,1)$", fontsize=12, color="0.4", ha="left", va="center")
e = np.array([tstar, 1.0])                                         # the warm day of fig 6
a.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2.6, color=c_red))
thr = (mus[:-1] + mus[1:]) / 2
a.scatter(thr, np.ones(4), marker="|", s=110, color="0.1", zorder=5)
for t_ in thr:
    a.text(t_, 0.86, f"{t_:+.2f}", fontsize=9.5, ha="center", va="top", color="0.35")
a.set_title("line: five cones — thresholds on the latent", fontsize=13, pad=14)
a.set_xlim(-1.9, 1.9); a.set_ylim(-0.95, 1.85)
a.set_aspect("equal"); plot_utils.FIG.clean(a)

# right — weektime circle: seven equal sectors
a = ax[2]
hue7 = plt.cm.twilight(np.arange(7) / 7)
region_bg(a, *argmax_regions(Dw.T, (-2.25, 2.25), (-1.85, 2.1)), hue7, fade=0.45)
_code_axes(a, 1.95, -1.5, 1.68)
tt = np.linspace(0, 2 * np.pi, 200)
a.plot(np.cos(tt), np.sin(tt), ls=":", color="0.55", lw=1.2, zorder=1)
for dv, lab in zip(Dw, days):
    a.annotate("", xy=dv, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.6, color="0.15"))
    a.annotate(lab, dv, dv / 0.85 * 1.3, fontsize=11, ha="center", va="center", color="0.25")
phi_wed = 90 - 360 * 2.3 / 7                                       # Wed night, inside Wed's sector
e = np.array([np.cos(np.deg2rad(phi_wed)), np.sin(np.deg2rad(phi_wed))])
a.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2.6, color=c_red))
a.annotate(r"$e_\varphi$  (Wed night)", e * 1.78, ha="center", va="center",
           fontsize=12, color=c_red)
a.set_title("circle: seven equal sectors", fontsize=13, pad=14)
a.set_xlim(-2.25, 2.25); a.set_ylim(-1.85, 2.1)
a.set_aspect("equal"); plot_utils.FIG.clean(a)
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig6b_decision_regions.png")

# %% [markdown]
# ## Figure 7 — inference: given examples of x, the history walks the code space
#
# The inference setting of `notes/inference_derivation.md`: $z$ is hidden, the model sees
# examples $x_{1:t} \sim p(x \mid z)$, and the target posterior factors through a belief state,
# $\log \tilde p(z \mid x_{1:t}) = E(z)^\top \phi_z(x_{1:t})$. With conditionally independent
# examples the state is **additive** — $\phi_z(x_{1:t}) = \phi_z(\varnothing) + \sum_i D_z(x_i)$
# — so a history is a *trajectory*: each example adds its evidence vector, and the decision
# regions of fig 6b (now over $z$, via the rows of $E$) give the running argmax. The three
# worlds of fig 6:
#
# - **Binary $z$.** Two concepts $e_A$, $e_B$ — in this symmetric world they *are* the
#   prototypes $d_{\mathrm{yes}}$, $d_{\mathrm{no}}$ of fig 6 — and noisy yes/no evidence. The *belief point* —
#   the posterior-weighted concept, fig 3's chord — travels the segment between them: each
#   observation nudges it toward one end, position $= \sigma(\Lambda_t)$ of the cumulative
#   log-odds. The $\hat z$ boundary ($\perp e_A - e_B$, through the origin) cuts the chord at
#   its midpoint; single paths can wander across early before the drift wins.
# - **Continuous $z$ (the temperature line).** The latent-inference embedding is *not* the
#   next-token decoder: $\log p(x_{1:t} \mid \theta) = \theta \sum_i \mu(x_i) - t\,A(\theta)$
#   needs the count, so $D_z(x) = (1, \mu_x)$ and $E_z(\theta) = (-A(\theta), \theta)$ — the same
#   belief state, a different target-adapted embedding (the derivation's central point). In the
#   $(t, \sum\mu)$ plane the day-types' regions are again cones from the origin, and the
#   trajectory is a staircase whose *slope* is the running mean evidence.
# - **Day of the week.** Noisy day mentions around a true day; the circle's homogeneity makes
#   $A(z)$ constant, so the raw additive sum $\phi_z = \sum_i d(x_i)$ is the whole state: it
#   spirals out into the true day's sector — angle converges (content), radius grows
#   (confidence). Binary shows the *normalized* belief view (bounded on the chord); the week
#   shows the *raw evidence* view (growing) — the two faces of the same walk (cf. fig 4).

# %%
T = 40
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))

# left — binary z: the belief point travels the chord between the two concepts
a = ax[0]
eA, eB = d_yes, d_no                                 # symmetric world: concepts = the prototypes
xr, yr = (-1.65, 1.65), (-0.55, 1.45)
region_bg(a, *argmax_regions(np.stack([eA, eB], 1), xr, yr), yes_no, fade=0.16)
a.plot(*zip(2.6 * mAB, -1.0 * mAB), ls="--", color="0.4", lw=1.3, zorder=2)   # boundary ⟂ e_A - e_B
for ez, lab, off in [(eA, r"$e_A$", (0.14, 0.06)), (eB, r"$e_B$", (-0.14, 0.06))]:
    a.annotate("", xy=ez, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2.0, color="0.15"))
    a.annotate(lab, ez, ez + np.array(off), fontsize=13, ha="center", va="bottom", color="0.2")
a.plot(*zip(eA, eB), color="0.75", lw=1.0, zorder=1)
p1 = 0.65                                                          # p(x says "yes" | z = A)
llr = np.log(p1 / (1 - p1))
lam = np.concatenate([[0.0], np.cumsum(np.where(np.random.default_rng(0).random(T) < p1,
                                                llr, -llr))])
b = 1 / (1 + np.exp(-lam))                                         # posterior p(A | x_{1:t})
pts = eB + np.outer(b, eA - eB)
a.plot(*pts.T, color="0.6", lw=0.8, zorder=3)
a.scatter(*pts.T, c=range(T + 1), cmap="viridis", s=30, zorder=4)
a.annotate("$t=0$", mAB, mAB + np.array([-0.1, -0.3]), fontsize=11, color="0.3",
           arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
a.text(1.25, -0.42, r"$\hat z = A$", fontsize=12.5, color="0.35", ha="center")
a.text(-1.25, -0.42, r"$\hat z = B$", fontsize=12.5, color="0.35", ha="center")
a.set_title("binary: the belief travels the chord", fontsize=13)
a.set_xlim(*xr); a.set_ylim(*yr)
a.set_aspect("equal"); plot_utils.FIG.clean(a)

# middle — day-type from adjective mentions: cones in the (count, evidence) plane
a = ax[1]
A_th = np.log(np.exp(mus[:, None] * mus[None, :] - mus[None, :] ** 2 / 2).sum(1))   # A(theta_k)
Slim = 0.45 * T
tg, Sg = np.meshgrid(np.linspace(0, T, 400), np.linspace(-Slim, Slim, 400))
win = (mus * Sg[..., None] - A_th * tg[..., None]).argmax(-1)     # argmax_k theta_k S - t A(theta_k)
a.imshow(1 - 0.55 * (1 - cols5[win][..., :3]), origin="lower",
         extent=(0, T, -Slim, Slim), aspect="auto", zorder=0)
theta_star = 0.55                                                  # the warm day of fig 6
p_star = np.exp(theta_star * mus - mus ** 2 / 2); p_star /= p_star.sum()
xs = np.random.default_rng(26).choice(5, T, p=p_star)
S = np.concatenate([[0.0], np.cumsum(mus[xs])])
a.plot([0, T], [0, T * (p_star @ mus)], ls="--", color="0.15", lw=1.1, zorder=2)
a.plot(range(T + 1), S, color="0.35", lw=1.0, zorder=3)
a.scatter(range(T + 1), S, c=range(T + 1), cmap="viridis", s=22, zorder=4)
bounds = np.diff(A_th) / np.diff(mus)                              # region slopes S/t
mid = np.concatenate([[bounds[0] - 0.14], (bounds[:-1] + bounds[1:]) / 2, [bounds[-1] + 0.14]])
for slope, lab in zip(mid, adjs):
    a.text(0.985 * T, np.clip(slope * T, -0.93 * Slim, 0.93 * Slim), lab,
           fontsize=10.5, ha="right", va="center", color="0.25",
           bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))
a.set(xlim=(0, T), ylim=(-Slim, Slim), xlabel="examples  $t$",
      ylabel=r"evidence  $S_t = \sum_i \mu(x_i)$")
a.set_title("continuous: cones in (count, evidence)", fontsize=13)
plot_utils.FIG.style_axes(a)

# right — day of the week: additive evidence spirals into the true day's sector
a = ax[2]
zday = 2                                                           # true day: Wednesday
ang = np.deg2rad(day_angs)
pw = np.exp(1.5 * np.cos(ang - ang[zday])); pw /= pw.sum()         # von Mises day mentions, kappa = 1.5
xs = np.random.default_rng(2).choice(7, T, p=pw)
traj = np.vstack([np.zeros(2), np.cumsum(Dw[xs], axis=0)])
rr = np.abs(traj).max() * 1.35
region_bg(a, *argmax_regions(Dw.T, (-rr, rr), (-rr, rr)), hue7, fade=0.45)
for dv, lab in zip(Dw, days):
    a.text(*(dv / 0.85 * 0.88 * rr), lab, fontsize=11, ha="center", va="center",
           color="0.1" if lab == days[zday] else "0.35")
udir = Dw[zday] / np.linalg.norm(Dw[zday])
a.plot([0, udir[0] * 0.78 * rr], [0, udir[1] * 0.78 * rr], ls="--", color="0.15", lw=1.2, zorder=2)
a.plot(*traj.T, color="0.35", lw=1.0, zorder=3)
a.scatter(*traj.T, c=range(T + 1), cmap="viridis", s=26, zorder=4)
a.scatter(*(udir * 0.78 * rr), marker="*", s=260, facecolor="gold", edgecolor="k", lw=0.8, zorder=5)
a.annotate("$t=0$", traj[0], traj[0] + rr * np.array([-0.22, -0.06]), fontsize=11, color="0.3")
a.set_title("week: evidence spirals into the true day", fontsize=13)
a.set_aspect("equal"); plot_utils.FIG.clean(a)
fig.tight_layout()
plot_utils.save_figure(fig, FIGDIR / "fig7_inference_trajectories.png")

# %% [markdown]
# ## Summary
#
# | notion | formal object | in the figures |
# |---|---|---|
# | conceptual dimension | row of $D$: an axis + its expression over outcomes | fig 1, red |
# | outcome position / prototype | column of $D$: the point $d(x)$ | fig 1, blue |
# | concept | row of $E$; $= d$ in the symmetric case | fig 1b, circle |
# | belief state | a point of the convex hull of the prototypes | fig 3, interior |
# | uncertainty between concepts | chord interior; output = mixture (two peaks) | fig 3, ● |
# | confidence in a concept | radius; output = tempered (sharp → flat) | fig 3, ■ |
# | prototype = concept | exact under symmetry; splits at boundaries | fig 4 |
# | property (binary factor) | difference of decoder columns $d(\mathrm{fem}_+) - d(\mathrm{fem}_-)$: one direction carries every concept | fig 5 |
# | decision region | cone of an argmax readout: angle decides, radius never | fig 6b |
# | inference from examples | additive evidence $\phi_z = \phi_z(\varnothing) + \sum_i D_z(x_i)$; posterior read out by $E(z)^\top$ | fig 7 |
# | confidence under inference | trajectory radius / drift margin, growing with $t$ | fig 7 |

# %%
