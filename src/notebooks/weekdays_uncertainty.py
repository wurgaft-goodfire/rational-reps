import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Uncertainty over weekdays: where an uncertain answer sits on the ring

    *Spokes, Not a Ring* (Gupta et al. 2026) read "Usually on Monday, sometimes on Tuesday, her flight
    leaves." at the final period and found uncertain prompts on one day's spoke, nearer the centre. We
    append an answer slot, "This week, her flight leaves on", and read there too: same sentence, but now
    the model is predicting a day. Three families: their frequency adverbs, stated percentages, and sets
    of one to seven equally likely days. Llama-3.1-8B, eight continuations each.

    **Prediction** (tenets 1 and 4): at the answer slot the representation is a linear image of the centred
    log-probabilities over the days. So more days in play means nearer the centre; two days put the point
    between them and further in the further apart they are, part of the move leaving the ring's plane.
    We predict each prompt's position from the model's own output distribution, with a map fitted on the
    set family only and tested on the two-day families, against the same map on probabilities (centroids
    averaged by belief) and against Spokes' picture (the most likely day's spoke, at its best radius).
    Coordinates are the ring's harmonics, from single-day prompts ("Her flight leaves on Monday.").
    """)
    return


@app.cell
def _(mo):
    import os, sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt

    ROOT = mo.notebook_dir().parent.parent
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.weekdays import (DAYS, all_prompts, day_span, fit_predict, harmonic_basis, off_plane,
                                   pair_position, r2, read_prompts, ring_coords, spoke_predict)
    from src.code.plot_utils import save_figure

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures" / "weekdays"; FIGURES.mkdir(parents=True, exist_ok=True)
    SEP_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}          # categorical slots 1-3
    SIZE_COLORS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # ordinal blue
    INK, MUTED = "#2b2b2a", "#8a8984"
    return (CACHE, DAYS, FIGURES, INK, MUTED, Path, SEP_COLORS, SIZE_COLORS, all_prompts, day_span,
            fit_predict, harmonic_basis, np, off_plane, pair_position, pd, plt, r2, read_prompts,
            ring_coords, save_figure, spoke_predict, torch)


@app.cell
def _(all_prompts, np, pd):
    P = all_prompts()
    df = pd.DataFrame([{k: v for k, v in p.items() if k not in ("weights",)} for p in P])
    df["A"] = df.A.fillna(0).astype(int); df["B"] = df.B.fillna(0).astype(int)
    df["sep"] = df.sep.fillna(0).astype(int); df["size"] = df["size"].fillna(0).astype(int)
    df["start"] = df.start.fillna(0).astype(int)
    W = np.stack([p["weights"] for p in P])
    for _f in ("adverb", "percent", "set"):
        _p = P[df.index[df.family == _f][17]]; print(f"{_f:8s}", _p["pre"] + _p["post"])
    print(len(P), "prompts")
    return P, W, df


@app.cell
def _(CACHE, P, np, read_prompts, torch):
    READ_DIR = CACHE / "weekdays" / "llama-3.1-8b"; READ_DIR.mkdir(parents=True, exist_ok=True)
    KEYS = ("mark", "answer", "q_mark", "q_answer")
    if not all((READ_DIR / f"{k}.npy").exists() for k in KEYS):          # the expensive step, cached
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        _model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16,
                                                      device_map="cuda").eval()
        for _k, _v in read_prompts(_model, _tok, P, batch_size=32).items():
            np.save(READ_DIR / f"{_k}.npy", _v)
        del _model; torch.cuda.empty_cache()
    reads = {k: np.load(READ_DIR / f"{k}.npy", mmap_mode="r") for k in KEYS}   # (layers, N, d) / (N, 8)
    Q = reads["q_answer"]
    return Q, reads


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the model predicts

    At the period almost no next-token mass is on a day; at the answer slot most of it is. The base
    model barely follows the stated weights: "Always on Monday, never on Tuesday" splits much like
    "sometimes / sometimes", and every percent level gives the first-named day about half. So we
    predict from the measured distribution, not the wording. Sets spread mass over their days.
    """)
    return


@app.cell
def _(Q, W, df, np, pd, reads):
    _i = np.arange(len(df))
    _b = pd.DataFrame({"family": df.family, "level": df.level, "stated on A": W[_i, df.A],
                       "P(A)": Q[_i, df.A], "P(B)": Q[_i, df.B], "P(any day)": Q[:, :7].sum(1),
                       "P(day) at period": reads["q_mark"][:, :7].sum(1)})
    _b[df.family != "set"].groupby(["family", "level"]).mean().round(2)
    return


@app.cell
def _(Q, W, df, pd):
    pd.DataFrame({"size": df["size"], "P(in set)": (Q[:, :7] * (W > 0)).sum(1), "P(any day)": Q[:, :7].sum(1)}
                 )[df.family == "set"].groupby("size").mean().round(2)
    return


@app.cell
def _(Q, day_span, df, fit_predict, harmonic_basis, np, off_plane, pair_position, r2, reads, ring_coords,
      spoke_predict):
    REF = ((df.family == "set") & (df["size"] == 1)).values            # "Her flight leaves on Monday."
    TWO = df.family.isin(["adverb", "percent"]).values
    SETS = (df.family == "set").values


    SPOKES_REF = ((df.family == "adverb") & (df.level == 0)).values    # Spokes' ring: "Always on A, never on B"


    def ring(site, L, ref=REF, key="start"):
        """Ring coordinates and hypothesis predictions at one site and layer; the ring through the
        centroids of the ``ref`` prompts grouped by ``key`` (the stated day)."""
        H = np.asarray(reads[site][L], np.float32)
        C = np.stack([H[ref & (df[key].values == k)].mean(0) for k in range(7)])
        mu, B = harmonic_basis(C)
        Z = day_span(H, mu, B); basis = np.linalg.qr(B)[0]
        out = {"Z": Z, "C": day_span(C, mu, B)}
        out["angle"], out["radius"] = ring_coords(H, mu, B)
        for k in ("log", "prob"):                                       # fitted on sets, applied to all
            out[k] = fit_predict(Q[SETS], Z[SETS], Q, k)
            out["angle_" + k], out["radius_" + k] = ring_coords(mu + out[k] @ basis.T, mu, B)
        out["spoke"] = spoke_predict(Z, Q, out["C"], np.zeros(6))
        out["off"] = off_plane(Z)
        return out


    def scores(L):
        o = ring("answer", L)
        return {k: r2(o["Z"][TWO], o[k][TWO]) for k in ("log", "prob", "spoke")}


    L_MAIN = 28
    at = {s: ring(s, L_MAIN) for s in ("mark", "answer")}
    for _s in at:
        at[_s]["w"] = pair_position(at[_s]["angle"], df.A.values, df.B.values, df.sep.values)
    at["answer"]["w_log"] = pair_position(at["answer"]["angle_log"], df.A.values, df.B.values, df.sep.values)
    spk = {s: ring(s, L_MAIN, SPOKES_REF, "A") for s in ("mark", "answer")}
    for _s in spk:
        spk[_s]["w"] = pair_position(spk[_s]["angle"], df.A.values, df.B.values, np.maximum(df.sep.values, 1))
    print({k: round(v, 2) for k, v in scores(L_MAIN).items()})
    return L_MAIN, SETS, TWO, at, scores, spk


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two named days: on the first spoke at the period, between and inward at the answer slot

    Each two-day prompt drawn in the ring plane, rotated so the first-named day A is at the right and B
    lies counter-clockwise, `sep` days on. The ring is built as Spokes built it, from the "Always on A,
    never on B" prompts (grey circle, radius 1). At the period the points stay near A's spoke (Spokes'
    finding; the table below gives their radius dip at the uncertain levels). At the answer slot they sit
    between the two days, nearer the centre the further apart the days are.
    """)
    return


@app.cell
def _(DAYS, FIGURES, INK, MUTED, SEP_COLORS, TWO, df, np, plt, save_figure, spk):
    _fig, _axes = plt.subplots(2, 3, figsize=(9.6, 6.6), sharex=True, sharey=True)
    _t = np.linspace(0, 2 * np.pi, 200)
    for _r, (_site, _name) in enumerate((("mark", "period (Spokes' read)"), ("answer", "answer slot"))):
        for _c, _sep in enumerate((1, 2, 3)):
            _ax = _axes[_r, _c]; _m = TWO & (df.sep.values == _sep)
            _rel = spk[_site]["w"][_m] * _sep * 2 * np.pi / 7; _rad = spk[_site]["radius"][_m]
            _ax.plot(np.cos(_t), np.sin(_t), color=MUTED, lw=1)
            for _k, _lab in ((0, "A"), (_sep, "B")):
                _a = 2 * np.pi * _k / 7
                _ax.plot([0, 1.25 * np.cos(_a)], [0, 1.25 * np.sin(_a)], color=MUTED, lw=1, ls=":")
                _ax.annotate(_lab, (1.38 * np.cos(_a), 1.38 * np.sin(_a)), ha="center", va="center", color=INK)
            _ax.scatter(_rad * np.cos(_rel), _rad * np.sin(_rel), s=8, alpha=.35, color=SEP_COLORS[_sep], lw=0)
            _mx, _my = np.median(_rad * np.cos(_rel)), np.median(_rad * np.sin(_rel))
            _ax.scatter([_mx], [_my], s=70, color=SEP_COLORS[_sep], edgecolor="white", lw=2, zorder=3)
            _ax.scatter([0], [0], s=12, color=INK, marker="+")
            _ax.set_aspect("equal"); _ax.set_xlim(-2.2, 2.6); _ax.set_ylim(-2.2, 2.6)
            _ax.set_xticks([]); _ax.set_yticks([])
            for _sp in _ax.spines.values(): _sp.set_visible(False)
            if _r == 0: _ax.set_title(f"{_sep} day{'s' if _sep > 1 else ''} apart", color=INK)
            if _c == 0: _ax.set_ylabel(_name, color=INK)
    _fig.suptitle(f"Llama-3.1-8B layer 28, two-day prompts (large dot: median)", color=INK)
    _fig.tight_layout()
    save_figure(_fig, str(FIGURES / "two_day_rings.png"))
    return


@app.cell
def _(TWO, df, pd, spk):
    _adv = TWO & (df.family == "adverb").values
    pd.concat({_name: pd.DataFrame({"sep": df.sep[_adv], "level": df.level[_adv], "radius": spk[_s]["radius"][_adv],
                                    "angle (0 = A, 1 = B)": spk[_s]["w"][_adv]}).groupby(["level", "sep"]).median().unstack()
               for _s, _name in (("mark", "period"), ("answer", "answer slot"))}).round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How far in: observed against predicted from the model's own distribution

    Median radius (1 = single-day ring). Left, two-day prompts by separation; right, sets by size. The
    log-probability map and the probability map are fitted on the sets only, so the left panel is out of
    sample. At the period the radius does not shrink (for sets it grows: more days named).
    """)
    return


@app.cell
def _(FIGURES, INK, MUTED, SETS, TWO, at, df, np, plt, save_figure):
    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    _a = at["answer"]
    for _ax, _mask, _key, _xs, _xl in ((_axes[0], TWO, "sep", (1, 2, 3), "days between the two named days"),
                                        (_axes[1], SETS, "size", range(1, 8), "days in the set")):
        _med = lambda v: [np.median(v[_mask & (df[_key].values == x)]) for x in _xs]
        _ax.plot(_xs, _med(_a["radius"]), "o-", color=INK, lw=2, ms=8, label="observed, answer slot")
        _ax.plot(_xs, _med(_a["radius_log"]), "s--", color="#2a78d6", lw=2, ms=8, label="predicted, log-probability map")
        _ax.plot(_xs, _med(_a["radius_prob"]), "^:", color="#eb6834", lw=2, ms=8, label="predicted, probability map")
        _ax.plot(_xs, _med(at["mark"]["radius"]), "o-", color=MUTED, lw=1.5, ms=6, label="observed, period")
        _ax.axhline(1, color=MUTED, lw=.8, ls=":"); _ax.axhline(0, color=MUTED, lw=.8)
        _ax.set_xlabel(_xl, color=INK); _ax.set_xticks(list(_xs)); _ax.set_ylim(0, 2.1)
        for _sp in ("top", "right"): _ax.spines[_sp].set_visible(False)
    _axes[0].set_ylabel("radius (1 = single-day ring)", color=INK)
    _h, _l = _axes[0].get_legend_handles_labels()
    _fig.legend(_h, _l, frameon=False, fontsize=8, loc="lower center", ncol=4)
    _fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(_fig, str(FIGURES / "radius_by_spread.png"))
    return


@app.cell
def _(SEP_COLORS, TWO, at, df, np, pd):
    pd.DataFrame({"sep": df.sep[TWO], "observed": at["answer"]["off"][TWO], "period": at["mark"]["off"][TWO]}
                 ).groupby("sep").median().round(2).rename_axis("share of the move outside the ring plane, by sep")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which layers carry the prediction

    Out-of-sample R² for the two-day prompts' positions in the six-dimensional day span (fitted on sets).
    Spokes' picture gets its best radius per prompt, so it is an upper bound for that account.
    """)
    return


@app.cell
def _(FIGURES, INK, MUTED, np, pd, plt, save_figure, scores):
    LAYERS = list(range(2, 33, 2))
    by_layer = pd.DataFrame([scores(L) for L in LAYERS], index=LAYERS)
    _fig, _ax = plt.subplots(figsize=(6.4, 3.6))
    for _k, _c, _lab in (("log", "#2a78d6", "log-probability map"), ("prob", "#eb6834", "probability map"),
                         ("spoke", "#1baf7a", "spoke of the likeliest day, best radius")):
        _ax.plot(LAYERS, by_layer[_k].clip(lower=-0.5), "o-", color=_c, lw=2, ms=5, label=_lab)
    _ax.axhline(0, color=MUTED, lw=.8)
    _ax.set_xlabel("layer", color=INK); _ax.set_ylabel("held-out R² (two-day prompts)", color=INK)
    _ax.set_ylim(-0.5, 1); _ax.legend(frameon=False, fontsize=8, loc="upper left")
    for _sp in ("top", "right"): _ax.spines[_sp].set_visible(False)
    _fig.tight_layout()
    save_figure(_fig, str(FIGURES / "r2_by_layer.png"))
    return (by_layer,)


@app.cell
def _(by_layer):
    by_layer.round(2).T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The angle against the measured split (the weak part)

    Position between the spokes (0 on A, 1 on B) against the model's share of the pair's mass on B. Where
    the model leans clearly to one day the point sits on that day's spoke; across the middle the angle
    tracks the split only loosely, while the log map predicts it should follow it.
    """)
    return


@app.cell
def _(FIGURES, INK, MUTED, Q, SEP_COLORS, TWO, at, df, np, plt, save_figure):
    _i = np.arange(len(df))
    _qb = Q[_i, df.B] / (Q[_i, df.A] + Q[_i, df.B])
    _bins = np.linspace(0, 1, 6); _mid = (_bins[1:] + _bins[:-1]) / 2
    _fig, _axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True)
    for _ax, _sep in zip(_axes, (1, 2, 3)):
        _m = TWO & (df.sep.values == _sep)
        _ax.scatter(_qb[_m], at["answer"]["w"][_m], s=6, alpha=.25, color=SEP_COLORS[_sep], lw=0)
        _k = np.digitize(_qb[_m], _bins[1:-1])
        for _key, _c, _ls, _lab in (("w", INK, "-", "observed"), ("w_log", "#2a78d6", "--", "predicted, log map")):
            _ax.plot(_mid, [np.median(at["answer"][_key][_m][_k == j]) if (_k == j).any() else np.nan for j in range(5)],
                     "o" + _ls, color=_c, lw=2, ms=6, label=_lab)
        _ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=":")
        _ax.set_title(f"{_sep} day{'s' if _sep > 1 else ''} apart", color=INK); _ax.set_xlabel("P(B) / (P(A) + P(B))", color=INK)
        _ax.set_ylim(-0.6, 1.6)
        for _sp in ("top", "right"): _ax.spines[_sp].set_visible(False)
    _axes[0].set_ylabel("angle: 0 = A, 1 = B", color=INK)
    _axes[0].legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    save_figure(_fig, str(FIGURES / "angle_vs_split.png"))
    return


if __name__ == "__main__":
    app.run()
