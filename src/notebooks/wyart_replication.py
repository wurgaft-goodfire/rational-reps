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
    # Wyart-group replications: corpus statistics vs Gemma-2-2B

    Karkada et al. (symmetry: years, states, months), Nava & Wyart (WordNet trees) and Korchinski et al. (analogies),
    each on their corpus (Wikipedia 2023-11) and on Gutenberg and FineWeb-Edu as transfer. Every setting has a main
    panel, Gemma beside three codes for the same words: their word embedding (the top modes of their target matrix over
    the full vocabulary), the same construction under the repo's row-centred log-conditional, and our block code from
    the words' own co-occurrences. Their specific tests follow each panel. Gemma reads use their prompt templates at
    the final token; words without a template are read bare.
    """)
    return


@app.cell
def _():
    import hashlib
    import json
    import os
    import sys
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import marimo as _mo

    ROOT = _mo.notebook_dir().parent.parent
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.corpus import (analogy_accuracy, binary_subtrees, full_embedding, load_corpus, log_conditional,
                                 read_analogies, wordnet_hierarchy)
    from src.code.geometry import predicted_geometry
    from src.code import plot_utils
    from src.code.analysis_utils import cka, cka_null, cka_shuffle_null, rsa
    from src.code.llm import word_means
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return (AutoModelForCausalLM, AutoTokenizer, CACHE, FIGURES, ListedColormap, analogy_accuracy, binary_subtrees, cka,
            cka_null, cka_shuffle_null, full_embedding, hashlib, load_corpus, log_conditional, np, plot_utils, plt,
            predicted_geometry, read_analogies, rsa, torch, word_means, wordnet_hierarchy)


@app.cell
def _(CACHE, load_corpus):
    TAGS = {"wikipedia": "wikipedia_20231101_en", "gutenberg": "gutenberg_en", "fineweb": "fineweb_edu_10bt"}
    corpora = {(c, w): load_corpus(CACHE / "corpus" / tag, weighting=w) for c, tag in TAGS.items() for w in ("linear", "1/d")}
    index = {c: {w: i for i, w in enumerate(corpora[c, "linear"][1])} for c in TAGS}
    vocab = {c: corpora[c, "linear"][1].tolist() for c in TAGS}
    for _c in TAGS:
        print(f"{_c:10s} {len(vocab[_c]):,} words; linear and 1/d window-16 matrices")
    return TAGS, corpora, index, vocab


@app.cell
def _(mo):
    view_corpus = mo.ui.dropdown(options=["wikipedia", "gutenberg", "fineweb"], value="wikipedia", label="corpus shown in panels")
    view_corpus
    return (view_corpus,)


@app.cell
def _(CACHE, TAGS, cka, corpora, full_embedding, index, log_conditional, np, predicted_geometry, rsa):
    def embedding(c, weighting, V, target, d, label=None):
        """Cached (V, d) full-vocabulary embedding of the corpus: rows are its first V words, or the words at indices V."""
        path = CACHE / "corpus" / TAGS[c] / f"emb_{weighting.replace('/', '')}_{target}_V{label or V}_d{d}.npy"
        if not path.exists():
            np.save(path, full_embedding(corpora[c, weighting][0], V, target, d).astype(np.float32))
        return np.load(path)


    def block_code(c, weighting, words):
        """Our code: the balanced factor of the words' own row-centred log P(j | i) block."""
        C, _, _, r = corpora[c, weighting]
        ids = [index[c][w] for w in words]
        return predicted_geometry(log_conditional(C[np.ix_(ids, ids)], r[ids]))


    def scores(E, R):
        return {"RSA": rsa(E, R), "CKA": cka(E, R)}


    return block_code, embedding, scores


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, CACHE, hashlib, np, torch, word_means):
    GEMMA = "google/gemma-2-2b"
    GEMMA_DIR = CACHE / "corpus" / "llm" / "gemma-2-2b"; GEMMA_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES = {"months": "The month of the year is", "years": "In the year", "states": "The location of the US state", "word": ""}
    _loaded = {}


    def gemma_model():
        if not _loaded:
            _loaded["tok"] = AutoTokenizer.from_pretrained(GEMMA)
            _loaded["model"] = AutoModelForCausalLM.from_pretrained(GEMMA, dtype=torch.bfloat16, device_map="cuda").eval()
        return _loaded["tok"], _loaded["model"]


    def gemma_reps(name, forms, template):
        """(layers, n, d) residuals at the prompt's final token, one prompt per surface form: Karkada's template
        followed by the form, or the bare form after BOS. Layer 0 is the (tied) embedding, i.e. the unembedding."""
        path = GEMMA_DIR / f"{name}_{hashlib.md5(' '.join([template, *forms]).encode()).hexdigest()[:8]}.npz"
        if not path.exists():
            tok, model = gemma_model()
            pre = [tok.bos_token_id] + (tok.encode(template, add_special_tokens=False) if template else [])
            reps = word_means(model, {f: tok.encode(" " + f, add_special_tokens=False) for f in forms}, [(pre, [])], at_end=True)
            np.savez(path, forms=np.array(forms), residual=reps.astype(np.float16))
            print(f"gemma {name}: {len(forms)} forms cached")
        return np.load(path)["residual"].astype(np.float32)


    def gemma_whitener():
        """(mu, Sigma^-1/2) of Gemma's embedding table scaled as layer 0 is (Nava & Wyart whiten the unembeddings so)."""
        path = GEMMA_DIR / "whitener.npz"
        if not path.exists():
            tok, model = gemma_model()
            W = model.get_input_embeddings().weight.detach().float() * model.config.hidden_size ** 0.5
            mu = W.mean(0); X = W - mu
            lam, U = torch.linalg.eigh(X.T @ X / len(X))
            np.savez(path, mu=mu.cpu().numpy(), isqrt=(U * lam.clamp(min=1e-6) ** -0.5 @ U.T).cpu().numpy())
        z = np.load(path)
        return z["mu"], z["isqrt"]


    def whitened(E):
        mu, isqrt = gemma_whitener()
        return (E - mu) @ isqrt


    return TEMPLATES, gemma_reps, whitened


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Years (Karkada et al.)

    Their setting: Wikipedia, 25k words, window 16 with weights 16+1-d, M* = 2 tanh(PMI/2), top 1000 modes; years 1700-2020 read from
    Gemma as "In the year x" at the final token. Main panel: the year rows of each code beside Gemma's best layer.
    """)
    return


@app.cell
def _(TAGS, block_code, corpora, embedding, index):
    V_K, D_K = 25000, 1000                                                   # Karkada's vocabulary and embedding dimension
    YEARS_ALL = [str(y) for y in range(1700, 2021)]


    def in_vocab(c, words, V=V_K):
        return [w for w in words if index[c].get(w, V) < V]


    def code_rows(c, words, weighting="linear", V=V_K, d=D_K, target="mstar"):
        """Their embedding rows, our log-conditional embedding rows, and our block code, for words of the corpus."""
        ids = [index[c][w] for w in words]
        return {"M* embedding": embedding(c, weighting, V, target, d)[ids],
                "log-cond embedding": embedding(c, weighting, V, "logcond", d)[ids],
                "block code": block_code(c, weighting, words)}


    years = {c: in_vocab(c, YEARS_ALL) for c in TAGS}
    for _c in TAGS:
        print(f"{_c:10s} {len(years[_c])} of {len(YEARS_ALL)} years in the top {V_K:,} words ({years[_c][0]}-{years[_c][-1]})")
    return D_K, V_K, YEARS_ALL, code_rows, in_vocab, years


@app.cell
def _(TAGS, TEMPLATES, YEARS_ALL, code_rows, gemma_reps, np, rsa, years):
    _G = gemma_reps("years", YEARS_ALL, TEMPLATES["years"])                  # (layers, 321, d)
    _pos = {y: i for i, y in enumerate(YEARS_ALL)}
    gemma_years = {c: _G[:, [_pos[y] for y in years[c]]] for c in TAGS}
    year_codes = {c: code_rows(c, years[c]) for c in TAGS}
    year_rsa = {c: np.array([[rsa(E, gemma_years[c][l]) for E in year_codes[c].values()] for l in range(_G.shape[0])]) for c in TAGS}
    YEAR_LAYER = int(np.mean([v.mean(1) for v in year_rsa.values()], 0).argmax())
    print(f"Gemma layer {YEAR_LAYER} (best mean RSA over codes and corpora)")
    return YEAR_LAYER, gemma_years, year_codes, year_rsa


@app.cell
def _(FIGURES, YEAR_LAYER, gemma_years, plot_utils, scores, view_corpus, year_codes):
    def years_panel(c):
        G = gemma_years[c][YEAR_LAYER]
        fig = plot_utils.plot_embedding_panels(G, year_codes[c], None, cmap="magma", scores={k: scores(E, G) for k, E in year_codes[c].items()},
                                               first_name=f"Gemma L{YEAR_LAYER}", starred="rest")
        fig.suptitle(f"years, {c}: Gemma-2-2B beside their M* embedding, the log-conditional embedding and our block code", y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"wyart_years_{c}.png")


    years_panel(view_corpus.value)
    return


@app.cell
def _(FIGURES, TAGS, YEAR_LAYER, cka, cka_null, cka_shuffle_null, gemma_years, np, plot_utils, plt, year_codes, year_rsa):
    def layer_curves(name, rsa_by, cka_by, layer, labels, fname):
        """RSA (solid) and CKA (dashed) of each code against Gemma, by layer, one panel per corpus."""
        fig, axes = plt.subplots(1, len(TAGS), figsize=(4 * len(TAGS), 3.2), sharey=True)
        for ax, c in zip(axes, TAGS):
            for j, lab in enumerate(labels):
                line, = ax.plot(rsa_by[c][:, j], label=lab)
                ax.plot(cka_by[c][:, j], ls="--", lw=0.9, color=line.get_color())
            ax.axvline(layer, color="0.8", lw=0.8); ax.axhline(0, color="0.7", lw=0.8); ax.set_title(c); ax.set_xlabel("Gemma layer"); ax.set_ylim(-0.3, 1)
        axes[0].set_ylabel(f"{name}: code vs Gemma\nRSA (solid), CKA (dashed)"); axes[0].legend(frameon=False, fontsize=8)
        return plot_utils.save_figure(fig, FIGURES / fname)


    year_cka = {c: np.array([[cka(E, gemma_years[c][l]) for E in year_codes[c].values()] for l in range(gemma_years[c].shape[0])]) for c in TAGS}
    for _c in TAGS:
        _G = gemma_years[_c][YEAR_LAYER]
        print(f"{_c:10s} " + "  ".join(f"{k} {year_rsa[_c][YEAR_LAYER, j]:.2f}/{year_cka[_c][YEAR_LAYER, j]:.2f}" for j, k in enumerate(year_codes[_c]))
              + f"   CKA floor {cka_null(_G):.2f}/{cka_shuffle_null(_G):.2f}")
    layer_curves("years", year_rsa, year_cka, YEAR_LAYER, list(year_codes["wikipedia"]), "wyart_years_layers.png")
    return (layer_curves,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Their tests.** Lissajous curves: pairs of the year rows' PCs trace parametric sinusoids; we fit a sinusoid of the
    year to each PC and report R². Decoding: ridgeless regression of the year on the top r PCs for years from 1900,
    half the years to fit and half to test, 100 splits, MSE vs r (their Figure 2).
    """)
    return


@app.cell
def _(np):
    def pcs(E):
        E = np.asarray(E, float); E = E - E.mean(0)
        U, S, _ = np.linalg.svd(E, full_matrices=False)
        return U * S


    def sinusoid_r2(x, t):
        """Best R² of x(t) = a sin(w t + phi) + c over a grid of frequencies (linear least squares per frequency)."""
        t = (np.asarray(t, float) - t.min()) / (t.max() - t.min())
        best = 0.0
        for w in np.linspace(0.25, 6, 400) * np.pi:
            X = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
            r = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
            best = max(best, 1 - (r ** 2).sum() / ((x - x.mean()) ** 2).sum())
        return best


    def decoding_curve(E, y, ranks, splits=100, seed=0):
        """Normalised MSE (train, test) of ridgeless regression of y on the top-r PCs of E, halves at random."""
        P = pcs(E); rng = np.random.default_rng(seed); y = np.asarray(y, float); n = len(y)
        out = np.zeros((len(ranks), 2))
        for s in range(splits):
            perm = rng.permutation(n); tr, te = perm[: n // 2], perm[n // 2:]
            for i, r in enumerate(ranks):
                X = np.column_stack([P[:, :r], np.ones(n)])
                beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
                out[i] += [((X[tr] @ beta - y[tr]) ** 2).mean(), ((X[te] @ beta - y[te]) ** 2).mean()]
        return out / splits / y.var()


    return decoding_curve, pcs, sinusoid_r2


@app.cell
def _(FIGURES, YEAR_LAYER, gemma_years, np, pcs, plot_utils, plt, sinusoid_r2, view_corpus, year_codes, years):
    def lissajous(c):
        codes = {f"Gemma L{YEAR_LAYER}": gemma_years[c][YEAR_LAYER], **year_codes[c]}
        t = np.array([int(y) for y in years[c]])
        pairs = [(0, 1), (0, 2), (1, 2)]
        fig, axes = plt.subplots(len(codes), len(pairs), figsize=(3 * len(pairs), 2.9 * len(codes)))
        for row, (name, E) in zip(axes, codes.items()):
            P = pcs(E)
            r2 = [sinusoid_r2(P[:, k], t) for k in range(3)]
            for ax, (a, b) in zip(row, pairs):
                ax.scatter(P[:, a], P[:, b], c=t, cmap="magma", s=8)
                ax.set_xlabel(f"PC {a + 1}"); ax.set_ylabel(f"PC {b + 1}"); ax.set_xticks([]); ax.set_yticks([])
            row[0].set_title(f"{name}   sinusoid R² PC1-3: {r2[0]:.2f} {r2[1]:.2f} {r2[2]:.2f}", loc="left", fontsize=9)
        fig.suptitle(f"years {years[c][0]}-{years[c][-1]}, {c}: PC pairs coloured by year (Karkada Fig. 2 left)", y=1.0)
        fig.tight_layout()
        return plot_utils.save_figure(fig, FIGURES / f"wyart_years_lissajous_{c}.png")


    lissajous(view_corpus.value)
    return


@app.cell
def _(FIGURES, YEAR_LAYER, decoding_curve, gemma_years, np, plot_utils, plt, view_corpus, year_codes, years):
    def year_decoding(c):
        keep = [i for i, y in enumerate(years[c]) if int(y) >= 1900]
        y = np.array([int(years[c][i]) for i in keep])
        ranks = [r for r in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48] if r < len(keep) // 2 - 1]
        codes = {f"Gemma L{YEAR_LAYER}": gemma_years[c][YEAR_LAYER][keep], **{k: E[keep] for k, E in year_codes[c].items()}}
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for name, E in codes.items():
            mse = decoding_curve(E, y, ranks)
            line, = ax.plot(ranks, mse[:, 1], marker="o", ms=3, label=name)
            ax.plot(ranks, mse[:, 0], ls="--", lw=0.9, color=line.get_color())
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("rank r of projected embedding"); ax.set_ylabel("MSE / var(year)")
        ax.set_title(f"decoding years {y.min()}-{y.max()}, {c}: test (solid), train (dashed)", fontsize=9); ax.legend(frameon=False, fontsize=8)
        return plot_utils.save_figure(fig, FIGURES / f"wyart_years_decoding_{c}.png")


    year_decoding(view_corpus.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. US states (Karkada et al.)

    The 48 contiguous states, read from Gemma as "The location of the US state x". The full-vocabulary embeddings have no
    row for two-word names, so this setting uses the 38 single-word states throughout. Main panel: PCA of the state rows.
    """)
    return


@app.cell
def _(TAGS, TEMPLATES, code_rows, gemma_reps, np, rsa):
    CENTROIDS = {"alabama": (32.8, -86.8), "arizona": (34.3, -111.7), "arkansas": (34.9, -92.4), "california": (37.2, -119.5),
                 "colorado": (39.0, -105.5), "connecticut": (41.6, -72.7), "delaware": (39.0, -75.5), "florida": (28.6, -82.4),
                 "georgia": (32.7, -83.4), "idaho": (44.4, -114.6), "illinois": (40.0, -89.2), "indiana": (39.9, -86.3),
                 "iowa": (42.1, -93.5), "kansas": (38.5, -98.4), "kentucky": (37.5, -85.3), "louisiana": (31.1, -92.0),
                 "maine": (45.4, -69.2), "maryland": (39.0, -76.8), "massachusetts": (42.3, -71.8), "michigan": (44.3, -85.4),
                 "minnesota": (46.3, -94.3), "mississippi": (32.7, -89.7), "missouri": (38.4, -92.5), "montana": (47.0, -109.6),
                 "nebraska": (41.5, -99.8), "nevada": (39.3, -116.6), "ohio": (40.3, -82.8), "oklahoma": (35.6, -97.5),
                 "oregon": (43.9, -120.6), "pennsylvania": (40.9, -77.8), "tennessee": (35.9, -86.4), "texas": (31.5, -99.3),
                 "utah": (39.3, -111.7), "vermont": (44.1, -72.7), "virginia": (37.5, -78.8), "washington": (47.4, -120.5),
                 "wisconsin": (44.6, -90.0), "wyoming": (43.0, -107.5)}                   # (latitude, longitude) of the state centroid
    STATES = list(CENTROIDS)
    LATLON = np.array([CENTROIDS[s] for s in STATES])
    _G = gemma_reps("states", [s.title() for s in STATES], TEMPLATES["states"])
    gemma_states = {c: _G for c in TAGS}
    state_codes = {c: code_rows(c, STATES) for c in TAGS}
    state_rsa = {c: np.array([[rsa(E, _G[l]) for E in state_codes[c].values()] for l in range(_G.shape[0])]) for c in TAGS}
    STATE_LAYER = int(np.mean([v.mean(1) for v in state_rsa.values()], 0).argmax())
    print(f"{len(STATES)} single-word contiguous states; Gemma layer {STATE_LAYER}")
    return LATLON, STATES, STATE_LAYER, gemma_states, state_codes, state_rsa


@app.cell
def _(FIGURES, STATE_LAYER, gemma_states, plot_utils, scores, state_codes, view_corpus):
    def states_panel(c):
        G = gemma_states[c][STATE_LAYER]
        fig = plot_utils.plot_embedding_panels(G, state_codes[c], None, scores={k: scores(E, G) for k, E in state_codes[c].items()},
                                               first_name=f"Gemma L{STATE_LAYER}", starred="rest")
        fig.suptitle(f"states, {c}: Gemma-2-2B beside the three codes", y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"wyart_states_{c}.png")


    states_panel(view_corpus.value)
    return


@app.cell
def _(STATE_LAYER, TAGS, cka, cka_null, cka_shuffle_null, gemma_states, layer_curves, np, state_codes, state_rsa):
    state_cka = {c: np.array([[cka(E, gemma_states[c][l]) for E in state_codes[c].values()] for l in range(gemma_states[c].shape[0])]) for c in TAGS}
    for _c in TAGS:
        _G = gemma_states[_c][STATE_LAYER]
        print(f"{_c:10s} " + "  ".join(f"{k} {state_rsa[_c][STATE_LAYER, j]:.2f}/{state_cka[_c][STATE_LAYER, j]:.2f}" for j, k in enumerate(state_codes[_c]))
              + f"   CKA floor {cka_null(_G):.2f}/{cka_shuffle_null(_G):.2f}")
    layer_curves("states", state_rsa, state_cka, STATE_LAYER, list(state_codes["wikipedia"]), "wyart_states_layers.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Their tests.** Eigenmodes on the map: the top three PCs of each code's centred state Gram, drawn as colours at the
    state centroids (their Figure 3; they compare with a distance-kernel model, which we leave out). Coordinate
    decoding: leave-one-out ridgeless regression of latitude and longitude on the top r PCs, R² vs r.
    """)
    return


@app.cell
def _(FIGURES, LATLON, STATES, STATE_LAYER, gemma_states, pcs, plot_utils, plt, state_codes, view_corpus):
    def state_modes(c):
        codes = {f"Gemma L{STATE_LAYER}": gemma_states[c][STATE_LAYER], **state_codes[c]}
        fig, axes = plt.subplots(len(codes), 3, figsize=(10, 2.4 * len(codes)))
        for row, (name, E) in zip(axes, codes.items()):
            P = pcs(E)
            for k, ax in enumerate(row):
                lim = abs(P[:, k]).max()
                ax.scatter(LATLON[:, 1], LATLON[:, 0], c=P[:, k], cmap=plot_utils.FIG.diverging_cmap, vmin=-lim, vmax=lim, s=60)
                for s, (lat, lon) in zip(STATES, LATLON):
                    ax.annotate(s[:2], (lon, lat), fontsize=5, ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect(1 / 0.78)
                ax.set_title(f"{name}: mode {k + 1}", fontsize=8)
        fig.suptitle(f"states, {c}: top Gram eigenmodes on the map (Karkada Fig. 3)", y=1.0)
        fig.tight_layout()
        return plot_utils.save_figure(fig, FIGURES / f"wyart_states_modes_{c}.png")


    state_modes(view_corpus.value)
    return


@app.cell
def _(FIGURES, LATLON, STATE_LAYER, gemma_states, np, pcs, plot_utils, plt, state_codes, view_corpus):
    def loo_r2(E, Y, ranks):
        """Leave-one-out ridgeless R² of Y (n, 2) from the top-r PCs of E, averaged over the two coordinates."""
        P = pcs(E); n = len(Y); out = []
        for r in ranks:
            X = np.column_stack([P[:, :r], np.ones(n)]); pred = np.zeros_like(Y)
            for i in range(n):
                m = np.arange(n) != i
                pred[i] = X[i] @ np.linalg.lstsq(X[m], Y[m], rcond=None)[0]
            out.append(np.mean(1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)))
        return np.array(out)


    def state_decoding(c):
        ranks = [1, 2, 3, 4, 6, 8, 10, 12]
        codes = {f"Gemma L{STATE_LAYER}": gemma_states[c][STATE_LAYER], **state_codes[c]}
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for name, E in codes.items():
            r2 = loo_r2(E, LATLON, ranks)
            ax.plot(ranks, r2, marker="o", ms=3, label=name)
            print(f"{c:10s} {name:20s} R² at r=6: {r2[ranks.index(6)]:.2f}, best {r2.max():.2f} at r={ranks[int(r2.argmax())]}")
        ax.axhline(0.5, color="0.7", lw=0.8); ax.set_ylim(-0.5, 1); ax.set_xlabel("rank r"); ax.set_ylabel("leave-one-out R² (lat, lon)")
        ax.set_title(f"decoding state coordinates, {c}", fontsize=9); ax.legend(frameon=False, fontsize=8)
        return plot_utils.save_figure(fig, FIGURES / f"wyart_states_decoding_{c}.png")


    state_decoding(view_corpus.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Months (Karkada et al.)

    "The month of the year is x". Main panel: the month rows on the ring. Their test is the ring itself in the full
    embedding; we count how many of the 12 calendar-adjacent pairs are adjacent around PC1-PC2.
    """)
    return


@app.cell
def _(TAGS, TEMPLATES, cka, cka_null, cka_shuffle_null, code_rows, gemma_reps, layer_curves, np, pcs, rsa):
    MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    gemma_months = gemma_reps("months", [m.capitalize() for m in MONTHS], TEMPLATES["months"])
    month_codes = {c: code_rows(c, MONTHS) for c in TAGS}
    month_rsa = {c: np.array([[rsa(E, gemma_months[l]) for E in month_codes[c].values()] for l in range(gemma_months.shape[0])]) for c in TAGS}
    month_cka = {c: np.array([[cka(E, gemma_months[l]) for E in month_codes[c].values()] for l in range(gemma_months.shape[0])]) for c in TAGS}
    MONTH_LAYER = int(np.mean([v.mean(1) for v in month_rsa.values()], 0).argmax())


    def ring_adjacency(E):
        """How many of the 12 calendar neighbours are angular neighbours in the PC1-PC2 plane (12 = a perfect ring)."""
        P = pcs(E)[:, :2]; order = np.argsort(np.arctan2(P[:, 1], P[:, 0]))
        nxt = {order[i]: order[(i + 1) % 12] for i in range(12)}
        return sum(nxt[i] == (i + 1) % 12 or nxt[(i + 1) % 12] == i for i in range(12))


    _G = gemma_months[MONTH_LAYER]
    print(f"Gemma layer {MONTH_LAYER}; Gemma ring {ring_adjacency(_G)}/12; CKA floor {cka_null(_G):.2f}/{cka_shuffle_null(_G):.2f}")
    for _c in TAGS:
        print(f"{_c:10s} " + "  ".join(f"{k} {month_rsa[_c][MONTH_LAYER, j]:.2f}/{month_cka[_c][MONTH_LAYER, j]:.2f} ring {ring_adjacency(E)}/12"
                                       for j, (k, E) in enumerate(month_codes[_c].items())))
    layer_curves("months", month_rsa, month_cka, MONTH_LAYER, list(month_codes["wikipedia"]), "wyart_months_layers.png")
    return MONTH_LAYER, gemma_months, month_codes


@app.cell
def _(FIGURES, MONTH_LAYER, gemma_months, month_codes, plot_utils, scores, view_corpus):
    _c = view_corpus.value
    _G = gemma_months[MONTH_LAYER]
    _fig = plot_utils.plot_embedding_panels(_G, month_codes[_c], {"structure": "ring", "n": 12},
                                            scores={k: scores(E, _G) for k, E in month_codes[_c].items()},
                                            first_name=f"Gemma L{MONTH_LAYER}", starred="rest")
    _fig.suptitle(f"months, {_c}: Gemma-2-2B beside the three codes", y=1.02)
    plot_utils.save_figure(_fig, FIGURES / f"wyart_months_{_c}.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. WordNet trees (Nava & Wyart)

    Their setting: Wikipedia, weights 16/d, vocabulary = the WordNet noun lemmas seen in the corpus (here: those within our
    50k words), M* over that vocabulary, top 2048 modes; Gemma's unembeddings (= layer 0) with and without whitening.
    Trees: depth-3 perfect binary subtrees of the contracted hierarchy under organism, artifact, event, 40 each.
    Main panel: one tree as ultrametric dendrograms, Gemma's best layer beside the three codes.
    """)
    return


@app.cell
def _(TAGS, binary_subtrees, np, vocab, wordnet_hierarchy):
    from nltk.corpus import wordnet as wn
    ROOTS = {"organism": "organism.n.01", "artifact": "instrumentality.n.03", "event": "event.n.01"}
    N_TREES, DEPTH = 40, 3
    EXCLUDE = {"cricket", "queen", "nobody", "gallant", "widower", "beau", "coon", "buff",
               "opener", "shear", "crunch", "dealings", "scourge", "gig", "transit", "gourd", "binder", "trapping"}


    def is_noun_lemma(w):
        return w.isalpha() and w.islower() and any(l.name() == w and l.synset().pos() == "n" for l in wn.lemmas(w))


    wn_words = {c: [w for w in vocab[c] if is_noun_lemma(w)] for c in TAGS}          # their vocabulary: WordNet noun lemmas in the corpus
    _others = [set(vocab[c]) for c in list(TAGS)[1:]]
    _shared = [w for w in vocab["wikipedia"] if all(w in s for s in _others) and w not in EXCLUDE]
    _children, _word = wordnet_hierarchy(_shared)
    trees = {r: [[[_word[s] for s in level] for level in t] for t in binary_subtrees(_children.get, wn.synset(name), DEPTH, N_TREES, seed=0)]
             for r, name in ROOTS.items()}
    tree_words = sorted({w for ts in trees.values() for t in ts for level in t for w in level})
    for _c in TAGS:
        print(f"{_c:10s} {len(wn_words[_c]):,} WordNet noun lemmas in its vocabulary")
    for _r, _ts in trees.items():
        print(f"{_r:10s} {len(_ts)} trees; e.g. " + " / ".join(", ".join(l) for l in _ts[0]))
    hierarchy = (_children, _word)
    return DEPTH, ROOTS, hierarchy, is_noun_lemma, tree_words, trees, wn, wn_words


@app.cell
def _(TAGS, TEMPLATES, block_code, embedding, gemma_reps, index, np, rsa, tree_words, trees, whitened, wn_words):
    D_N = 2048


    def tree_nodes(tree):
        return [w for level in tree for w in level]


    def wn_embedding(c, target):
        """Their embedding over the corpus's WordNet noun lemmas, 1/d weights: rows aligned with wn_words[c]."""
        ids = np.array([index[c][w] for w in wn_words[c]])
        return embedding(c, "1/d", ids, target, D_N, label=f"wn{len(ids)}")


    _pos = {c: {w: i for i, w in enumerate(wn_words[c])} for c in TAGS}
    _emb = {(c, t): wn_embedding(c, t) for c in TAGS for t in ("mstar", "logcond")}


    def tree_codes(c, tree):
        words = tree_nodes(tree); rows = [_pos[c][w] for w in words]
        return {"M* embedding": _emb[c, "mstar"][rows], "log-cond embedding": _emb[c, "logcond"][rows], "block code": block_code(c, "1/d", words)}


    _G = gemma_reps("tree_words", tree_words, TEMPLATES["word"])
    _tpos = {w: i for i, w in enumerate(tree_words)}
    gemma_tree = {"layers": _G, "L0 whitened": whitened(_G[0])}


    def gemma_tree_code(tree, layer):
        rows = [_tpos[w] for w in tree_nodes(tree)]
        return gemma_tree["L0 whitened"][rows] if layer == "L0 whitened" else gemma_tree["layers"][layer][rows]


    CODE_NAMES = ("M* embedding", "log-cond embedding", "block code")
    tree_code_sets = {(r, c): [tree_codes(c, t) for t in ts] for r, ts in trees.items() for c in TAGS}     # (root, corpus) -> per tree


    def tree_scores(metric, layer):
        """(root, corpus) -> (codes,) mean metric over the root's trees between each code and Gemma at ``layer``."""
        return {(r, c): np.array([np.mean([metric(codes[k], gemma_tree_code(t, layer)) for t, codes in zip(trees[r], tree_code_sets[r, c])])
                                  for k in CODE_NAMES]) for r in trees for c in TAGS}


    tree_rsa = {rc: np.array([v[rc] for v in [tree_scores(rsa, l) for l in range(_G.shape[0])]]) for rc in tree_code_sets}   # (layers, codes)
    TREE_LAYER = int(np.mean([v.mean(1) for v in tree_rsa.values()], 0).argmax())
    print(f"Gemma layer {TREE_LAYER}")
    return CODE_NAMES, TREE_LAYER, gemma_tree_code, tree_code_sets, tree_codes, tree_nodes, tree_rsa, tree_scores


@app.cell
def _(mo, trees):
    view_root = mo.ui.dropdown(options=list(trees), value="organism", label="root")
    view_tree = mo.ui.dropdown(options=[str(i) for i in range(len(trees["organism"]))], value="0", label="tree")
    mo.hstack([view_root, view_tree], justify="start")
    return view_root, view_tree


@app.cell
def _(FIGURES, ListedColormap, TREE_LAYER, gemma_tree_code, plot_utils, scores, tree_codes, trees, view_corpus, view_root, view_tree):
    LEAF_COLORS = ["#1f4e9c"] * 2 + ["#7fb2e5"] * 2 + ["#b22222"] * 2 + ["#f4a582"] * 2


    def tree_panel(c, root, i):
        tree = trees[root][i]
        spec = {"structure": "tree", "parents": [-1] + [(u - 1) // 2 for u in range(1, 15)], "view": "ultrametric"}
        G = gemma_tree_code(tree, TREE_LAYER); codes = tree_codes(c, tree)
        fig = plot_utils.plot_embedding_panels(G, codes, spec, cmap=ListedColormap(["0.5"] * 7 + LEAF_COLORS),
                                               scores={k: scores(E, G) for k, E in codes.items()}, first_name=f"Gemma L{TREE_LAYER}", starred="rest")
        fig.suptitle(f"{root} tree {i}, {c}: " + " / ".join(", ".join(l) for l in tree), y=1.04, fontsize=9)
        return plot_utils.save_figure(fig, FIGURES / f"wyart_tree_{root}_{c}.png")


    tree_panel(view_corpus.value, view_root.value, int(view_tree.value))
    return


@app.cell
def _(CODE_NAMES, FIGURES, TAGS, TREE_LAYER, cka, cka_null, cka_shuffle_null, gemma_tree_code, np, plot_utils, plt, tree_rsa, tree_scores, trees):
    tree_cka = {rc: np.array([v[rc] for v in [tree_scores(cka, l) for l in range(next(iter(tree_rsa.values())).shape[0])]]) for rc in tree_rsa}
    for _r in trees:
        _floor = np.mean([[cka_null(gemma_tree_code(t, TREE_LAYER)), cka_shuffle_null(gemma_tree_code(t, TREE_LAYER))] for t in trees[_r]], 0)
        for _c in TAGS:
            print(f"{_r:10s} {_c:10s} " + "  ".join(f"{k} {tree_rsa[_r, _c][TREE_LAYER, j]:.2f}/{tree_cka[_r, _c][TREE_LAYER, j]:.2f}" for j, k in enumerate(CODE_NAMES))
                  + (f"   CKA floor {_floor[0]:.2f}/{_floor[1]:.2f}" if _c == "wikipedia" else ""))


    def tree_layer_curves():
        fig, axes = plt.subplots(len(trees), len(TAGS), figsize=(4 * len(TAGS), 2.8 * len(trees)), sharey=True, sharex=True)
        for row, r in zip(axes, trees):
            for ax, c in zip(row, TAGS):
                for j, lab in enumerate(CODE_NAMES):
                    line, = ax.plot(tree_rsa[r, c][:, j], label=lab); ax.plot(tree_cka[r, c][:, j], ls="--", lw=0.9, color=line.get_color())
                ax.axvline(TREE_LAYER, color="0.8", lw=0.8); ax.axhline(0, color="0.7", lw=0.8); ax.set_title(f"{r}, {c}", fontsize=9); ax.set_ylim(-0.3, 1)
            row[0].set_ylabel("RSA (solid), CKA (dashed)")
        for ax in axes[-1]:
            ax.set_xlabel("Gemma layer")
        axes[0, 0].legend(frameon=False, fontsize=8)
        return plot_utils.save_figure(fig, FIGURES / "wyart_trees_layers.png")


    tree_layer_curves()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Their tests.** (a) Co-occurrence vs tree distance: mean M* and mean row-centred log P(j | i) over all pairs of
    named synsets at contracted-hierarchy distance 1-6 (their Figure 2; criterion: monotone decay). (b) Eigenspace
    alignment: per tree, g(k) = (1/k)‖U_kᵀV_k‖² between the top-k eigenvectors of the corpus code's centred Gram and
    Gemma's, k = 1..14, and the area above the k/14 diagonal, against a within-tree shuffle of Gemma's rows (their
    Figure 4, with Gemma in place of their kernel model). (c) Their organism example tree.
    """)
    return


@app.cell
def _(FIGURES, TAGS, corpora, hierarchy, index, log_conditional, np, plot_utils, plt, wn_words):
    def tree_distances(children, nodes):
        """Pairwise path lengths between nodes of the arborescence given as {parent: [children]}."""
        parent = {c: p for p, cs in children.items() for c in cs}
        paths = []
        for u in nodes:
            path = [u]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            paths.append(path[::-1])
        D = max(map(len, paths)); ids = {}
        P = np.full((len(nodes), D), -1)
        for i, path in enumerate(paths):
            P[i, :len(path)] = [ids.setdefault(s, len(ids)) for s in path]
        depth = (P >= 0).sum(1)
        common = np.zeros((len(nodes), len(nodes)), int)
        for k in range(D):
            common += ((P[:, None, k] == P[None, :, k]) & (P[:, None, k] >= 0)) * (common == k)
        return depth[:, None] + depth[None, :] - 2 * common


    def distance_decay(c):
        """Mean M* and mean row-centred log-conditional over named-synset pairs at each tree distance."""
        children, word = hierarchy
        nodes = [s for s in word if word[s] in index[c]]
        D = tree_distances(children, nodes)
        C, _, _, _ = corpora[c, "1/d"]
        wn_ids = np.array([index[c][w] for w in wn_words[c]])
        pos = {w: i for i, w in enumerate(wn_words[c])}
        rows = np.array([pos[word[s]] for s in nodes])
        X = np.asarray(C[wn_ids][:, wn_ids], dtype=np.float64)           # their vocabulary block, 1/d weights
        P = X / X.sum(); p = P.sum(1)
        M = 2 * (P - np.outer(p, p)) / (P + np.outer(p, p))
        L = log_conditional(X, X.sum(1)); L = L - L.mean(1, keepdims=True)
        out = {}
        for d in range(1, 7):
            m = np.triu(D == d, 1)
            out[d] = (M[np.ix_(rows, rows)][m].mean(), L[np.ix_(rows, rows)][m].mean(), int(m.sum()))
        return out


    decay = {c: distance_decay(c) for c in TAGS}
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for _c in TAGS:
        _d = decay[_c]
        _axes[0].plot(list(_d), [v[0] for v in _d.values()], marker="o", label=_c)
        _axes[1].plot(list(_d), [v[1] for v in _d.values()], marker="o", label=_c)
        print(f"{_c:10s} pairs per distance " + ", ".join(f"{d}: {v[2]:,}" for d, v in _d.items()))
    _axes[0].set_ylabel("mean M*"); _axes[1].set_ylabel("mean row-centred log P(j | i)")
    for _ax in _axes:
        _ax.set_xlabel("contracted WordNet distance"); _ax.axhline(0, color="0.7", lw=0.8)
    _axes[0].legend(frameon=False, fontsize=8)
    _fig.suptitle("co-occurrence vs tree distance over the named synsets (Nava & Wyart Fig. 2)", y=1.02)
    plot_utils.save_figure(_fig, FIGURES / "wyart_trees_decay.png")
    return


@app.cell
def _(CODE_NAMES, FIGURES, TAGS, TREE_LAYER, gemma_tree_code, np, plot_utils, plt, tree_code_sets, trees):
    def top_eigvecs(E):
        E = np.asarray(E, float); E = E - E.mean(0)
        return np.linalg.svd(E, full_matrices=False)[0]


    def alignment(E, R, ks=range(1, 15)):
        """g(k) = |U_k^T V_k|_F^2 / k between the top-k eigenvectors of the two centred Grams."""
        U, V = top_eigvecs(E), top_eigvecs(R)
        return np.array([np.linalg.norm(U[:, :k].T @ V[:, :k]) ** 2 / k for k in ks])


    def alignment_curves(root, c, layer, shuffles=5, seed=0):
        rng = np.random.default_rng(seed); ks = np.arange(1, 15)
        out = {k: [] for k in (*CODE_NAMES, "shuffled")}
        for t, codes in zip(trees[root], tree_code_sets[root, c]):
            G = gemma_tree_code(t, layer)
            for k, E in codes.items():
                out[k].append(alignment(E, G))
            out["shuffled"].append(np.mean([alignment(codes["M* embedding"], G[rng.permutation(15)]) for _ in range(shuffles)], 0))
        return ks, {k: np.array(v) for k, v in out.items()}


    def alignment_figure(layer):
        fig, axes = plt.subplots(len(trees), len(TAGS), figsize=(4 * len(TAGS), 2.8 * len(trees)), sharey=True, sharex=True)
        areas = {}
        for row, r in zip(axes, trees):
            for ax, c in zip(row, TAGS):
                ks, g = alignment_curves(r, c, layer)
                for k, v in g.items():
                    ax.plot(ks, v.mean(0), marker="o", ms=3, label=k, color="0.5" if k == "shuffled" else None, ls=":" if k == "shuffled" else "-")
                    areas[r, c, k] = float((v.mean(0) - ks / 14).sum())
                ax.plot(ks, ks / 14, color="0.85", lw=0.8); ax.set_title(f"{r}, {c}", fontsize=9); ax.set_ylim(0, 1)
            row[0].set_ylabel("g(k)")
        for ax in axes[-1]:
            ax.set_xlabel("top eigenspace dimension k")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"top-k eigenspace alignment, corpus code vs Gemma {layer if isinstance(layer, str) else f'L{layer}'} (Nava & Wyart Fig. 4)", y=1.0)
        return plot_utils.save_figure(fig, FIGURES / f"wyart_trees_alignment_{str(layer).replace(' ', '_')}.png"), areas


    _fig, _areas = alignment_figure(TREE_LAYER)
    print(f"alignment area (sum of g(k) - k/14) vs Gemma L{TREE_LAYER}:")
    for _r in trees:
        for _c in TAGS:
            print(f"  {_r:10s} {_c:10s} " + "  ".join(f"{k} {_areas[_r, _c, k]:5.2f}" for k in (*CODE_NAMES, "shuffled")))
    for _layer in (0, "L0 whitened"):
        _, _a = alignment_figure(_layer)
        print(f"vs Gemma {_layer}: " + "  ".join(f"{_r} {np.mean([_a[_r, c, 'M* embedding'] for c in TAGS]):.2f} (shuffled {np.mean([_a[_r, c, 'shuffled'] for c in TAGS]):.2f})" for _r in trees))
    _fig
    return


@app.cell
def _(TEMPLATES, block_code, embedding, gemma_reps, index, np, wn_words):
    EXAMPLE = [["organism"], ["animal", "plant"], ["bird", "fish", "flower", "tree"],
               ["owl", "eagle", "shark", "salmon", "daisy", "poppy", "willow", "pine"]]                 # Nava & Wyart's Figure 1
    example_words = [w for level in EXAMPLE for w in level]
    gemma_example = gemma_reps("example_tree", example_words, TEMPLATES["word"])


    def example_codes(c):
        pos = {w: i for i, w in enumerate(wn_words[c])}; rows = [pos[w] for w in example_words]
        ids = np.array([index[c][w] for w in wn_words[c]])
        return {"M* embedding": embedding(c, "1/d", ids, "mstar", 2048, label=f"wn{len(ids)}")[rows],
                "log-cond embedding": embedding(c, "1/d", ids, "logcond", 2048, label=f"wn{len(ids)}")[rows],
                "block code": block_code(c, "1/d", example_words)}


    return EXAMPLE, example_codes, example_words, gemma_example


@app.cell
def _(EXAMPLE, FIGURES, ListedColormap, TREE_LAYER, example_codes, gemma_example, plot_utils, scores, view_corpus):
    def example_panel(c):
        spec = {"structure": "tree", "parents": [-1] + [(u - 1) // 2 for u in range(1, 15)], "view": "ultrametric"}
        codes = example_codes(c); G = gemma_example[TREE_LAYER]
        fig = plot_utils.plot_embedding_panels(G, codes, spec, cmap=ListedColormap(["0.5"] * 7 + ["#1f4e9c"] * 2 + ["#7fb2e5"] * 2 + ["#b22222"] * 2 + ["#f4a582"] * 2),
                                               scores={k: scores(E, G) for k, E in codes.items()}, first_name=f"Gemma L{TREE_LAYER}", starred="rest")
        fig.suptitle(f"their organism example, {c}: " + " / ".join(", ".join(l) for l in EXAMPLE), y=1.04, fontsize=9)
        return plot_utils.save_figure(fig, FIGURES / f"wyart_example_tree_{c}.png")


    example_panel(view_corpus.value)
    return


@app.cell
def _(FIGURES, TREE_LAYER, example_codes, example_words, gemma_example, pcs, plot_utils, plt, view_corpus, whitened):
    def split_modes(c):
        """PCs 1-4 of each code over the 15 nodes in level order (their Figure 3): split modes take opposite signs on sibling subtrees."""
        codes = {f"Gemma L{TREE_LAYER}": gemma_example[TREE_LAYER], "Gemma L0 whitened": whitened(gemma_example[0]), **example_codes(c)}
        fig, axes = plt.subplots(len(codes), 4, figsize=(12, 1.6 * len(codes)), sharex=True)
        for row, (name, E) in zip(axes, codes.items()):
            P = pcs(E)
            for k, ax in enumerate(row):
                ax.bar(range(15), P[:, k], color=["0.4"] + ["#1f4e9c", "#b22222"] + ["#1f4e9c"] * 2 + ["#b22222"] * 2 + ["#1f4e9c"] * 4 + ["#b22222"] * 4)
                ax.axhline(0, color="0.7", lw=0.6); ax.set_yticks([])
                if k == 0:
                    ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
        for k, ax in enumerate(axes[0]):
            ax.set_title(f"PC {k + 1}", fontsize=9)
        for ax in axes[-1]:
            ax.set_xticks(range(15)); ax.set_xticklabels(example_words, rotation=90, fontsize=6)
        fig.suptitle(f"split modes of the example tree, {c} (blue: animal side, red: plant side)", y=1.0)
        return plot_utils.save_figure(fig, FIGURES / f"wyart_example_modes_{c}.png")


    split_modes(view_corpus.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Analogies (Korchinski et al.)

    Their setting: Wikipedia, the 10k most frequent words, window 16 with weights 16+1-d, targets M = P_ij / (P_i P_j) and
    log(M + 0.01), embeddings U_K √λ_K, top-1 by Euclidean distance over all 10k words (query words excluded), Mikolov's 14
    families restricted to questions the vocabulary covers. Main panel: the analogy words coloured by family.
    """)
    return


@app.cell
def _(CACHE, TAGS, TEMPLATES, gemma_reps, index, np, read_analogies, vocab):
    V_KO = 10000
    _q = read_analogies(CACHE / "corpus/analogies/questions-words.txt")
    analogies = {c: {f: [tuple(w.lower() for w in t) for t in qs if all(index[c].get(w.lower(), V_KO) < V_KO for w in t)] for f, qs in _q.items()} for c in TAGS}
    analogy_form = {w.lower(): w for qs in _q.values() for t in qs for w in t}
    analogy_words = {c: sorted({w for qs in analogies[c].values() for t in qs for w in t}) for c in TAGS}
    for _c in TAGS:
        print(f"{_c:10s} {sum(map(len, analogies[_c].values())):,} questions over {len(analogy_words[_c])} words in its top {V_KO:,}; per family "
              + ", ".join(f"{f.split('-', 1)[-1][:12]} {len(qs)}" for f, qs in analogies[_c].items()))
    gemma_vocab = gemma_reps("wikipedia_10k", vocab["wikipedia"][:V_KO], TEMPLATES["word"])     # (layers, 10k, d): Gemma's version of the 10k-word embedding
    return V_KO, analogies, analogy_words, gemma_vocab


@app.cell
def _(TAGS, V_KO, analogies, analogy_accuracy, embedding, gemma_vocab, index, np, vocab):
    RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    TARGETS = {"M": "m", "log M": "logm", "log-cond": "logcond"}


    def accuracy_by_rank(c):
        """{target: (ranks, families)} top-1 Euclidean accuracy over the corpus's 10k words."""
        idx = {w: i for i, w in enumerate(vocab[c][:V_KO])}
        return {name: np.array([[analogy_accuracy(embedding(c, "linear", V_KO, t, 1000)[:, :K], idx, qs, metric="euclidean")
                                 for qs in analogies[c].values()] for K in RANKS]) for name, t in TARGETS.items()}


    acc = {c: accuracy_by_rank(c) for c in TAGS}
    _idx = {w: i for i, w in enumerate(vocab["wikipedia"][:V_KO])}
    acc_gemma = np.array([[analogy_accuracy(gemma_vocab[l], _idx, qs, metric="euclidean") for qs in analogies["wikipedia"].values()]
                          for l in range(gemma_vocab.shape[0])])                                     # (layers, families), Wikipedia's questions
    GEMMA_ANALOGY_LAYER = int(acc_gemma.mean(1).argmax())
    print(f"Gemma over the 10k Wikipedia words: best layer {GEMMA_ANALOGY_LAYER}, mean accuracy {acc_gemma[GEMMA_ANALOGY_LAYER].mean():.2f}")
    for _c in TAGS:
        print(f"{_c:10s} " + "  ".join(f"{n} peak {a.mean(1).max():.2f} at K={RANKS[int(a.mean(1).argmax())]}" for n, a in acc[_c].items()))
    return GEMMA_ANALOGY_LAYER, RANKS, acc, acc_gemma


@app.cell
def _(FIGURES, GEMMA_ANALOGY_LAYER, RANKS, TAGS, acc, acc_gemma, plot_utils, plt):
    def accuracy_figure():
        fig, axes = plt.subplots(1, len(TAGS) + 1, figsize=(4 * (len(TAGS) + 1), 3.3), sharey=True)
        for ax, c in zip(axes, TAGS):
            for name, a in acc[c].items():
                m, s = a.mean(1), a.std(1)
                ax.plot(RANKS, m, marker="o", ms=3, label=name); ax.fill_between(RANKS, m - s, m + s, alpha=0.1)
            ax.set_xscale("log", base=2); ax.set_title(c); ax.set_xlabel("embedding dimension K"); ax.set_ylim(0, 1)
        axes[0].set_ylabel("top-1 analogy accuracy (mean over families)"); axes[0].legend(frameon=False, fontsize=8)
        ax = axes[-1]; m, s = acc_gemma.mean(1), acc_gemma.std(1)
        ax.plot(m, color="0.3"); ax.fill_between(range(len(m)), m - s, m + s, color="0.3", alpha=0.12)
        ax.set_title("Gemma-2-2B, 10k Wikipedia words"); ax.set_xlabel("layer")
        fig.suptitle("Google analogies over the 10k most frequent words (Korchinski Fig. 2b)", y=1.03)
        return plot_utils.save_figure(fig, FIGURES / "wyart_analogies.png")


    accuracy_figure()
    return


@app.cell
def _(FIGURES, GEMMA_ANALOGY_LAYER, RANKS, acc, acc_gemma, analogies, np, plot_utils, plt, view_corpus):
    def family_bars(c):
        fig, ax = plt.subplots(figsize=(12, 3.4))
        fams = list(analogies[c]); x = np.arange(len(fams)); w = 0.8 / (len(acc[c]) + 1)
        if c == "wikipedia":
            ax.bar(x, acc_gemma[GEMMA_ANALOGY_LAYER], w, color="0.3", label=f"Gemma L{GEMMA_ANALOGY_LAYER}")
        for j, (name, a) in enumerate(acc[c].items(), 1):
            K = int(a.mean(1).argmax())
            ax.bar(x + j * w, a[K], w, label=f"{name} K={RANKS[K]}")
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(fams, rotation=35, ha="right", fontsize=8); ax.set_ylim(0, 1)
        ax.set_ylabel("top-1 accuracy"); ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        return plot_utils.save_figure(fig, FIGURES / f"wyart_analogy_families_{c}.png")


    family_bars(view_corpus.value)
    return


@app.cell
def _(FIGURES, GEMMA_ANALOGY_LAYER, V_KO, analogies, analogy_words, block_code, embedding, gemma_vocab, index, np, plot_utils, scores, view_corpus, vocab):
    def analogy_panel(c):
        widx = {w: i for i, w in enumerate(vocab["wikipedia"][:V_KO])}                  # Gemma reps cover Wikipedia's 10k words
        words = [w for w in analogy_words[c] if w in widx]
        idx = {w: i for i, w in enumerate(vocab[c][:V_KO])}; rows = [idx[w] for w in words]
        fam = {w: j for j, qs in enumerate(analogies[c].values()) for t in qs for w in t}
        colors = np.array([fam[w] for w in words])
        codes = {"M embedding": embedding(c, "linear", V_KO, "m", 1000)[rows], "log M embedding": embedding(c, "linear", V_KO, "logm", 1000)[rows],
                 "log-cond embedding": embedding(c, "linear", V_KO, "logcond", 1000)[rows], "block code": block_code(c, "linear", words)}
        G = gemma_vocab[GEMMA_ANALOGY_LAYER][[widx[w] for w in words]]
        fig = plot_utils.plot_embedding_panels(G, codes, None, cmap="tab20", colors=colors, scores={k: scores(E, G) for k, E in codes.items()},
                                               first_name=f"Gemma L{GEMMA_ANALOGY_LAYER}", starred="rest", gram_percentile=99)
        fig.suptitle(f"analogy words coloured by family, {c}", y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"wyart_analogies_panel_{c}.png")


    analogy_panel(view_corpus.value)
    return


if __name__ == "__main__":
    app.run()
