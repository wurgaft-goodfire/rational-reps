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
    # Everyday predictions as remaining time, read by Llama-3.1-8B

    Griffiths & Tenenbaum's tasks, but asking for the **remaining** amount R = T − t instead of the total T.
    The posterior is p(R | t) ∝ p(t + R) / (t + R): there is no support cut-off, and the prior is evaluated
    at t + R, so it enters the t–R interaction. The prior family alone fixes the predicted code of t (the
    balanced factor of the row-centred log p(R | t)), up to scale:

    | Family | Domains | Prediction rule | Predicted code of t |
    |---|---|---|---|
    | Gaussian | life span, movie run time | T ≈ prior mean while t is small | a line, uniform in t |
    | power law | movie grosses | T = 2^(1/γ) t | one-dimensional in log t |
    | Erlang | pharaohs' reigns, House terms | R = β ln 2 for every t | a single point (collapse) |

    The total-T question asks the same thing from the same prefix. It contrasts with the remaining-time
    question: it shares the prompt up to the question, but its answer moves with t in every family.

    ```
    Each question below gives one piece of information ... Give your best guess as a single number.
    A soccer match has been going for 30 minutes. How many more minutes will it last? Answer: 60
    ... (five worked examples from held-out domains with mixed rules, asked the same way as the question)
    A pharaoh has been on the throne for 12 years. How many more years will his reign last? Years left: ␣
    ```

    The examples matter for the base model: zero-shot, about 10% of its answer mass goes to t itself and 19%
    of its total-question mass to totals below t. The total question gets the same situations answered as totals.

    Reads: the residual at the final token (the space before the answer) and at t's token, every layer, plus
    the next-token distribution over "0".."999" (all single tokens). Two paraphrases per question, averaged.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    from pathlib import Path
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.everyday import (DOMAINS, PREFIX, encode, fit_prior, log_prior, number_tokens, prior_prompts,
                                   prompts, remaining_logcond)
    from src.code.llm import read_prompts
    from src.code.geometry import predicted_geometry
    from src.code.analysis_utils import rsa
    from src.code.plot_utils import save_figure

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    FAMILY_COLOR = {"gaussian": "tab:blue", "power": "tab:orange", "erlang": "tab:green"}
    return (
        CACHE,
        DOMAINS,
        FAMILY_COLOR,
        PREFIX,
        FIGURES,
        encode,
        fit_prior,
        log_prior,
        np,
        number_tokens,
        plt,
        predicted_geometry,
        prior_prompts,
        prompts,
        read_prompts,
        remaining_logcond,
        rsa,
        save_figure,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The predictions

    Each domain's predicted code of t, on its top principal component against t. The Erlang codes are one
    point, so their PC is flat at zero.
    """)
    return


@app.cell
def _(
    DOMAINS,
    FAMILY_COLOR,
    FIGURES,
    np,
    plt,
    predicted_geometry,
    remaining_logcond,
    save_figure,
):
    KERNELS = {d: remaining_logcond(spec) for d, spec in DOMAINS.items()}          # (t, R, L)
    PRED = {d: predicted_geometry(L) for d, (t, R, L) in KERNELS.items()}


    def pc1(E):
        E = E - E.mean(0)
        U, S, _ = np.linalg.svd(E, full_matrices=False)
        return U[:, 0] * S[0] * np.sign(U[-1, 0] - U[0, 0] + 1e-12)


    _fig, _axes = plt.subplots(1, len(DOMAINS), figsize=(3 * len(DOMAINS), 2.6))
    for _ax, (_d, _E) in zip(_axes, PRED.items()):
        _ax.plot(KERNELS[_d][0], pc1(_E), color=FAMILY_COLOR[DOMAINS[_d]["family"]])
        _ax.set_title(f"{_d} ({DOMAINS[_d]['family']})", fontsize=10); _ax.set_xlabel("t")
    _axes[0].set_ylabel("predicted code, PC 1")
    _fig.tight_layout()
    save_figure(_fig, FIGURES / "everyday_predictions.png")
    return KERNELS, PRED


@app.cell
def _(CACHE, DOMAINS, PREFIX, encode, np, number_tokens, prior_prompts, prompts, read_prompts, torch):
    LLM_PATH, LLM_NAME = "meta-llama/Llama-3.1-8B", "llama-3.1-8b"
    QUESTIONS = ("remaining", "total")
    NUMBER = PREFIX["remaining"] + "Here is a number: {t}. Write the number again: "   # control: the same numbers, no domain
    READ_DIR = CACHE / "everyday" / f"{LLM_NAME}_shots-everyday"; READ_DIR.mkdir(parents=True, exist_ok=True)


    def jobs():
        """{file stem: [(t, text)]} for every domain x question x paraphrase, the number control per domain, and
        the t-free prior questions (t = -1)."""
        out = {}
        for d in DOMAINS:
            for q in QUESTIONS:
                for k, rows in enumerate(prompts(d, q)):
                    out[f"{d}_{q}_{k}"] = rows
            out[f"{d}_number"] = [(t, NUMBER.format(t=t)) for t in DOMAINS[d]["t"]]
            for k, text in enumerate(prior_prompts(d)):
                out[f"{d}_prior_{k}"] = [(-1, text)]
        return out


    def row(tok, t, text):
        """``(ids, at)``; a t-free prompt is read at its last token."""
        if t >= 0:
            return encode(tok, text)
        ids = [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)
        return ids, len(ids) - 1


    _jobs = jobs()
    _missing = [s for s in _jobs if not (READ_DIR / f"{s}.npz").exists()]
    if _missing:                                                     # the expensive step, cached per prompt set
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(LLM_PATH)
        _model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="cuda").eval()
        _nums = number_tokens(_tok)
        for _s in _missing:
            _out = read_prompts(_model, [row(_tok, t, text) for t, text in _jobs[_s]], _nums, pad_id=_tok.eos_token_id)
            np.savez(READ_DIR / f"{_s}.npz", t=np.array([t for t, _ in _jobs[_s]]), **_out)
        del _model; torch.cuda.empty_cache()


    def load(d, q):
        """Reads for a domain and question, averaged over paraphrases: last, at_t (layers, n_t, d), q (n_t, 1000)
        (the mean answer distribution). ``q='number'`` gives the control."""
        files = sorted(READ_DIR.glob(f"{d}_{q}_*.npz")) if q != "number" else [READ_DIR / f"{d}_number.npz"]
        zs = [np.load(f) for f in files]
        return {"t": zs[0]["t"], "last": np.mean([z["last"] for z in zs], 0), "at_t": np.mean([z["at_t"] for z in zs], 0),
                "q": np.mean([np.exp(z["logq"]) for z in zs], 0)}


    READS = {d: {q: load(d, q) for q in (*QUESTIONS, "number")} for d in DOMAINS}
    PRIOR_Q = {d: np.mean([np.exp(np.load(f)["logq"][0]) for f in sorted(READ_DIR.glob(f"{d}_prior_*.npz"))], 0)
               for d in DOMAINS}                                                       # (1000,) over totals 0..999
    print({d: READS[d]["remaining"]["last"].shape for d in DOMAINS})
    return PRIOR_Q, READS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Llama's own prior

    The t-free questions ("Think of a pharaoh of ancient Egypt. How many years did his reign last in total?")
    give Llama's distribution over totals, averaged over two paraphrases. Its raw answers heap on round numbers,
    which a log prior would carry straight into the kernel, so each family is fit to it by weighted maximum
    likelihood (as G&T did with people's subjective priors) and the best fit is the second prediction: Bayes with
    Llama's prior. Gray: the elicited distribution; color: the best fit; black: the world prior.
    """)
    return


@app.cell
def _(DOMAINS, FAMILY_COLOR, FIGURES, PRIOR_Q, fit_prior, log_prior, np, plt, predicted_geometry,
      remaining_logcond, save_figure):
    LLAMA_SPEC = {}
    _fig, _axes = plt.subplots(1, len(DOMAINS), figsize=(3 * len(DOMAINS), 2.6))
    _T = np.arange(1, 1000)
    for _ax, _d in zip(_axes, DOMAINS):
        _best, _fits = fit_prior(PRIOR_Q[_d][1:])
        LLAMA_SPEC[_d] = {**DOMAINS[_d], "family": _best, "prior": _fits[_best]["prior"]}
        print(f"{_d:9s} best {_best:8s} {_fits[_best]['prior']}  "
              + "  ".join(f"loglik {f} {v['loglik']:.3f}" for f, v in _fits.items()))
        _hi = int(np.searchsorted(np.cumsum(PRIOR_Q[_d][1:]) / PRIOR_Q[_d][1:].sum(), 0.99)) + 10
        _ax.bar(_T[:_hi], PRIOR_Q[_d][1:_hi + 1] / PRIOR_Q[_d][1:].sum(), color="0.8", width=1)
        for _spec, _style in ((LLAMA_SPEC[_d], dict(color=FAMILY_COLOR[_best])), (DOMAINS[_d], dict(color="k", lw=1))):
            _lp = log_prior(_spec, _T); _p = np.exp(_lp - _lp.max()); _ax.plot(_T[:_hi], (_p / _p.sum())[:_hi], **_style)
        _ax.set_title(f"{_d}: Llama {_best}", fontsize=10); _ax.set_xlabel("total T")
    _fig.tight_layout()
    KERNELS_LLAMA = {d: remaining_logcond(s) for d, s in LLAMA_SPEC.items()}
    PRED_LLAMA = {d: predicted_geometry(L) for d, (t, R, L) in KERNELS_LLAMA.items()}
    save_figure(_fig, FIGURES / "everyday_llama_prior.png")
    return KERNELS_LLAMA, LLAMA_SPEC, PRED_LLAMA


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Behavior

    Median of Llama's answer distribution against t: remaining-time question (solid) and the total question
    minus t (dashed), against the Bayes median from the world prior (black). Priors: G&T's fits (life span, grosses,
    pharaohs) or refits from their sources (run time, House; `fit_everyday_priors.py`).
    """)
    return


@app.cell
def _(DOMAINS, FAMILY_COLOR, FIGURES, KERNELS, KERNELS_LLAMA, READS, np, plt, save_figure):
    def median(P, grid):
        P = P / P.sum(1, keepdims=True)
        return grid[(np.cumsum(P, 1) < 0.5).sum(1)]


    _fig, _axes = plt.subplots(1, len(DOMAINS), figsize=(3 * len(DOMAINS), 2.8))
    _grid = np.arange(1000)
    for _ax, _d in zip(_axes, DOMAINS):
        _t, _R, _L = KERNELS[_d]
        _c = FAMILY_COLOR[DOMAINS[_d]["family"]]
        _ax.plot(_t, median(np.exp(_L), _R), color="k", lw=1, label="Bayes, world prior")
        _ax.plot(_t, median(np.exp(KERNELS_LLAMA[_d][2]), _R), color="0.5", lw=1.5, ls="-.", label="Bayes, Llama's prior")
        _ax.plot(_t, median(READS[_d]["remaining"]["q"], _grid), color=_c, label="Llama: remaining")
        _ax.plot(_t, median(READS[_d]["total"]["q"], _grid) - _t, "--", color=_c, label="Llama: total − t")
        _ax.set_title(_d, fontsize=10); _ax.set_xlabel("t")
    _axes[0].set_ylabel("median remaining R"); _axes[0].legend(frameon=False, fontsize=7)
    _fig.tight_layout()
    save_figure(_fig, FIGURES / "everyday_behavior.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Geometry by layer

    For each domain and layer, the RSA (Spearman correlation of pairwise distances over t) of Llama's code of t
    with: the predicted code (Erlang predictions are a point, so their RSA is undefined and not drawn); a line
    in t; a line in log t; and the same numbers in the domain-free control prompt. Solid: the final token of
    the remaining-time question. Dotted: t's own token.
    """)
    return


@app.cell
def _(DOMAINS, FIGURES, LLAMA_SPEC, PRED, PRED_LLAMA, READS, np, plt, rsa, save_figure):
    def layer_rsa(X, Y):
        """RSA of each layer of ``X`` (layers, n, d) with ``Y`` (n, k) or a matching (layers, n, d)."""
        return np.array([rsa(Xl, Y if Y.ndim == 2 else Y[l]) for l, Xl in enumerate(X)])


    SCORES = {}
    for _d in DOMAINS:
        _r = READS[_d]; _t = _r["remaining"]["t"].astype(float)[:, None]
        SCORES[_d] = {"predicted": layer_rsa(_r["remaining"]["last"], PRED[_d]) if DOMAINS[_d]["family"] != "erlang" else None,
                      "predicted, Llama's prior": (layer_rsa(_r["remaining"]["last"], PRED_LLAMA[_d])
                                                   if LLAMA_SPEC[_d]["family"] != "erlang" else None),
                      "t": layer_rsa(_r["remaining"]["last"], _t), "log t": layer_rsa(_r["remaining"]["last"], np.log(_t)),
                      "number control": layer_rsa(_r["remaining"]["last"], _r["number"]["last"]),
                      "log t (t's token)": layer_rsa(_r["remaining"]["at_t"], np.log(_t))}

    _fig, _axes = plt.subplots(1, len(DOMAINS), figsize=(3 * len(DOMAINS), 2.8), sharey=True)
    _style = {"predicted": ("k", "-"), "predicted, Llama's prior": ("0.5", "-."), "t": ("tab:blue", "-"), "log t": ("tab:orange", "-"),
              "number control": ("0.6", "-"), "log t (t's token)": ("tab:orange", ":")}
    for _ax, _d in zip(_axes, DOMAINS):
        for _k, _s in SCORES[_d].items():
            if _s is not None:
                _ax.plot(_s, color=_style[_k][0], ls=_style[_k][1], label=_k)
        _ax.set_title(f"{_d} ({DOMAINS[_d]['family']})", fontsize=10); _ax.set_xlabel("layer"); _ax.set_ylim(-0.2, 1)
    _axes[0].set_ylabel("RSA"); _axes[0].legend(frameon=False, fontsize=7)
    _fig.tight_layout()
    save_figure(_fig, FIGURES / "everyday_rsa_by_layer.png")
    return (SCORES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Collapse

    How much the final-token code varies with t under the remaining-time question, relative to the total
    question on the same prefix: the ratio of their spreads (mean squared distance to the centroid over t), by
    layer. Prediction: the Erlang domains fall well below the others, since there R carries nothing about t
    while T moves with it. The same ratio for Llama's answer distributions is
    printed per domain, in Hellinger space (square-root probabilities over 0..999, which weights each answer by its
    probability): the behavioral version of the collapse.
    """)
    return


@app.cell
def _(DOMAINS, FAMILY_COLOR, FIGURES, READS, np, plt, save_figure):
    def spread(X):
        """Mean squared distance to the centroid over the item axis (-2) of ``X``."""
        return ((X - X.mean(-2, keepdims=True)) ** 2).sum(-1).mean(-1)


    RATIO = {d: spread(READS[d]["remaining"]["last"]) / spread(READS[d]["total"]["last"]) for d in DOMAINS}
    _fig, _ax = plt.subplots(figsize=(5, 3.2))
    for _d, _r in RATIO.items():
        _f = DOMAINS[_d]["family"]
        _ax.plot(_r, color=FAMILY_COLOR[_f], ls="-" if _d in ("lifespan", "grosses", "pharaohs") else "--", label=f"{_d} ({_f})")
        _h = {q: np.sqrt(READS[_d][q]["q"]) for q in ("remaining", "total")}
        print(f"{_d:9s} answer-distribution (Hellinger) spread ratio {spread(_h['remaining']) / spread(_h['total']):.3f}")
    _ax.axhline(1, color="0.7", lw=0.8); _ax.set_xlabel("layer"); _ax.set_ylabel("spread: remaining / total")
    _ax.legend(frameon=False, fontsize=7)
    _fig.tight_layout()
    save_figure(_fig, FIGURES / "everyday_collapse.png")
    return


if __name__ == "__main__":
    app.run()
