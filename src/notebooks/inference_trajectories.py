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
# # Inference trajectories learned by transformers
#
# This notebook turns Figure 7 of `concepts_analysis.py` into trainable experiments. In each
# environment a sequence first draws a hidden concept $z$ and then iid observations
# $x_i\sim p(x\mid z)$. We train two otherwise identical causal transformers on the same process:
# one predicts the hidden concept (`labels=latent`), the other predicts the next observation
# (`labels=tokens`). The comparison tests the claim in `notes/inference_derivation.md`:
# **one Bayes belief state, with target-adapted output coordinates.**
#
# The three panels reproduce all three inference worlds from the mathematical figure:
# noisy binary evidence, the temperature/count-evidence plane, and noisy weekday mentions.
# The temperature line is discretized to the figure's five displayed alternatives because a
# categorical latent-token loss necessarily requires a finite latent vocabulary.

# %%
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(sys.prefix).parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.analysis_utils import align_representations, heldout_decode_r2
from src.code.eval import cca, cka
from src.code.geometry import factor_logcond
from src.code.train import train_inline
from src.code import plot_utils
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Train six small models
#
# Every row below is one DGP and every column is one label interface. `INFERENCE_STEPS` can be
# lowered for a smoke run; the default is the experiment setting used for the saved results.

# %%
STEPS = int(os.getenv("INFERENCE_STEPS", "4000"))
CONTEXT = 40
CASES = {
    "binary": ["family=binary_concepts", "~family_kwargs", "+family_kwargs.p=0.65"],
    "temperature": ["family=temperature_concepts", "~family_kwargs",
                    "+family_kwargs.n_theta=5", "+family_kwargs.n_observations=5"],
    "week": ["family=von_mises", "family_kwargs.n_mu=7", "family_kwargs.kappas=[1.5]",
             "family_kwargs.x_n=7"],
}
COMMON = [
    "runner=exchangeable_transformer", f"training_kwargs.train_steps={STEPS}",
    f"training_kwargs.eval_steps={max(STEPS // 8, 1)}", "training_kwargs.batch_size=128",
    f"training_kwargs.context_length={CONTEXT}", "training_kwargs.learning_rate=0.002",
    "model_kwargs.model_config.hidden_size=32", "model_kwargs.model_config.num_attention_heads=2",
    "model_kwargs.model_config.num_hidden_layers=2", "model_kwargs.model_config.intermediate_size=128",
    "eval_kwargs.num_eval_batches=2",
]

runs = {}
for case, case_overrides in CASES.items():
    runs[case] = {}
    for labels in ("latent", "tokens"):
        runs[case][labels] = train_inline(
            COMMON + case_overrides + [f"labels={labels}"], name=f"inference_{case}_{labels}")

# %%
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5), sharey=True)
for a, case in zip(ax, CASES):
    for labels, color in (("latent", plot_utils.SERIES[2]), ("tokens", plot_utils.SERIES[1])):
        h = runs[case][labels][3]
        a.plot(h.step, h.loss, marker="o", ms=3, color=color, label=labels)
    a.set(title=case, xlabel="training step"); plot_utils.FIG.style_axes(a)
ax[0].set_ylabel("KL from exact Bayes target"); ax[0].legend(frameon=False)
fig.tight_layout()
plot_utils.save_figure(fig, plot_utils.figure_path(runs["binary"]["latent"][4], "training_curves"))

# %% [markdown]
# ## 2. Evaluate both targets on the same histories
#
# We reuse exactly the same sampled histories within each row. The latent code is centered
# log posterior over $z$; the token code is centered log posterior-predictive over $x$. Their
# softmaxes are the exact targets seen by the two models.

# %%
def centered_log(p):
    g = np.log(np.asarray(p) + 1e-12)
    return g - g.mean(-1, keepdims=True)

records, results = [], {}
B, L = 128, CONTEXT
for case in CASES:
    mz, dz, _, _, _ = runs[case]["latent"]
    mx, dx, _, _, _ = runs[case]["tokens"]
    dz.rng.manual_seed(7)
    theta, z = dz.sample_emission(B, return_labels=True)
    x = torch.multinomial(theta, L, replacement=True, generator=dz.rng)
    ids = torch.cat([torch.full((B, 1), dz.start_token, dtype=torch.long, device=dz.device), x], 1)

    belief = dz.optimal_predictions(ids).cpu().numpy()
    predictive = dx.optimal_predictions(ids).cpu().numpy()
    latent_code, token_code = centered_log(belief), centered_log(predictive)
    groups = np.repeat(np.arange(B), L)
    result = {"x": x.cpu().numpy(), "z": z.cpu().numpy(), "belief": belief,
              "latent_code": latent_code, "token_code": token_code}

    for labels, model, ds, target, code in (
        ("latent", mz, dz, belief, latent_code), ("tokens", mx, dx, predictive, token_code)):
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        H = out["hidden_states"][-1][:, :-1].reshape(B * L, -1).cpu().numpy()
        logits = out["logits"][:, :-1][..., ds.item_tokens].cpu().numpy()
        probs = torch.softmax(torch.tensor(logits), -1).numpy()
        learned_code = logits - logits.mean(-1, keepdims=True)
        kl = (target * (np.log(target + 1e-12) - np.log(probs + 1e-12))).sum(-1).mean()
        records.append({"case": case, "labels": labels, "KL(Bayes || model)": kl,
                        "belief probe R2": heldout_decode_r2(H, belief.reshape(B * L, -1), groups),
                        "target-code CCA": cca(H, code.reshape(B * L, -1)),
                        "target-code CKA": cka(learned_code.reshape(B * L, -1),
                                               code.reshape(B * L, -1))})
        result[labels] = {"H": H.reshape(B, L, -1), "code": learned_code}
    results[case] = result

scores = pd.DataFrame(records)
scores

# %% [markdown]
# The behavioral KL tests whether each model learned its own Bayes target. The held-out belief
# probe tests the shared-state claim without allowing prefixes from the same sequence into both
# splits. CCA tests whether the hidden state carries the target code up to a linear coordinate
# change; CKA compares the model's actual output logits to the exact target-adapted logits.

# %% [markdown]
# ## 3. The learned histories trace the three inference paths
#
# For display we use the coordinates of the original mathematical figure: posterior position on
# the binary chord; $(t,\sum_i\mu(x_i))$ for temperature; and cumulative circular evidence for
# weekdays. A held-out affine map puts each model's hidden state into those interpretable
# coordinates. Dots are exact inference, crosses/pluses are the two trained transformers.

# %%
def trajectory_coordinates(case, result, ds):
    x, belief = result["x"], result["belief"]
    if case == "binary":
        q = belief[..., 1] - belief[..., 0]
        return np.stack([q, np.zeros_like(q)], -1)
    if case == "temperature":
        mu = ds.info["observations"]
        evidence = np.concatenate([np.zeros((B, 1)), np.cumsum(mu[x], 1)[:, :-1]], 1)
        time = np.broadcast_to(np.arange(L), (B, L))
        return np.stack([time / L, evidence / L], -1)
    _, _, D, _, _ = factor_logcond(ds.atom_logcond)
    obs = D[:2, x].transpose(1, 2, 0)
    return np.concatenate([np.zeros((B, 1, 2)), np.cumsum(obs, 1)[:, :-1]], 1)

fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))
for a, case in zip(ax, CASES):
    ds = runs[case]["latent"][1]; r = results[case]
    coords = trajectory_coordinates(case, r, ds)
    mapped = {}
    for labels in ("latent", "tokens"):
        mapped[labels], _ = align_representations(r[labels]["H"].reshape(B * L, -1),
                                                   coords.reshape(B * L, 2), method="linear")
        mapped[labels] = mapped[labels].reshape(B, L, 2)
    seq = int(np.argmax(r["z"]))
    a.plot(*coords[seq].T, color="0.75", lw=1)
    a.scatter(*coords[seq].T, c=np.arange(L), cmap="viridis", s=34, label="exact")
    a.scatter(*mapped["latent"][seq].T, c=np.arange(L), cmap="viridis", marker="x", s=29,
              label="latent labels")
    a.scatter(*mapped["tokens"][seq].T, c=np.arange(L), cmap="viridis", marker="+", s=35,
              label="token labels")
    a.set_title(case); plot_utils.FIG.clean(a)
    if case != "temperature": a.set_aspect("equal")
ax[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("one exact belief path, recovered from two target-adapted models", y=1.01)
fig.tight_layout()
plot_utils.save_figure(fig, plot_utils.figure_path(runs["binary"]["latent"][4], "learned_trajectories"))

# %% [markdown]
# ## 4. The output coordinates depend on the target
#
# The upper row compares learned latent logits with centered log posterior; the lower row compares
# learned token logits with centered log posterior-predictive. These are two vector embeddings of
# the same histories, selected by the linear readout interface.

# %%
fig, ax = plt.subplots(2, 3, figsize=(12, 7.4))
for col, case in enumerate(CASES):
    r = results[case]
    for row, (labels, exact_name) in enumerate((("latent", "latent_code"), ("tokens", "token_code"))):
        exact = r[exact_name].reshape(B * L, -1)
        V = np.linalg.svd(exact - exact.mean(0), full_matrices=False)[2][:2]
        exact2 = (exact - exact.mean(0)) @ V.T
        learned, r2 = align_representations(r[labels]["code"].reshape(B * L, -1), exact2,
                                             method="linear")
        keep = np.arange(0, B * L, 8)
        ax[row, col].scatter(*exact2[keep].T, s=8, alpha=.35, color="0.65", label="exact")
        ax[row, col].scatter(*learned[keep].T, s=7, alpha=.35, color=plot_utils.SERIES[row + 1],
                             label="model")
        ax[row, col].set_title(f"{case}: {labels} logits\nR2={r2:.3f}")
        ax[row, col].set_aspect("equal"); plot_utils.FIG.clean(ax[row, col])
ax[0, 0].legend(frameon=False, fontsize=9)
fig.tight_layout()
plot_utils.save_figure(fig, plot_utils.figure_path(runs["binary"]["latent"][4], "target_codes"))

# %% [markdown]
# ## Reading the experiment
#
# Success has three layers: both models approach their own Bayes target; both hidden states permit
# held-out linear decoding of the same posterior belief; and their output logits match different
# exact codes—log posterior over concepts versus log posterior-predictive over observations. The
# trajectories are therefore properties of the DGP's belief state, while the final vector
# coordinates are properties of the requested label interface.

# %%
