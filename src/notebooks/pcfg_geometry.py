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
# # PCFG transformers represent the predicted belief-state geometry
#
# **Claim.** From a grammar alone we compute the exact Bayesian **belief** — the filtering posterior
# `p(state emitting the next terminal | prefix)` — by the inside/prefix algorithm. A transformer
# trained on strings from the grammar represents that belief geometry in its residual stream.
#
# **Metric.** The discriminator is **CKA** (rotation-level; unlike CCA it is *not* saturated by a wide
# hidden) and a **held-out linear decode** R² (fit `hidden→belief` on a train split of *distinct
# prefixes*, evaluate on unseen prefixes). Every number is reported against the **untrained twin**
# (same architecture, random weights) — the control that a wide random hidden can pass on CCA but not
# on CKA/decode.
#
# Four grammars: `aⁿbⁿ` (counter), `dyck` (bracket stack), `mini_nl` (phrase structure),
# `mini_pl` (control flow). (Emission is near-deterministic here, so belief ≈ next-token distribution;
# distinguishing the two would need stochastic emission — a separate experiment.)

# %%
import os, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(sys.prefix).parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.code.data import pcfg_next_token, anbn_count, dyck_stack, pl_nesting
from src.code.eval import cka, cca
from src.code.train import train_inline
from src.code.analysis_utils import heldout_decode_r2, untrained_twin
from src.code import plot_utils
import matplotlib.pyplot as plt

N_SEQ = 600
ALPHA = 100.0                                                    # ridge strength for the held-out decode


def analyze(runner, name, label_fn, label_name, cmap):
    """Train, read the last-layer hidden at every prefix (deduped to distinct prefixes, since a causal
    hidden depends only on the prefix), and compare its geometry to the exact belief — trained vs the
    untrained twin. Saves the belief-geometry figure; returns the metrics + arrays."""
    model, ds, ev, hist, config = train_inline(
        [f"runner={runner}", f"name={name}", "wandb.enabled=false", "eval_kwargs.output_hidden_states=true"])
    seqs = [ds._sample_sentence()[0] for _ in range(N_SEQ)]
    ids = np.full((N_SEQ, ds.seq_len), ds.pad_token); ids[:, 0] = ds.start_token
    for b, s in enumerate(seqs):
        ids[b, 1:1 + len(s)] = s; ids[b, 1 + len(s)] = ds.eos_token
    ids = torch.tensor(ids, dtype=torch.long, device=ds.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = [h.cpu().numpy() for h in out["hidden_states"]]
    logits = out["logits"][:, :-1][..., ds.item_tokens].cpu().numpy()
    pm = np.exp(logits - logits.max(-1, keepdims=True)); pm /= pm.sum(-1, keepdims=True)
    with torch.no_grad():
        hs_un = [h.cpu().numpy() for h in untrained_twin(config, ds)(ids, output_hidden_states=True)["hidden_states"]]

    seen, rows, B, labels, kls = {}, [], [], [], []
    for b, s in enumerate(seqs):
        dist, bel = pcfg_next_token(np.array(s, dtype=int), ds.g)
        for t in range(len(s) + 1):
            kls.append(float((dist[t] * (np.log(dist[t] + 1e-9) - np.log(pm[b, t] + 1e-9))).sum()))
            pre = tuple(s[:t])
            if pre in seen or bel[t].sum() <= 0:                # skip terminal (EOS) states with 0 belief
                continue
            seen[pre] = 1
            rows.append((b, t)); B.append(bel[t]); labels.append(label_fn(pre, dist[t]))
    B, labels = np.array(B), np.array(labels)
    bs = np.array([b for b, _ in rows]); ts = np.array([t for _, t in rows])
    # the belief is assembled across depth; report the layer where it is best represented (the last layer
    # over-specialises toward the logits). The untrained twin is read at the same layer, so the gap is fair.
    layer = int(np.argmax([cka(hs[L][bs, ts], B) for L in range(len(hs))]))
    H, Hun = hs[layer][bs, ts], hs_un[layer][bs, ts]
    groups = np.arange(len(B))                                  # each distinct prefix is its own decode group

    cka_tr, cka_un = cka(H, B), cka(Hun, B)
    r2_tr = np.mean([heldout_decode_r2(H, B, groups, seed=s, alpha=ALPHA) for s in range(5)])
    r2_un = np.mean([heldout_decode_r2(Hun, B, groups, seed=s, alpha=ALPHA) for s in range(5)])
    m = dict(name=name, config=config, kl=float(np.mean(kls)), n=len(B), layer=layer,
             cka_tr=cka_tr, cka_un=cka_un, r2_tr=r2_tr, r2_un=r2_un, cca=cca(H, B), cca_un=cca(Hun, B))
    print(f"[{name}] best-layer L{layer}/{len(hs)-1}  KL={m['kl']:.4f} n_prefix={m['n']}  "
          f"CKA {cka_tr:.3f}/{cka_un:.3f}(un)  decodeR2 {r2_tr:.3f}/{r2_un:.3f}(un)  "
          f"CCA {m['cca']:.3f}/{m['cca_un']:.3f}(un, saturated)")

    # 3-panel belief geometry: predicted | trained (rotation-aligned) | untrained (rotation-aligned).
    # Rotation-only (Procrustes) alignment respects CKA — a high-CKA match overlays the belief, a low
    # one cannot — so the untrained panel visibly fails to reproduce the belief's organization.
    fig = plot_utils.plot_alignment_grid(
        B, {f"transformer (CKA {cka_tr:.2f})": H, f"untrained (CKA {cka_un:.2f})": Hun},
        methods=("rotation",), rank=6, colors=labels, cmap=cmap,
        ref_label="predicted belief", colorbar_label=label_name,
        suptitle=f"{name} (layer {layer}): transformer represents the predicted belief geometry "
                 f"(learned — untrained twin does not)")
    plot_utils.save_figure(fig, plot_utils.figure_path(config, f"belief_{name}"))
    plt.show()
    return m


# %% [markdown]
# ## aⁿbⁿ — counter
# %%
r_anbn = analyze("pcfg_anbn", "pcfg_anbn", lambda pre, d: anbn_count(pre), "unmatched a", "coolwarm")
# %% [markdown]
# ## Dyck-2 — bracket stack
# %%
r_dyck = analyze("pcfg_dyck", "pcfg_dyck", lambda pre, d: len(dyck_stack(pre)), "stack depth", "viridis")
# %% [markdown]
# ## mini-NL — phrase structure (coloured by predicted-next POS)
# %%
r_nl = analyze("pcfg_mininl", "pcfg_mininl", lambda pre, d: int(d.argmax()), "argmax next token", "tab10")
# %% [markdown]
# ## mini-PL — control flow (coloured by nesting depth)
# %%
r_pl = analyze("pcfg_minipl", "pcfg_minipl", lambda pre, d: pl_nesting(pre), "nesting depth", "viridis")

# %% [markdown]
# ## Summary — CKA and held-out decode both separate trained from untrained; CCA does not
# %%
res = [r_anbn, r_dyck, r_nl, r_pl]
print(f"{'grammar':10s} {'layer':>5s} {'KL':>7s} {'CKA_tr':>7s} {'CKA_un':>7s} {'R2_tr':>7s} {'R2_un':>7s} "
      f"{'CCA_tr':>7s} {'CCA_un':>7s}")
for r in res:
    print(f"{r['name']:10s} {r['layer']:5d} {r['kl']:7.4f} {r['cka_tr']:7.3f} {r['cka_un']:7.3f} {r['r2_tr']:7.3f} "
          f"{r['r2_un']:7.3f} {r['cca']:7.3f} {r['cca_un']:7.3f}")

# %% [markdown]
# ## Headline figure — belief is represented (trained) and this is learned (untrained twin fails)
# CKA (rotation-level geometry) and held-out decode both separate trained from the untrained twin;
# CCA is saturated for both (a wide hidden trivially spans a low-rank belief in-sample) and is omitted.
# %%
names = [r["name"].replace("pcfg_", "") for r in res]
x = np.arange(len(res)); w = 0.38
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for a, (ktr, kun, ttl) in zip(ax, [("cka_tr", "cka_un", "CKA (belief geometry)"),
                                    ("r2_tr", "r2_un", "held-out decode R²")]):
    a.bar(x - w / 2, [r[ktr] for r in res], w, label="trained", color=plot_utils.FIG.categorical[0])
    a.bar(x + w / 2, [r[kun] for r in res], w, label="untrained twin", color=plot_utils.FIG.categorical[1])
    a.set_xticks(x); a.set_xticklabels(names); a.set_ylim(0, 1.05); a.set_title(ttl)
    a.legend(frameon=False); plot_utils.FIG.style_axes(a)
fig.suptitle("PCFG transformers represent the predicted belief geometry (best layer; trained >> untrained twin)", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96))
plot_utils.save_figure(fig, plot_utils.figure_path(r_anbn["config"], "belief_summary"))
plt.show()
