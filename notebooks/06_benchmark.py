import scanpy as sc
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gseapy as gp
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from pathlib import Path

SIGS   = Path("signatures/ifn_modules_v1.json")
OUTDIR = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Load already-scored A549 dataset ─────────────────────────────────────────
print("Loading A549 scored dataset...")
adata = sc.read_h5ad("data/processed/A549_TGFB1_IFN_scored.h5ad")
print(f"  {adata.shape[0]} cells, EMT states: {adata.obs['EMT_state'].value_counts().to_dict()}")

with open(SIGS) as f:
    sig_data = json.load(f)

cfg = sig_data["scoring_config"]["weights"]

# ── Method 2: ssGSEA via gseapy ──────────────────────────────────────────────
print("Method 2: ssGSEA (takes 3-5 min)...")
gene_sets = {mod_id: mod["genes"] for mod_id, mod in sig_data["modules"].items()}

expr_df = pd.DataFrame(
    adata.raw.X if adata.raw else adata.X,
    index=adata.obs_names,
    columns=adata.raw.var_names if adata.raw else adata.var_names
).T

ss = gp.ssgsea(
    data=expr_df,
    gene_sets=gene_sets,
    sample_norm_method="rank",
    no_plot=True,
    threads=4,
    min_size=5,
    verbose=False,
)

ssgsea_scores = ss.res2d.pivot(index="Term", columns="Name", values="NES").T
ssgsea_scores.columns = [f"ssgsea_{c}" for c in ssgsea_scores.columns]
print(f"  ssGSEA terms found: {ss.res2d['Term'].unique().tolist()}")

w_ss = pd.Series({f"ssgsea_{k}": v for k, v in cfg.items()})
for c in w_ss.index:
    if c not in ssgsea_scores.columns:
        print(f"  WARNING: {c} missing from ssGSEA — filling with 0")
        ssgsea_scores[c] = 0.0

raw_ss2 = ssgsea_scores[w_ss.index].multiply(w_ss.values, axis=1).sum(axis=1)
ssgsea_ifn = (raw_ss2 - raw_ss2.min()) / (raw_ss2.max() - raw_ss2.min())
adata.obs["ssGSEA_IFN_SS"] = ssgsea_ifn.reindex(adata.obs_names).values
print("  ssGSEA done")

# ── Method 3: Mean module expression (naive baseline) ────────────────────────
print("Method 3: Mean expression baseline...")
for mod_id, mod in sig_data["modules"].items():
    genes = [g for g in mod["genes"] if g in adata.raw.var_names]
    gene_idx = [list(adata.raw.var_names).index(g) for g in genes]
    raw_mat = adata.raw.X[:, gene_idx]
    if hasattr(raw_mat, "toarray"):
        raw_mat = raw_mat.toarray()
    adata.obs[f"mean_{mod_id}"] = raw_mat.mean(axis=1)

w_mean = pd.Series({f"mean_{k}": v for k, v in cfg.items()})
raw_mean = adata.obs[w_mean.index].multiply(w_mean.values, axis=1).sum(axis=1)
mean_ifn = (raw_mean - raw_mean.min()) / (raw_mean.max() - raw_mean.min())
adata.obs["Mean_IFN_SS"] = mean_ifn.values
print("  Mean expression done")

# ── Method 4: Single best module (M2 chronic) ────────────────────────────────
m2 = adata.obs["score_M2_chronic_IFN_TME"]
adata.obs["Single_M2"] = (m2 - m2.min()) / (m2.max() - m2.min())
print("  Single module (M2) done")

# ── Evaluate all methods ──────────────────────────────────────────────────────
methods = {
    "IFN-SS (ours)":    "IFN_SS",
    "ssGSEA + weights": "ssGSEA_IFN_SS",
    "Mean expression":  "Mean_IFN_SS",
    "Single module M2": "Single_M2",
}

epi_mes = adata.obs[adata.obs["EMT_state"].isin(["Epithelial", "Mesenchymal"])].copy()
y_true  = (epi_mes["EMT_state"] == "Epithelial").astype(int)

print("\n── Benchmark results ──────────────────────────────────────────────")
print(f"{'Method':<25} {'AUC':>6}  {'Cohen d':>8}  {'MWU p':>12}")
print("-" * 58)

results = {}
for name, col in methods.items():
    subset   = epi_mes[[col, "EMT_state"]].dropna()
    scores   = subset[col].astype(float)
    y        = (subset["EMT_state"] == "Epithelial").astype(int)
    auc      = roc_auc_score(y, scores)
    epi_s    = subset[subset["EMT_state"] == "Epithelial"][col].astype(float)
    mes_s    = subset[subset["EMT_state"] == "Mesenchymal"][col].astype(float)
    pooled   = np.sqrt((epi_s.std()**2 + mes_s.std()**2) / 2)
    cohend   = (epi_s.mean() - mes_s.mean()) / pooled
    _, pval  = mannwhitneyu(epi_s.values, mes_s.values, alternative="two-sided")
    results[name] = {"AUC": auc, "Cohen_d": cohend, "p": pval}
    print(f"{name:<25} {auc:>6.3f}  {cohend:>8.3f}  {pval:>12.2e}")
    
# ── Plot ──────────────────────────────────────────────────────────────────────
state_colors = {
    "Epithelial": "#2196F3",
    "Hybrid E/M": "#9C27B0",
    "Mesenchymal": "#FF5722"
}
state_order = ["Epithelial", "Hybrid E/M", "Mesenchymal"]

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

for idx, (name, col) in enumerate(methods.items()):
    ax      = fig.add_subplot(gs[idx // 2, idx % 2])
    bp_data = [adata.obs[adata.obs["EMT_state"] == s][col].values
               for s in state_order]
    bp = ax.boxplot(bp_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))
    for patch, state in zip(bp["boxes"], state_order):
        patch.set_facecolor(state_colors[state])
        patch.set_alpha(0.8)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(state_order, fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    r = results[name]
    ax.set_title(
        f"{name}\nAUC={r['AUC']:.3f}  Cohen d={r['Cohen_d']:.3f}  p={r['p']:.1e}",
        fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

plt.suptitle(
    "InterferonAtlas IFN-SS vs existing approaches\n"
    "(A549 TGF-β1 EMT — Epithelial vs Mesenchymal separation)",
    fontsize=11, y=1.01)
plt.savefig(OUTDIR / "fig4_benchmark.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/figures/fig4_benchmark.png")
plt.close()

pd.DataFrame(results).T.to_csv("results/tables/benchmark_results.csv")
print("Saved: results/tables/benchmark_results.csv")
print("Done.")