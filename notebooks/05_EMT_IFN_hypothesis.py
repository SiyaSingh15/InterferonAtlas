import scanpy as sc
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kruskal, spearmanr
from pathlib import Path

SIGS    = Path("signatures/ifn_modules_v1.json")
OUTDIR  = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading A549 TGF-β1 time course...")
meta = pd.read_csv("data/raw/A549_TGFB1_metadata.csv.gz", index_col=0)
print(f"  {meta.shape[0]} cells")
print(f"  Timepoints: {sorted(meta['Time'].unique())}")
print(f"  Pseudotime range: {meta['Pseudotime'].min():.3f} - {meta['Pseudotime'].max():.3f}")

# Load UMI matrix (genes x cells) — transpose to cells x genes
print("Loading UMI matrix (this takes ~1 min)...")
mat = pd.read_csv("data/raw/A549_TGFB1_matrix.csv.gz",
                  index_col=0, on_bad_lines="skip")
print(f"  Matrix shape (genes x cells): {mat.shape}")
mat = mat.T
print(f"  Transposed: {mat.shape}")

# ── Build AnnData ─────────────────────────────────────────────────────────────
# Keep only singlets
keep = meta[meta["Doublet"] == "Singlet"].index
meta = meta.loc[keep]
mat  = mat.loc[mat.index.intersection(keep)]
print(f"  After singlet filter: {mat.shape[0]} cells")

adata = sc.AnnData(X=mat.values, obs=meta.loc[mat.index], var=pd.DataFrame(index=mat.columns))
adata.obs_names = mat.index
adata.var_names = mat.columns

# ── Preprocess ────────────────────────────────────────────────────────────────
print("Preprocessing...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)
print(f"  UMAP computed")

# ── Score IFN modules ─────────────────────────────────────────────────────────
print("Scoring IFN modules...")
with open(SIGS) as f:
    sig_data = json.load(f)

for mod_id, mod in sig_data["modules"].items():
    genes_present = [g for g in mod["genes"] if g in adata.raw.var_names]
    sc.tl.score_genes(adata, gene_list=genes_present, score_name=f"score_{mod_id}")
    print(f"  {mod['label']}: {len(genes_present)}/{len(mod['genes'])} genes")

cfg = sig_data["scoring_config"]["weights"]
w   = pd.Series({f"score_{k}": v for k, v in cfg.items()})
scores = adata.obs[[f"score_{k}" for k in cfg.keys()]]
raw_ss = scores.multiply(w.values, axis=1).sum(axis=1)
ifn_ss = (raw_ss - raw_ss.min()) / (raw_ss.max() - raw_ss.min())
adata.obs["IFN_SS"] = ifn_ss.values

# ── Define EMT states from timepoints ────────────────────────────────────────
# 0d = Epithelial, early days = Hybrid E/M, late days = Mesenchymal
# Define EMT states from pseudotime tertiles — more balanced than timepoints
pt = adata.obs["Pseudotime"]
t33, t66 = pt.quantile(0.33), pt.quantile(0.66)
adata.obs["EMT_state"] = pd.cut(pt,
    bins=[-0.001, t33, t66, 1.001],
    labels=["Epithelial", "Hybrid E/M", "Mesenchymal"])
print(f"\nEMT state counts (pseudotime tertiles):\n{adata.obs['EMT_state'].value_counts()}")


# ── Statistics ────────────────────────────────────────────────────────────────
groups = [adata.obs[adata.obs["EMT_state"]==s]["IFN_SS"].values
          for s in ["Epithelial", "Hybrid E/M", "Mesenchymal"]]
stat, pval = kruskal(*groups)
rho, spval = spearmanr(adata.obs["Pseudotime"], adata.obs["IFN_SS"])
print(f"\nKruskal-Wallis across EMT states: H={stat:.2f}, p={pval:.2e}")
print(f"Spearman IFN-SS ~ Pseudotime: rho={rho:.3f}, p={spval:.2e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
state_colors = {"Epithelial": "#2196F3", "Hybrid E/M": "#9C27B0", "Mesenchymal": "#FF5722"}
adata.obs["EMT_state"] = pd.Categorical(adata.obs["EMT_state"],
    categories=["Epithelial", "Hybrid E/M", "Mesenchymal"])

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig)

# UMAP colored by EMT state
ax1 = fig.add_subplot(gs[0, 0])
sc.pl.umap(adata, color="EMT_state", ax=ax1, show=False,
           title="EMT state (TGF-β1 time course)",
           palette=state_colors)

# UMAP colored by IFN-SS
ax2 = fig.add_subplot(gs[0, 1])
sc.pl.umap(adata, color="IFN_SS", color_map="RdBu_r", vmin=0, vmax=1,
           ax=ax2, show=False, title="IFN Spectrum Score")

# UMAP colored by Pseudotime
ax3 = fig.add_subplot(gs[0, 2])
sc.pl.umap(adata, color="Pseudotime", color_map="viridis",
           ax=ax3, show=False, title="EMT Pseudotime")

# Boxplot: IFN-SS by EMT state — THE KEY FIGURE
ax4 = fig.add_subplot(gs[1, 0:2])
state_order = ["Epithelial", "Hybrid E/M", "Mesenchymal"]
bp_data = [adata.obs[adata.obs["EMT_state"]==s]["IFN_SS"].values for s in state_order]
bp = ax4.boxplot(bp_data, patch_artist=True, widths=0.5,
                 medianprops=dict(color="white", linewidth=2))
for patch, state in zip(bp["boxes"], state_order):
    patch.set_facecolor(state_colors[state])
    patch.set_alpha(0.8)
ax4.set_xticks([1, 2, 3])
ax4.set_xticklabels(state_order, fontsize=12)
ax4.set_ylabel("IFN Spectrum Score", fontsize=12)
ax4.set_title(f"IFN-SS across EMT states  (Kruskal-Wallis p={pval:.2e})", fontsize=12)
ax4.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# Scatter: IFN-SS vs Pseudotime
ax5 = fig.add_subplot(gs[1, 2])
colors = adata.obs["EMT_state"].map(state_colors)
ax5.scatter(adata.obs["Pseudotime"], adata.obs["IFN_SS"],
            c=colors, alpha=0.3, s=5, rasterized=True)
ax5.set_xlabel("EMT Pseudotime", fontsize=11)
ax5.set_ylabel("IFN Spectrum Score", fontsize=11)
ax5.set_title(f"IFN-SS ~ Pseudotime  (ρ={rho:.3f}, p={spval:.2e})", fontsize=11)

plt.tight_layout()
plt.savefig(OUTDIR / "fig3_EMT_IFN_hypothesis.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/figures/fig3_EMT_IFN_hypothesis.png")
plt.close()

adata.write_h5ad("data/processed/A549_TGFB1_IFN_scored.h5ad")
print("Saved: data/processed/A549_TGFB1_IFN_scored.h5ad")
print("\nDone.")