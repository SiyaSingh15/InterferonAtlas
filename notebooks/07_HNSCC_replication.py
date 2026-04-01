import scanpy as sc
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from pathlib import Path

SIGS   = Path("signatures/ifn_modules_v1.json")
OUTDIR = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

DATA_HNSCC    = "/home/siyasingh15/cd4_til_project/data/processed/HNSCC_CD4_purified.h5ad"
DATA_MELANOMA = "data/processed/melanoma_CD4_final.h5ad"

with open(SIGS) as f:
    sig_data = json.load(f)

# ── Score HNSCC ───────────────────────────────────────────────────────────────
print("Loading HNSCC dataset...")
hnsc = sc.read_h5ad(DATA_HNSCC)
print(f"  {hnsc.shape[0]} cells x {hnsc.shape[1]} genes")

# UMAP exists but no clustering — compute neighbors + leiden
print("Computing neighbors and clustering...")
sc.pp.highly_variable_genes(hnsc, n_top_genes=2000, flavor="seurat")
hnsc.raw = hnsc
sc.pp.scale(hnsc, max_value=10)
sc.tl.pca(hnsc, n_comps=30)
sc.pp.neighbors(hnsc, n_pcs=30)
sc.tl.umap(hnsc)
sc.tl.leiden(hnsc, resolution=0.5, flavor="igraph", directed=False, n_iterations=2)
print(f"  Leiden clusters: {hnsc.obs['leiden'].nunique()}")

# Score IFN modules
print("Scoring IFN modules...")
module_keys = []
for mod_id, mod in sig_data["modules"].items():
    genes = [g for g in mod["genes"] if g in hnsc.raw.var_names]
    key   = f"score_{mod_id}"
    sc.tl.score_genes(hnsc, gene_list=genes, score_name=key)
    print(f"  {mod['label']}: {len(genes)}/{len(mod['genes'])} genes")
    module_keys.append(key)

cfg   = sig_data["scoring_config"]["weights"]
w     = pd.Series({f"score_{k}": v for k, v in cfg.items()})
raw   = hnsc.obs[w.index].multiply(w.values, axis=1).sum(axis=1)
ifn   = (raw - raw.min()) / (raw.max() - raw.min())
hnsc.obs["IFN_SS"] = ifn.values
hnsc.obs["cancer"] = "HNSCC"
print(f"  IFN-SS: {ifn.min():.3f} - {ifn.max():.3f}, mean {ifn.mean():.3f}")

# Annotate clusters with canonical markers
marker_genes = {
    "Th1 effector":    ["TBX21", "IFNG", "TNF", "CXCR3", "CCR5"],
    "Treg":            ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18"],
    "Exhausted CD4":   ["TOX", "PDCD1", "LAG3", "HAVCR2", "ENTPD1"],
    "Effector memory": ["CCR7", "SELL", "TCF7", "IL7R", "S100A4"],
    "Tfh-like":        ["CXCR5", "BCL6", "ICOS", "IL21"],
}
sc.tl.rank_genes_groups(hnsc, groupby="leiden", method="wilcoxon")

print("\nTop markers per cluster:")
for cluster in hnsc.obs["leiden"].unique():
    top = sc.get.rank_genes_groups_df(hnsc, group=cluster).head(5)["names"].tolist()
    ifn_mean = hnsc.obs[hnsc.obs["leiden"]==cluster]["IFN_SS"].mean()
    print(f"  Cluster {cluster} (IFN-SS={ifn_mean:.3f}): {top}")

hnsc.write_h5ad("data/processed/HNSCC_CD4_IFN_scored.h5ad")
print("Saved: data/processed/HNSCC_CD4_IFN_scored.h5ad")

# ── Load melanoma for comparison ──────────────────────────────────────────────
print("\nLoading melanoma dataset for cross-cancer comparison...")
mel = sc.read_h5ad(DATA_MELANOMA)
mel.obs["cancer"] = "Melanoma"

# ── Cross-cancer IFN-SS comparison ───────────────────────────────────────────
combined = pd.DataFrame({
    "IFN_SS": pd.concat([hnsc.obs["IFN_SS"], mel.obs["IFN_SS"]]),
    "cancer": pd.concat([hnsc.obs["cancer"], mel.obs["cancer"]]),
    "cell_type": pd.concat([
        hnsc.obs["leiden"].astype(str).rename("cell_type"),
        mel.obs.get("cell_type", mel.obs["leiden"].astype(str)).rename("cell_type")
    ])
})

stat, pval = mannwhitneyu(
    hnsc.obs["IFN_SS"].values,
    mel.obs["IFN_SS"].values,
    alternative="two-sided"
)
print(f"\nMann-Whitney U HNSCC vs Melanoma: p={pval:.3e}")
print(f"HNSCC IFN-SS mean: {hnsc.obs['IFN_SS'].mean():.3f}")
print(f"Melanoma IFN-SS mean: {mel.obs['IFN_SS'].mean():.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
sc.settings.figdir = str(OUTDIR)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: HNSCC UMAPs
sc.pl.umap(hnsc, color="leiden", title="HNSCC CD4+ TIL clusters",
           ax=axes[0,0], show=False)
sc.pl.umap(hnsc, color="IFN_SS", color_map="RdBu_r", vmin=0, vmax=1,
           title="HNSCC IFN Spectrum Score", ax=axes[0,1], show=False)
sc.pl.umap(hnsc, color="score_M3_exhaustion_IFN_overlap", color_map="Oranges",
           title="HNSCC M3: Exhaustion-IFN overlap", ax=axes[0,2], show=False)

# Row 2: Cross-cancer comparison
cancer_colors = {"HNSCC": "#E91E63", "Melanoma": "#2196F3"}
for i, cancer in enumerate(["HNSCC", "Melanoma"]):
    vals = combined[combined["cancer"]==cancer]["IFN_SS"]
    axes[1,i].hist(vals, bins=40, color=list(cancer_colors.values())[i],
                   alpha=0.8, edgecolor="white", linewidth=0.5)
    axes[1,i].set_xlabel("IFN Spectrum Score")
    axes[1,i].set_ylabel("Cells")
    axes[1,i].set_title(f"{cancer} CD4+ TILs\n(n={len(vals)}, mean={vals.mean():.3f})")
    axes[1,i].set_xlim(0, 1)
    axes[1,i].axvline(vals.mean(), color="black", linestyle="--", linewidth=1.5)

# Violin comparison
axes[1,2].violinplot(
    [combined[combined["cancer"]=="HNSCC"]["IFN_SS"].values,
     combined[combined["cancer"]=="Melanoma"]["IFN_SS"].values],
    positions=[1, 2], showmedians=True
)
axes[1,2].set_xticks([1, 2])
axes[1,2].set_xticklabels(["HNSCC\n(n=384)", f"Melanoma\n(n={len(mel)})"])
axes[1,2].set_ylabel("IFN Spectrum Score")
axes[1,2].set_title(f"Cross-cancer IFN-SS comparison\n(MWU p={pval:.2e})")
axes[1,2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(OUTDIR / "fig_replication_HNSCC.png", dpi=150, bbox_inches="tight")
print("Saved: results/figures/fig_replication_HNSCC.png")
plt.close()
print("Done.")