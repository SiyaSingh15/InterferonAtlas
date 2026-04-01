import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("results/figures")
adata = sc.read_h5ad("data/processed/melanoma_CD4_final.h5ad")
print(f"Loaded: {adata.shape[0]} cells, {adata.obs['leiden'].nunique()} clusters")

marker_genes = {
    "Th1 effector":    ["TBX21", "IFNG", "TNF", "CXCR3", "CCR5"],
    "Treg":            ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18"],
    "Exhausted CD4":   ["TOX", "PDCD1", "LAG3", "HAVCR2", "ENTPD1"],
    "Effector memory": ["CCR7", "SELL", "TCF7", "IL7R", "S100A4"],
    "Tfh-like":        ["CXCR5", "BCL6", "ICOS", "IL21", "PDCD1"],
}

sc.settings.figdir = str(OUTDIR)
fig, ax = plt.subplots(figsize=(16, 5))
sc.pl.dotplot(adata, var_names=marker_genes, groupby="leiden",
              ax=ax, show=False, standard_scale="var")
plt.tight_layout()
plt.savefig(OUTDIR / "fig2b_cluster_markers_dotplot.png", dpi=150, bbox_inches="tight")
print("Saved: results/figures/fig2b_cluster_markers_dotplot.png")
plt.close()

print("\nPer-cluster IFN-SS and module scores:")
cols = ["IFN_SS", "IFN_bifurcation",
        "score_M1_acute_IFN_antiviral", "score_M2_chronic_IFN_TME",
        "score_M3_exhaustion_IFN_overlap"]
summary = adata.obs.groupby("leiden")[cols].mean().round(3)
print(summary.to_string())

# Add annotations to adata and re-plot UMAP with labels
cluster_names = {
    "0": "Treg-like exhausted",
    "1": "Th1-like exhausted",
    "2": "Effector memory / stem-like",
    "3": "Terminally exhausted",
    "4": "Tfh-like exhausted",
}
adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_names).astype("category")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata, color="cell_type", title="CD4+ TIL subsets",
           ax=axes[0], show=False, legend_loc="on data", legend_fontsize=7)
sc.pl.umap(adata, color="IFN_SS", color_map="RdBu_r", vmin=0, vmax=1,
           title="IFN Spectrum Score", ax=axes[1], show=False)
sc.pl.umap(adata, color="IFN_bifurcation", color_map="viridis",
           title="Bifurcation entropy", ax=axes[2], show=False)
plt.tight_layout()
plt.savefig(OUTDIR / "fig2_final_annotated_UMAP.png", dpi=150, bbox_inches="tight")
print("Saved: results/figures/fig2_final_annotated_UMAP.png")
plt.close()

adata.write_h5ad("data/processed/melanoma_CD4_final.h5ad")
print("Saved annotated h5ad.")