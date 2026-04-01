import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad("data/processed/melanoma_CD4_IFN_scored.h5ad")
print("obsm keys:", list(adata.obsm.keys()))
print("obs columns:", list(adata.obs.columns))

sc.settings.figdir = str(OUTDIR)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

sc.pl.umap(adata, color="leiden",
           title="Leiden clusters", ax=axes[0,0], show=False)
sc.pl.umap(adata, color="IFN_SS", color_map="RdBu_r", vmin=0, vmax=1,
           title="IFN Spectrum Score (0=chronic, 1=acute)", ax=axes[0,1], show=False)
sc.pl.umap(adata, color="IFN_bifurcation", color_map="viridis",
           title="Bifurcation entropy", ax=axes[0,2], show=False)
sc.pl.umap(adata, color="score_M1_acute_IFN_antiviral", color_map="Reds",
           title="M1: Acute IFN-I antiviral", ax=axes[1,0], show=False)
sc.pl.umap(adata, color="score_M2_chronic_IFN_TME", color_map="Purples",
           title="M2: Chronic IFN-γ TME", ax=axes[1,1], show=False)
sc.pl.umap(adata, color="score_M3_exhaustion_IFN_overlap", color_map="Oranges",
           title="M3: Exhaustion-IFN overlap", ax=axes[1,2], show=False)

plt.tight_layout()
plt.savefig(OUTDIR / "fig2_IFN_SS_UMAP.png", dpi=150, bbox_inches="tight")
print("Saved: results/figures/fig2_IFN_SS_UMAP.png")
plt.close()