import scanpy as sc
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import entropy
from pathlib import Path

DATA   = Path("/home/siyasingh15/cd4_til_project/data/processed/melanoma_CD4_purified.h5ad")
SIGS   = Path("signatures/ifn_modules_v1.json")
OUTDIR = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
adata = sc.read_h5ad(DATA)
print(f"  {adata.shape[0]} cells x {adata.shape[1]} genes")
print(f"  obs columns: {list(adata.obs.columns)}")

with open(SIGS) as f:
    sig_data = json.load(f)

# Score each module with sc.tl.score_genes (adds to adata.obs)
module_score_keys = []
for mod_id, mod in sig_data["modules"].items():
    key = f"score_{mod_id}"
    sc.tl.score_genes(adata, gene_list=mod["genes"], score_name=key)
    print(f"  Scored {mod['label']}: {key}")
    module_score_keys.append(key)

scores = adata.obs[module_score_keys].copy()
print(f"\nScore matrix shape: {scores.shape}")
print(scores.describe().round(3))

# IFN Spectrum Score — weighted sum, normalized to [0, 1]
cfg = sig_data["scoring_config"]["weights"]
w = pd.Series({f"score_{k}": v for k, v in cfg.items()})
raw_ss = scores[w.index].multiply(w.values, axis=1).sum(axis=1)
ifn_ss = (raw_ss - raw_ss.min()) / (raw_ss.max() - raw_ss.min())
adata.obs["IFN_SS"] = ifn_ss.values
print(f"\nIFN-SS: {ifn_ss.min():.3f} to {ifn_ss.max():.3f}, mean {ifn_ss.mean():.3f}")

# Bifurcation metric — Shannon entropy across module scores
# Shift scores to be non-negative before normalizing to probabilities
def row_entropy(row):
    row_shifted = row - row.min() + 1e-10
    p = row_shifted / row_shifted.sum()
    return entropy(p)

bif = scores.apply(row_entropy, axis=1)
bif_norm = (bif - bif.min()) / (bif.max() - bif.min())
adata.obs["IFN_bifurcation"] = bif_norm.values
print(f"Bifurcation entropy: {bif_norm.min():.3f} to {bif_norm.max():.3f}")

# Plot
sc.settings.figdir = str(OUTDIR)
if "X_umap" in adata.obsm:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(adata, color="IFN_SS", color_map="RdBu_r", vmin=0, vmax=1,
               title="IFN Spectrum Score (0=chronic, 1=acute)", ax=axes[0], show=False)
    sc.pl.umap(adata, color="IFN_bifurcation", color_map="viridis",
               title="IFN Bifurcation entropy", ax=axes[1], show=False)
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig2_IFN_SS_UMAP.png", dpi=150, bbox_inches="tight")
    print("Saved: results/figures/fig2_IFN_SS_UMAP.png")
    plt.close()
else:
    print("No X_umap found — plotting score distributions instead")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(ifn_ss, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("IFN Spectrum Score"); axes[0].set_ylabel("Cells")
    axes[0].set_title("IFN-SS distribution")
    axes[1].hist(bif_norm, bins=50, color="darkorange", edgecolor="white")
    axes[1].set_xlabel("Bifurcation entropy"); axes[1].set_ylabel("Cells")
    axes[1].set_title("Bifurcation entropy distribution")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig2_IFN_SS_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved: results/figures/fig2_IFN_SS_distributions.png")
    plt.close()

adata.write_h5ad("data/processed/melanoma_CD4_IFN_scored.h5ad")
print("Saved: data/processed/melanoma_CD4_IFN_scored.h5ad")
print("\nDone.")