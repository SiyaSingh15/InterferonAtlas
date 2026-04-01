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
OUT_H5 = Path("data/processed/melanoma_CD4_final.h5ad")
OUTDIR.mkdir(parents=True, exist_ok=True)

# 1. Load
print("Loading...")
adata = sc.read_h5ad(DATA)
print(f"  {adata.shape[0]} cells x {adata.shape[1]} genes")

# 2. Preprocess + UMAP
# Data is already log-normalized TPM — skip normalize_total and log1p
print("Computing PCA, neighbors, UMAP...")
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
adata.raw = adata          # ← add this line
sc.pp.scale(adata, max_value=10)   # scale to unit variance for PCA
sc.tl.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, flavor="igraph", directed=False, n_iterations=2)
print(f"  Leiden clusters: {adata.obs['leiden'].nunique()}")
print(f"  obsm keys: {list(adata.obsm.keys())}")
n_before = adata.shape[0]
adata = adata[adata.obs['leiden'] != '5'].copy()
print(f"  Removed cluster 5 ({n_before - adata.shape[0]} cells). Remaining: {adata.shape[0]}")

# 3. Score IFN modules
print("Scoring IFN modules...")
with open(SIGS) as f:
    sig_data = json.load(f)

module_keys = []
for mod_id, mod in sig_data["modules"].items():
    key = f"score_{mod_id}"
    sc.tl.score_genes(adata, gene_list=mod["genes"], score_name=key)
    print(f"  {mod['label']}: {key}")
    module_keys.append(key)

scores = adata.obs[module_keys].copy()

# 4. IFN Spectrum Score
cfg = sig_data["scoring_config"]["weights"]
w = pd.Series({f"score_{k}": v for k, v in cfg.items()})
raw_ss = scores[w.index].multiply(w.values, axis=1).sum(axis=1)
ifn_ss = (raw_ss - raw_ss.min()) / (raw_ss.max() - raw_ss.min())
adata.obs["IFN_SS"] = ifn_ss.values
print(f"IFN-SS: {ifn_ss.min():.3f} to {ifn_ss.max():.3f}, mean {ifn_ss.mean():.3f}")

# 5. Bifurcation entropy
def row_entropy(row):
    row_shifted = row - row.min() + 1e-10
    p = row_shifted / row_shifted.sum()
    return entropy(p)

bif = scores.apply(row_entropy, axis=1)
bif_norm = (bif - bif.min()) / (bif.max() - bif.min())
adata.obs["IFN_bifurcation"] = bif_norm.values
print(f"Bifurcation entropy: {bif_norm.min():.3f} to {bif_norm.max():.3f}")

# 6. Save once
adata.write_h5ad(OUT_H5)
print(f"Saved: {OUT_H5}")

# 7. Plot
print("Plotting...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

sc.pl.umap(adata, color="leiden",
           title="Leiden clusters", ax=axes[0,0], show=False)
sc.pl.umap(adata, color="IFN_SS", color_map="RdBu_r", vmin=0, vmax=1,
           title="IFN Spectrum Score (0=chronic, 1=acute)", ax=axes[0,1], show=False)
sc.pl.umap(adata, color="IFN_bifurcation", color_map="viridis",
           title="Bifurcation entropy (high=uncommitted)", ax=axes[0,2], show=False)
sc.pl.umap(adata, color="score_M1_acute_IFN_antiviral", color_map="Reds",
           title="M1: Acute IFN-I antiviral", ax=axes[1,0], show=False)
sc.pl.umap(adata, color="score_M2_chronic_IFN_TME", color_map="Purples",
           title="M2: Chronic IFN-γ TME", ax=axes[1,1], show=False)
sc.pl.umap(adata, color="score_M3_exhaustion_IFN_overlap", color_map="Oranges",
           title="M3: Exhaustion-IFN overlap", ax=axes[1,2], show=False)

plt.tight_layout()
out = OUTDIR / "fig2_IFN_SS_UMAP.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
print("Done.")