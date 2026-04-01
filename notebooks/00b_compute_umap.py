import scanpy as sc
from pathlib import Path

adata = sc.read_h5ad("data/processed/melanoma_CD4_IFN_scored.h5ad")
print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

# Standard preprocessing pipeline
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
print(f"Highly variable genes: {adata.var.highly_variable.sum()}")

sc.tl.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
print(f"Leiden clusters: {adata.obs['leiden'].nunique()}")

adata.write_h5ad("data/processed/melanoma_CD4_IFN_scored.h5ad")
print("Saved with UMAP. Now run 01_scoring_pipeline.py again to plot.")