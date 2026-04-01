import scanpy as sc
import pandas as pd
import numpy as np
import json
from scipy.stats import entropy
from pathlib import Path

SIGS = Path(__file__).parent.parent / "signatures" / "ifn_modules_v1.json"

with open(SIGS) as f:
    SIG_DATA = json.load(f)


def load_signatures():
    return SIG_DATA


def preprocess(adata: sc.AnnData) -> sc.AnnData:
    """Normalize, scale, compute PCA + UMAP if not present."""
    # Normalize only if data looks like raw counts
    if adata.X.max() > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_pcs=30)

    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)

    sc.tl.leiden(adata, resolution=0.5, flavor="igraph",
                 directed=False, n_iterations=2)
    return adata


def score_ifn(adata: sc.AnnData) -> sc.AnnData:
    """Score all four IFN modules and compute IFN-SS + bifurcation entropy."""
    sig_data = load_signatures()
    module_keys = []
    module_coverage = {}

    var_names = set(adata.raw.var_names if adata.raw else adata.var_names)

    for mod_id, mod in sig_data["modules"].items():
        genes_present = [g for g in mod["genes"] if g in var_names]
        key = f"score_{mod_id}"
        sc.tl.score_genes(adata, gene_list=genes_present, score_name=key)
        module_keys.append(key)
        module_coverage[mod_id] = {
            "found": len(genes_present),
            "total": len(mod["genes"]),
            "label": mod["label"]
        }

    # IFN Spectrum Score
    cfg = sig_data["scoring_config"]["weights"]
    w = pd.Series({f"score_{k}": v for k, v in cfg.items()})
    scores = adata.obs[w.index]
    raw_ss = scores.multiply(w.values, axis=1).sum(axis=1)
    ifn_ss = (raw_ss - raw_ss.min()) / (raw_ss.max() - raw_ss.min())
    adata.obs["IFN_SS"] = ifn_ss.values

    # Bifurcation entropy
    def row_entropy(row):
        row_shifted = row - row.min() + 1e-10
        p = row_shifted / row_shifted.sum()
        return entropy(p)

    module_scores = adata.obs[module_keys]
    bif = module_scores.apply(row_entropy, axis=1)
    bif_norm = (bif - bif.min()) / (bif.max() - bif.min())
    adata.obs["IFN_bifurcation"] = bif_norm.values

    adata.uns["module_coverage"] = module_coverage
    return adata


def get_score_summary(adata: sc.AnnData) -> dict:
    return {
        "n_cells": adata.shape[0],
        "ifn_ss_mean": float(adata.obs["IFN_SS"].mean()),
        "ifn_ss_min":  float(adata.obs["IFN_SS"].min()),
        "ifn_ss_max":  float(adata.obs["IFN_SS"].max()),
        "bifurcation_mean": float(adata.obs["IFN_bifurcation"].mean()),
        "module_coverage": adata.uns.get("module_coverage", {}),
    }