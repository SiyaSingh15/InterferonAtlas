import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import requests
from pathlib import Path

SIGS   = Path("signatures/ifn_modules_v1.json")
OUTDIR = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Load expression matrix ────────────────────────────────────────────────────
print("Loading IFN gene expression matrix...")
expr = pd.read_csv("data/tcga/pancan_ifn_genes.tsv", sep="\t", index_col=0)
expr = expr.T  # samples x genes
expr = expr.apply(pd.to_numeric, errors="coerce")
expr = expr.dropna(thresh=int(expr.shape[1] * 0.8))
print(f"  {expr.shape[0]} samples x {expr.shape[1]} genes")

# ── Load cancer type annotations from local file ──────────────────────────────
print("Loading cancer type annotations...")
pheno = pd.read_csv("data/tcga/tcga_phenotype.tsv.gz", sep="\t",
                    index_col=0, usecols=["sample", "_primary_disease"])
pheno.columns = ["cancer_type"]
pheno["cancer_type"] = pheno["cancer_type"].str.strip().str.lower()
print(f"  {pheno.shape[0]} samples with cancer type annotations")
print(f"  Cancer types: {pheno['cancer_type'].nunique()}")

# ── Load signatures and score ─────────────────────────────────────────────────
with open(SIGS) as f:
    sig_data = json.load(f)

print("Scoring IFN modules...")
module_scores = {}
for mod_id, mod in sig_data["modules"].items():
    genes_present = [g for g in mod["genes"] if g in expr.columns]
    module_scores[mod_id] = expr[genes_present].mean(axis=1)
    print(f"  {mod['label']}: {len(genes_present)}/{len(mod['genes'])} genes")

scores_df = pd.DataFrame(module_scores)
cfg = sig_data["scoring_config"]["weights"]
w = pd.Series(cfg)
raw_ss = scores_df[w.index].multiply(w.values, axis=1).sum(axis=1)
ifn_ss = (raw_ss - raw_ss.min()) / (raw_ss.max() - raw_ss.min())

# ── Combine with phenotype ────────────────────────────────────────────────────
results = pd.DataFrame({"IFN_SS": ifn_ss})
results = results.join(pheno, how="inner")
results = results.dropna()
print(f"\nFinal: {results.shape[0]} samples, {results['cancer_type'].nunique()} cancer types")

# ── Plot ──────────────────────────────────────────────────────────────────────
order = results.groupby("cancer_type")["IFN_SS"].median().sort_values(ascending=False)
top_types = order.index[:20]
plot_data = results[results["cancer_type"].isin(top_types)]

fig, ax = plt.subplots(figsize=(14, 6))
for i, ct in enumerate(top_types):
    vals = plot_data[plot_data["cancer_type"] == ct]["IFN_SS"]
    ax.boxplot(vals, positions=[i], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor="#4e79a7", alpha=0.7),
               medianprops=dict(color="white", linewidth=2),
               flierprops=dict(marker=".", markersize=2, alpha=0.3),
               whiskerprops=dict(linewidth=0.8),
               capprops=dict(linewidth=0.8))

ax.set_xticks(range(len(top_types)))
ax.set_xticklabels([c.upper() for c in top_types], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("IFN Spectrum Score (0=chronic, 1=acute)")
ax.set_title("IFN Spectrum Score across TCGA cancer types (top 20 by median)")
ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="Spectrum midpoint")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "fig1_TCGA_IFN_SS_by_cancer.png", dpi=150, bbox_inches="tight")
print("Saved: results/figures/fig1_TCGA_IFN_SS_by_cancer.png")
plt.close()

results.to_csv("results/tables/tcga_ifn_ss_scores.csv")
print("Saved: results/tables/tcga_ifn_ss_scores.csv")
print("\nDone.")