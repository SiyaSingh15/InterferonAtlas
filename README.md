# InterferonAtlas

**A web tool for continuous classification of interferon signaling states in cancer and immune biology**

> Investigator: Siya | IISER Tirupati, Biological Sciences (BS-MS, Year 3)  
> Collaborator: Prof. Mohit Kumar Jolly | PHCCO Lab, IISc Bangalore (Summer 2026)  
> Advisor: Prof. Sreenivas Chavali | IISER Tirupati  
> Target: Nucleic Acids Research — Web Server Issue (deadline ~Nov 2026)

---

## Project overview

InterferonAtlas computes a continuous **IFN Spectrum Score (IFN-SS)** for bulk or single-cell RNA-seq data, positioning each sample or cell between two biological attractor states:

- **Acute IFN pole** — early antiviral, pro-inflammatory (IFIT1, MX1, ISG15...)
- **Chronic IFN pole** — tumor-driven, immunosuppressive (STAT1, CXCL9, CD274...)

A Shannon entropy-based **bifurcation metric** identifies cells uncommitted between IFN states.

---

## Repository structure

```
InterferonAtlas/
├── environment.yml           # Conda environment (start here)
├── signatures/
│   └── ifn_modules_v1.json   # Four curated IFN gene modules + scoring config
├── notebooks/
│   ├── 00_env_test.py        # Verify your environment
│   ├── 01_scoring_pipeline.py  # IFN-SS on your CD4+ T cell scRNA-seq
│   └── 02_tcga_validation.py   # Pan-cancer bulk RNA-seq validation
├── src/interferonAtlas/
│   ├── scoring.py            # IFN-SS + bifurcation computation
│   ├── trajectory.py         # PAGA + diffusion pseudotime
│   └── utils.py              # I/O, gene symbol mapping, QC helpers
├── data/
│   ├── raw/                  # Downloaded GEO datasets (gitignored)
│   ├── processed/            # AnnData .h5ad objects (gitignored)
│   └── tcga/                 # TCGA matrices (gitignored)
├── results/
│   ├── figures/              # Output plots
│   └── tables/               # Output CSV/TSV tables
├── backend/                  # Phase 1 — FastAPI (May–June 2026)
└── frontend/                 # Phase 2 — React + Vite (July–Sept 2026)
```

---

## Quickstart

```bash
# 1. Create and activate environment
conda env create -f environment.yml
conda activate interferonAtlas

# 2. Verify installation
python notebooks/00_env_test.py

# 3. Open notebooks in Jupyter
jupyter lab
```

---

## Roadmap

| Phase | Timeline | Goal |
|-------|----------|------|
| **Phase 0** | Apr 2026 | Scoring pipeline in Jupyter; validate on CD4+ T cell scRNA-seq |
| **Phase 1** | May–Jun 2026 | FastAPI backend; IISc hybrid E/M data; IFN-EMT finding |
| **Phase 2** | Jul–Sep 2026 | React frontend; Hugging Face Spaces deployment |
| **Phase 3** | Oct–Nov 2026 | NAR Web Server Issue submission + bioRxiv preprint |

---

## Key references

- Schneider WM et al. (2014) Annu Rev Immunol — ISG functions
- Benci JL et al. (2019) Cell — Opposing IFN functions in tumor editing
- Wherry EJ & Kurachi M (2015) Nat Rev Immunol — T cell exhaustion
- Sade-Feldman M et al. (2018) Cell (GSE120575) — Melanoma TIL scRNA-seq
- Jolly MK et al. (2015) Front Oncol — Hybrid E/M phenotype
