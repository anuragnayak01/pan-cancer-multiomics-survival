 

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-Scientific%20Reports-green.svg)](#)

> **Causally Grounded Pan-Cancer Survival Prediction from Multi-Omics Molecular Subtypes with Conformal Uncertainty Quantification**
>
> Neha Kasture · Anurag Nayak · Raghav Singh · Varad Patil — IIIT Nagpur

---

## Overview

It computational framework for pan-cancer survival prediction from multi-omics molecular data. Applied to **6,967 TCGA patients across 30 cancer types**, it integrates mRNA expression, miRNA, copy number variation, and somatic mutation data through a strictly leakage-free eleven-phase pipeline.

| Component | Key result |
|---|---|
| MOFA+ molecular subtypes | 9 subtypes, ARI = 0.9996 ± 0.0009 |
| Survival discrimination | C = 0.776 (95% CI [0.752, 0.799]) |
| Risk stratification | log-rank p = 3.44×10⁻⁵⁸ |
| Causal stage effect | −100 to −235 days RMST (four estimators) |
| Conformal coverage | 99.3% (target: 90%) |
| External validation | C = 0.568 on METABRIC (n=1,980) |

---

## Repository Structure

```
pan-cancer-multiomics-survival/
├── initial_preprocessing.py
├── patches.py                          
├── cohort_assembly.py           QC, outlier removal, cohort construction
├── post_preprocessing.py        Modality-specific normalisation
├── feature_selection.py         MAD → Cox-BH → Elastic net (tri-stage)
├── integration.py               MOFA+ factor extraction + Leiden clustering
├── survival_eda.py              Kaplan-Meier plots, subtype visualisation
├── causal_analysis.py           Double-robust ATE estimation (IVW pooled)
├── survival_model.py            Multi-task learning (4 architectures)
├── uncertainty.py               Mondrian conformal + MC Dropout + ensembles
├── interpretability.py          SHAP attributions + HITL review queue
├── external_validation.py       METABRIC transfer with feature alignment
└── ablation_studies.py          Modality/loss/task ablations + baselines
│
├── run_pipeline.py                 Master runner  with checkpointing
│
├── configs/
│   └── default.yaml                Default hyperparameters (all phases)
│
├── scripts/
│   ├── download_tcga.sh            Download TCGA parquet files from GDC
│   └── download_metabric.sh        Download METABRIC from cBioPortal
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/anuragnayak01/MOSAIC.git
cd MOSAIC
pip install -r requirements.txt
```

---

## Data

### TCGA (training)
```bash
# Option 1: use the provided script
bash scripts/download_tcga.sh /path/to/output/

# Option 2: download manually from GDC portal
# https://portal.gdc.cancer.gov
# Required files (parquet format):
#   mRNA.parquet, miRNA.parquet, CNV.parquet,
#   mutations.parquet, clinical.parquet
```

### METABRIC (external validation, Phase 10 only)
```bash
bash scripts/download_metabric.sh /path/to/output/

# Or manually:
git clone https://github.com/cBioPortal/datahub.git
# Data at: datahub/public/brca_metabric/
```

---

## Usage

### Run full pipeline (Phases 1–11)
```bash
python run_pipeline.py /path/to/tcga/parquets/
```

### Start from a specific phase (uses saved checkpoints)
```bash
python run_pipeline.py /path/to/data/ --from 5
```

### Run with METABRIC validation
```bash
python run_pipeline.py /path/to/data/ \
    --metabric-dir /path/to/brca_metabric/
```

### Check pipeline status
```bash
python run_pipeline.py --status
```

### Skip optional phases
```bash
python run_pipeline.py /path/to/data/ --no-validation --no-ablations
```

### Re-run from scratch (ignores checkpoints)
```bash
python run_pipeline.py /path/to/data/ --no-resume
```

---

## Pipeline Phases

| Phase | Module | Description |
|---|---|---|
| 1 | `cohort_assembly` | Modality intersection, IsolationForest QC, 90-day filter → **6,967 patients** |
| 2 | `preprocessing` | VST (mRNA), CPM (miRNA), ±3σ winsorisation (CNV), ComBat batch correction |
| 3 | `feature_selection` | MAD → univariate Cox-BH (FDR < 0.05) → elastic net Cox (α=0.9) → **1,366 features** |
| 4 | `integration` | MOFA+ (K=15) → KNN graph → Leiden (γ=0.3) → **9 molecular subtypes** |
| 5 | `survival_eda` | KM plots, UMAP, subtype biological composition |
| 6 | `causal_analysis` | CausalForestDML, AIPW, IPW, VirtualTwins → IVW meta-analysis |
| 7 | `survival_model` | HardSharing / Hierarchical / CrossOmicsMTL / NMF-MTLR with Nash-MTL |
| 8 | `uncertainty` | Mondrian conformal, MC Dropout (T=50), deep ensembles (M=5), IBS/D-Cal |
| 9 | `interpretability` | SHAP KernelExplainer, HITL flagging queue |
| 10 | `external_validation` | METABRIC feature alignment, zero-fill missing modalities, C-index |
| 11 | `ablation_studies` | Modality ablation (A1), loss balancing (A3), auxiliary tasks (A4), baselines (C1, C2) |

---

## Programmatic API

```python
# Run individual modules
from mosaic.integration import run_integration
from mosaic.survival_model import run_survival_model, HardSharingMTL

# Load a checkpoint
import pickle
with open("output/checkpoints/phase4_checkpoint.pkl", "rb") as f:
    integration_result = pickle.load(f)

# Run external validation
from mosaic.external_validation import (
    load_metabric, run_external_validation, load_tcga_train_stats
)

cohort = load_metabric("/path/to/brca_metabric/")
tcga_stats, tcga_feats = load_tcga_train_stats("output/tcga_train_stats.pkl")

result = run_external_validation(
    cohort       = cohort,
    weights_path = "output/mtl/HardSharing_model.pt",
    tcga_stats   = tcga_stats,
    tcga_feats   = tcga_feats,
)
print(f"C-index: {result['cindex']:.4f}")
```

---

## Key Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| IsolationForest | contamination | 1% |
| MAD filter | top-K (mRNA/miRNA/CNV) | 3,000 / 432 / 2,000 |
| Elastic net | α, CV folds | 0.9, 5 |
| MOFA+ | K factors | 15 |
| Leiden | resolution γ | 0.3 |
| MTL encoder | hidden dim / dropout | 256 / 0.3 |
| Training | lr, batch, epochs | 1e-3, 256, 100 |
| Conformal | target coverage | 90% |
| MC Dropout | passes T | 50 |
| Ensembles | members M | 5 |
| Random seed | all experiments | 42 |

Full table: `configs/default.yaml`

 

---

## Data Availability

- TCGA: [GDC Data Portal](https://portal.gdc.cancer.gov)
- METABRIC: [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric)

---

## License

MIT — see [LICENSE](LICENSE)
