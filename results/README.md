# Results

| Folder | Content | Produced by |
|---|---|---|
| `final/` | Global metrics with 95% Wilson intervals, per-class metrics, ordinal metrics, exact McNemar test with Bonferroni threshold, ensembles and the 12 figures of the paper (training curves and confusion matrices) | `03_final_comparison.ipynb` |
| `audit/` | Spatial overlap between test and training patches, Fisher and Mantel-Haenszel tests, accuracy by range of loss_pct, ordinal metrics, classical baselines, patches misclassified by all models and Random Forest formulations | `02_audit.ipynb` |
| `reproducibility/efficientnet_b0/` | Second run of EfficientNetB0 with the same code, data and seed (same predictions and training history as the first run) | `04_reproducibility_efficientnet_b0.ipynb` |
| `models/` | Predictions, metrics and training history of the six models, which are the inputs of the audit and of the final comparison | `01_training.ipynb` |

The audit figures were produced by the first version of the audit notebook, so some of their titles keep the original variable names.
All numbers of the paper come from these files.
