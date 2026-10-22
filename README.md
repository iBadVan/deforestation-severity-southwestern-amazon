# CNN, Transformer, and Spatiotemporal Models for Deforestation Severity Classification in the Southwestern Amazon

Replication package of the study. It contains the code that prepares the data, trains the six models, evaluates them,
audits the test subset for data leakage and analyzes the errors, together with the result files used in the paper.

## Data

The study uses the public dataset *data de deforestación* on Kaggle:
https://www.kaggle.com/datasets/vergaramarcoseduardo/data-de-deforestacin

- 1500 patches of 512 x 512 pixels (5.12 km x 5.12 km) with five annual Sentinel-2 L2A images (2020 to 2024) and six bands (B2, B3, B4, B8, B11, B12).
- Four classes from the percentage of Hansen tree cover loss (`loss_pct`) and the thresholds of the dataset author: no deforestation (0%), low (up to 3%), moderate (up to 7%) and severe (more than 7%).
- 375 patches per class.

The data are not redistributed here. The preprocessing notebook builds a cache of about 6 GB from the original files.

## Pipeline

| Notebook | Hardware | Input | Output |
|---|---|---|---|
| `00_preprocessing_cache.ipynb` | CPU | Kaggle dataset | Stratified partition 80/10/10 (seed 42: 1200/150/150), gap filling and clipping with training statistics, complete patch resized to 224 x 224, NDVI and NBR, normalization with training statistics (`X224.npy`, `meta.csv`, `stats.json`) |
| `01_training.ipynb` | GPU T4 | Cache | One model per run (change `MODEL` and `MODE`): predictions, metrics, history and figures |
| `02_audit.ipynb` | CPU | Cache and outputs of the six models | Leakage audit (identical tensors, spatial overlap, Fisher and Mantel-Haenszel tests) and error analysis (accuracy by range of loss_pct, ordinal metrics, classical baselines, Random Forest formulations) |
| `03_final_comparison.ipynb` | CPU | Outputs of the six models | Global table with Wilson intervals, per-class metrics, ordinal metrics, exact McNemar test for the 15 pairs, soft voting ensembles and the figures of the paper |
| `04_reproducibility_efficientnet_b0.ipynb` | GPU T4 | Cache and outputs of the six models | Second run of EfficientNetB0 and comparison with the first run |

Common training conditions: ImageNet weights from timm (ConvLSTM from random weights), AdamW (learning rate 1e-4, weight decay 1e-4),
batch size 16, cross-entropy loss, ReduceLROnPlateau (factor 0.5, patience 3), early stopping on the validation loss (patience 8, maximum 50 epochs),
weights of the epoch with the lowest validation loss, flips and rotations only in training, mixed precision and seed 42.
The CNNs and transformers receive the 2020 and 2024 images stacked as 16 channels, and the spatiotemporal models receive the five dates with eight channels.

## Main results (150 test patches)

| Model | Family | Accuracy (%) | 95% Wilson CI (%) | Macro F1 (%) | Quadratic kappa | Within one class (%) | Two groups (%) | Parameters (M) |
|---|---|---|---|---|---|---|---|---|
| ResNet50 | CNN | 71.33 | 63.64 to 77.97 | 70.99 | 0.85 | 96.00 | 85.33 | 23.56 |
| EfficientNetB0 | CNN | 58.00 | 50.00 to 65.60 | 58.16 | 0.80 | 97.33 | 78.67 | 4.02 |
| ViT-B/16 | Transformer | 72.00 | 64.33 to 78.57 | 70.90 | 0.89 | 100.00 | 86.67 | 88.36 |
| Swin-T | Transformer | 72.67 | 65.04 to 79.17 | 72.45 | 0.89 | 100.00 | 89.33 | 27.54 |
| CNN+LSTM | Spatiotemporal | 69.33 | 61.55 to 76.15 | 69.20 | 0.86 | 99.33 | 86.00 | 5.58 |
| ConvLSTM | Spatiotemporal | 64.67 | 56.74 to 71.86 | 63.82 | 0.84 | 98.00 | 84.00 | 0.61 |

- Exact McNemar test: only the pair EfficientNetB0 and Swin-T is below the Bonferroni threshold of 0.0033 (p = 0.0021).
- Audit: no identical tensors; 56 of the 150 test patches share 25% or 50% of their area with a training patch, and the accuracy shows no statistically significant difference with and without overlap (Fisher p from 0.2377 to 1.0000; Mantel-Haenszel p from 0.2597 to 0.9935).
- Error analysis: 224 of the 288 test errors of the six models are confusions between no deforestation and low or between moderate and severe. A logistic regression with 76 spectral descriptors reaches 69.33%, and no ensemble differs from Swin-T.
- Reproducibility: a second run of EfficientNetB0 with the same code, data and seed gave the same predictions on the 150 test patches and the same training history. Inference times depend on the state of the Kaggle machine.

## Environment

Kaggle notebooks with Python 3.13.15, an NVIDIA Tesla T4 GPU and CUDA 12.8. Library versions are listed in `requirements.txt` and in `results/models/environment.json`.

## How to reproduce

1. Create a Kaggle notebook from `00_preprocessing_cache.ipynb` (CPU) with the Kaggle dataset as input, run it with Save Version and publish its output as a private dataset (the cache).
2. For each model, run `01_training.ipynb` with GPU T4 and Internet, the cache as input and the values of `MODEL` and `MODE` given in the notebook. First run one epoch with `TEST_RUN = True`.
3. Upload the six output folders as a dataset and run `03_final_comparison.ipynb` and `02_audit.ipynb` (CPU) with it and the cache as inputs.
4. Optionally, run `04_reproducibility_efficientnet_b0.ipynb` to check the reproducibility of the training.

## Repository structure

```
notebooks/      five notebooks of the pipeline
results/final/  tables and figures of the final comparison
results/audit/  tables and figures of the audit and the error analysis
results/reproducibility/efficientnet_b0/  second run of EfficientNetB0
results/models/ predictions, metrics and training history of the six models
figures/        methodology diagram and summary figures of the paper
```

## License

MIT License. See `LICENSE`.
