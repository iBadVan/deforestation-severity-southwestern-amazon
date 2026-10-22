# Outputs of the six models

One folder per model, produced by `01_training.ipynb` (`/kaggle/working/resultados/<model>/`):

| File | Content |
|---|---|
| `pred_test_<model>.csv` | The 150 test patches with the reference class (`y_true`), the predicted class (`y_pred`) and the four class probabilities (`p0` to `p3`) |
| `metrics_<model>.json` | Accuracy, 95% Wilson interval, macro F1-score, per-class precision, recall and F1-score, parameters, epochs, training time and inference time per patch |
| `history_<model>.csv` | Loss, accuracy and learning rate per epoch on the training and validation subsets |
| `fig_curves_<model>.png`, `fig_cm_<model>.png` | Training curves and confusion matrix as produced by the training run |

These files are the inputs of `02_audit.ipynb` and `03_final_comparison.ipynb`.
`environment.json` records the library versions of the training environment.
The weights (`best_<model>.pt`) are not versioned because of their size.

| Model | Test accuracy (%) | Stopping epoch | Epoch of lowest validation loss | Lowest validation loss |
|---|---|---|---|---|
| ResNet50 | 71.33 | 19 | 11 | 0.64 |
| EfficientNetB0 | 58.00 | 20 | 12 | 1.33 |
| ViT-B/16 | 72.00 | 18 | 10 | 0.65 |
| Swin-T | 72.67 | 13 | 5 | 0.58 |
| CNN+LSTM | 69.33 | 11 | 3 | 0.71 |
| ConvLSTM | 64.67 | 50 | 46 | 0.76 |
