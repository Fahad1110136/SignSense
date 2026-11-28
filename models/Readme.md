# Models

This folder containing the Trained models, evaluation tables and configuration files of **SignSense**.

The models are trained on the processed data described in [`dataset/Readme.md`](../dataset/Readme.md). The web app does not read this folder directly: it uses the copies in `app/models/`.

## Models at a glance

| Task | Model | Input | Output | File |
|---|---|---|---|---|
| Static letters (36 classes) | Baseline CNN | 224x224 padded RGB image | class probabilities | `baseline_cnn_fp32.keras` |
| Static letters | MobileNetV2 (ImageNet, fine-tuned) | 224x224 padded RGB image | class probabilities | `mobilenetv2_fp32.keras` |
| Static letters | EfficientNet-B0 (ImageNet, fine-tuned) | 224x224 padded RGB image | class probabilities | `efficientnetb0_fp32.keras` |
| Dynamic letters (4 classes) | Random Forest | summary statistics of an 88-frame landmark sequence | class probabilities | `dynamic_rf.joblib` |
| Control gestures (space / delete / other) | Gradient boosting (HistGradientBoosting) | 63 hand-landmark values of one frame | class probabilities | `control_gesture_clf.joblib` |

The app averages the probabilities of the three static models (ensemble) or uses MobileNetV2 alone in fast mode. Class names and Urdu characters are listed in [`dataset/Readme.md`](../dataset/Readme.md).

## Files in this folder

### Model files

| File | Produced by | Description |
|---|---|---|
| `baseline_cnn.keras` | `static_models.ipynb` | Baseline CNN as trained on the GPU (20.7 MB) |
| `mobilenetv2.keras` | `static_models.ipynb` | MobileNetV2 as trained on the GPU, 80 layers unfrozen, learning rate 1e-4 (26.5 MB) |
| `efficientnetb0.keras` | `static_models.ipynb` | EfficientNet-B0 as trained on the GPU (33.9 MB) |
| `baseline_cnn_fp32.keras` | `app_prep.ipynb` | Float32 export of the baseline CNN for CPU use (4.3 MB) |
| `mobilenetv2_fp32.keras` | `app_prep.ipynb` | Float32 export of MobileNetV2 for CPU use (9.8 MB) |
| `efficientnetb0_fp32.keras` | `app_prep.ipynb` | Float32 export of EfficientNet-B0 for CPU use (17.2 MB) |
| `dynamic_rf.joblib` | `pipeline.ipynb`, `app_prep.ipynb` | Random Forest for the four dynamic letters (400 trees) |
| `control_gesture_clf.joblib` | `pipeline.ipynb`, `app_prep.ipynb` | Space / delete / other classifier on hand landmarks |

The LSTM, GRU, 1D-CNN and MLP that were compared with the Random Forest for dynamic letters are not saved; only their results are kept (see the `dynamic_*.csv` files).

### Configuration and resources

| File | Produced by | Description |
|---|---|---|
| `app_config.json` | `app_prep.ipynb` | Settings read by the app: static model files and mode, padding colour, dynamic model and frame count, control threshold and classes, typing-engine settings, lexicon file |
| `pipeline_config.json` | `pipeline.ipynb` | Settings chosen in the pipeline study (including the rejected motion-router parameters) |
| `dynamic_config.json` | `dynamic_models.ipynb` | Best setting per dynamic model family (velocity features and augmentation on, Random Forest as primary) |
| `urdu_lexicon.txt` | `pipeline.ipynb` | Demo dictionary of 170 Urdu words used for correction when a word ends |

The typing engine itself (`typing_engine.py`) lives in `scripts/`, and a copy is used by the app.

### Static letter results

| File | Produced by | Description |
|---|---|---|
| `static_histories.json` | `static_models.ipynb` | Per-epoch training and validation accuracy and loss of the three models |
| `static_tuning_val.csv` | `static_models.ipynb` | Validation results of the four MobileNetV2 fine-tuning setups |
| `static_results_val.csv` | `static_models.ipynb` | Validation accuracy, macro-F1, size, latency and training time per model |
| `static_results_val_test_all.csv` | `static_evaluation.ipynb` | Validation and test metrics (accuracy, precision, recall, macro and weighted F1, top-3) for all images and for originals only |
| `static_results_test.csv` | `static_evaluation.ipynb` | The test rows of the table above |
| `static_per_class_test.csv` | `static_evaluation.ipynb` | Precision, recall, F1 and support per letter and model on the test set |
| `static_bootstrap_ci.csv` | `static_evaluation.ipynb` | Test accuracy with 95% subject-level bootstrap intervals |
| `static_bootstrap_diffs.csv` | `static_evaluation.ipynb` | Bootstrap differences between models and whether they are clearly different |
| `static_confidence_thresholds.csv` | `static_evaluation.ipynb` | Coverage and accuracy of accepted predictions as the confidence threshold rises |
| `static_model_speed_size.csv` | `static_evaluation.ipynb` | Parameters, size and GPU latency / throughput |
| `static_cpu_latency.csv` | `app_prep.ipynb` | CPU milliseconds per frame, file size and frames per second, including the ensemble |
| `float32_parity.csv` | `app_prep.ipynb` | Check that each float32 export gives the same predictions as the original model |
| `static_test_predictions.npz` | `static_evaluation.ipynb` | Test probabilities of every model (3,238 images x 36 classes) with labels, subject ids and original flag |

### Dynamic letter results

| File | Produced by | Description |
|---|---|---|
| `dynamic_ablation.csv` | `dynamic_models.ipynb` | Cross-validated LSTM with and without velocity features and augmentation |
| `dynamic_cv_tuning.csv` | `dynamic_models.ipynb` | Cross-validated results of every tried configuration per model family |
| `dynamic_cv_best.csv` | `dynamic_models.ipynb` | Best configuration of each family under 5-fold cross-validation |
| `dynamic_results_test.csv` | `dynamic_models.ipynb` | Accuracy, precision, recall and macro-F1 on the 59 held-out test videos, with bootstrap intervals |
| `dynamic_bootstrap_diffs.csv` | `dynamic_models.ipynb` | Bootstrap differences between the Random Forest and the other families |
| `dynamic_latency.csv` | `dynamic_models.ipynb` | Parameters and CPU latency per video of the neural models |
| `dynamic_test_predictions.npz` | `dynamic_models.ipynb` | Test probabilities of all five families (59 videos x 4 classes) with labels and file names |

### Pipeline results

| File | Produced by | Description |
|---|---|---|
| `control_classifier_comparison.csv` | `pipeline.ipynb` | Random Forest against gradient boosting for the control gestures at threshold 0.9 |
| `control_threshold_table.csv` | `pipeline.ipynb` | Recall of space and delete and false-alarm rates for thresholds from 0.50 upward |
| `pipeline_simulation_sweep.csv` | `pipeline.ipynb` | Simulated word accuracy, character error rate, delay and error types for every model, confidence threshold, stability setting and correction on or off, for dictionary and unknown words |

### Static improvement round results

| File | Produced by | Description |
|---|---|---|
| `mobilenetv2_v2a.keras`, `mobilenetv2_v2b.keras` | `static_improvement.ipynb` | MobileNetV2 trained with the improved recipe (v2a: stronger augmentation + label smoothing; v2b: also all layers fine-tuned with a cosine schedule). Mixed-precision models: convert to float32 for the CPU app, as in `app_prep.ipynb`. Keep only the chosen one |
| `static_landmark_clf.joblib` | `static_improvement.ipynb` | Classifier on hand landmarks (63 normalized coordinates + 210 pairwise distances). scikit-learn version-sensitive: the app retrains it from data if loading fails |
| `static_improvement_config.json` | `static_improvement.ipynb` | Selected candidate, fusion weights and file names |
| `static_improvement_val.csv` | `static_improvement.ipynb` | Validation accuracy and macro-F1 of every fusion candidate (weights tuned on validation) |
| `static_improvement_test.csv` | `static_improvement.ipynb` | Test metrics (all images / originals only) of every candidate |
| `static_improvement_ci.csv` | `static_improvement.ipynb` | 95% subject-level bootstrap intervals of the test accuracy |
| `static_improvement_diffs.csv` | `static_improvement.ipynb` | Paired gain of each candidate over the current v1 ensemble, with a `real_improvement` flag |
| `static_improvement_per_class_gain.csv` | `static_improvement.ipynb` | Change in test F1 per letter |
| `static_landmark_models_val.csv` | `static_improvement.ipynb` | Validation results of the landmark model families and augmentation options |
| `static_cnn_v2_val.csv` | `static_improvement.ipynb` | Validation results of the two CNN recipes |
| `static_improvement_test_predictions.npz` | `static_improvement.ipynb` | Fused test probabilities (float16) of every candidate, with the labels |

## Results summary

### Static letters (test set, people never seen in training)

| Model | Val acc | Test acc | 95% interval | Test macro-F1 | Top-3 | CPU ms/frame | Size (float32) |
|---|---|---|---|---|---|---|---|
| Baseline CNN | 92.8% | 80.5% | 60.6 to 93.5% | 0.802 | 93.7% | 99 | 4.3 MB |
| MobileNetV2 | 93.2% | 87.4% | 83.8 to 91.6% | 0.869 | 97.3% | 55 | 9.8 MB |
| EfficientNet-B0 | 90.9% | 85.5% | 82.8 to 88.3% | 0.853 | 97.7% | 68 | 17.2 MB |
| Ensemble of 3 | n/a | 89.8% | 86.3 to 94.6% | 0.897 | 97.7% | 222 | 31.3 MB |

MobileNetV2 was tuned over four setups; unfreezing 80 layers at a learning rate of 1e-4 was best (validation macro-F1 0.931). All models are clearly worse on new people than on validation, and the differences between single models are within the bootstrap intervals.

### Dynamic letters (5-fold cross-validation and 59 held-out test videos)

| Model | CV macro-F1 | Test accuracy |
|---|---|---|
| Random Forest (used in the app) | 0.988 +/- 0.006 | 100% |
| LSTM (32 units, dropout 0.3) | 0.985 +/- 0.030 | 98.3% |
| 1D-CNN (32 filters, dropout 0.3) | 0.981 +/- 0.023 | 100% |
| GRU (32 units, dropout 0.3) | 0.967 +/- 0.031 | 96.6% |
| MLP (256 units, dropout 0.3) | 0.963 +/- 0.039 | 100% |

No difference between the Random Forest and the other families is clearly significant. The Random Forest was chosen because it is the steadiest across folds, trains in seconds and needs no GPU. The dynamic split was random, because the videos carry no signer id, so these numbers may be optimistic for new signers.

### Control gestures

| Model | Threshold | Recall space | Recall delete | False alarms on letters | False alarms on video frames |
|---|---|---|---|---|---|
| HistGradientBoosting (used in the app) | 0.9 | 99.1% | 98.4% | 0.16% | 0% |
| Random Forest | 0.9 | 94.9% | 93.3% | 0% | 0% |

The app uses a lower threshold of 0.83, chosen from `control_threshold_table.csv`, which gives recall of about 99% (space) and 98.5% (delete) with false alarms around 0.2% per frame. The classifier was trained on gestures recorded by 3 people.


## What the app uses

`app/models/` contains only the five files needed at run time:

```
baseline_cnn_fp32.keras
mobilenetv2_fp32.keras
efficientnetb0_fp32.keras
dynamic_rf.joblib
control_gesture_clf.joblib
```

The app reads its settings from `app/app_config.json` (a copy of `models/app_config.json`). The original `.keras` files in this folder are not needed to run the app.

A few details worth knowing:

- **Float32 exports.** The original models were trained on a GPU and exported again in float32 for CPU use. `float32_parity.csv` shows that the baseline CNN and MobileNetV2 give identical predictions after export. EfficientNet-B0 agrees on 93.3% of 300 test images (accuracy 82.7% against 82.0%), so the export was accepted but is not bit-identical.
- **scikit-learn version.** The two `.joblib` files were saved with scikit-learn 1.6.1 and should be loaded with the same version (`pip install scikit-learn==1.6.1`). With another version the app prints a warning and, if loading fails, retrains the two small models from `app/data/` and saves them again. This is why the files in `app/models/` can differ in size from the ones in this folder.
- **Stability setting.** `pipeline_config.json` holds the value used in the simulation (3 stable frames); the app's `app_config.json` starts at 5 for the slower single-frame loop, and the value can be changed with the slider in the app.
- **Dynamic letters** are recorded with a button; the automatic motion router that was tested in `pipeline.ipynb` found only 13 of 59 dynamic test videos and was dropped.

## Load the models

```python
import joblib
from tensorflow import keras

mobilenet = keras.models.load_model('models/mobilenetv2_fp32.keras')   # input: (1, 224, 224, 3), float32, pixel values 0-255
dynamic_rf = joblib.load('models/dynamic_rf.joblib')                   # input: dynamic_features(X, M) from scripts/typing_engine.py
control_clf = joblib.load('models/control_gesture_clf.joblib')         # input: (n, 63) wrist-centred, scaled landmarks
```

Images must be padded to a square (not stretched or cropped) with the padding colour stored in `app_config.json` before they are resized to 224x224. The full pre-processing and the order of the class probabilities (`sorted` static class names, then the four dynamic classes) are implemented in `app/recognizer.py` and `scripts/typing_engine.py`.

## Regenerate

The `.keras` models need a **T4 GPU**. Run the notebooks in this order in Google Colab to regenerate:

1. `static_models.ipynb` trains the three static models and writes the original `.keras` files and the validation tables.
2. `static_evaluation.ipynb` writes the test tables, bootstrap intervals and test predictions.
3. `dynamic_models.ipynb` compares the dynamic models and writes the `dynamic_*` files.
4. `pipeline.ipynb` trains the control-gesture classifier and writes the pipeline results.
5. `app_prep.ipynb` exports the float32 models, measures CPU speed and writes `app_config.json`.

The two `.joblib` models can also be retrained on a CPU in a few seconds by running `app/app.py` after deleting them from `app/models/`.