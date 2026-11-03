# Hand Gesture Recognition — Assignment 2

Streamlit application and reproducible experiments for classifying three hand gestures: `peace`, `stop`, and `thumbs_up`.

## Project overview

Assignment 2 uses an ImageNet-initialized EfficientNetV2B0 classifier. The ArtAug experiment adds 30 manually reviewed images generated with FLUX.1-dev and the ArtAug LoRA through ModelScope. Generation is a separate dataset-preparation step; the classifier and application do not call ModelScope.

## Dataset and splits

The original dataset contains 307 images: 100 `peace`, 104 `stop`, and 103 `thumbs_up`. The 30 ArtAug-enhanced images add 10 images per class, for 337 experiment samples in total. `data/artaug_input/` contains copies of selected original training images and is not an additional set of unique samples.

| Split | Baseline | ArtAug experiment | Data source |
|---|---:|---:|---|
| Training | 196 original | 226 (196 original + 30 ArtAug) | Original training split; approved ArtAug images are training-only |
| Validation | 49 | 49 | Original images only |
| Test | 62 | 62 | Same original images in both experiments |

Both experiments use a stratified two-stage split with seed 42. The original dataset is expected outside this project at `../dataset/`, with `peace/`, `stop/`, and `thumbs_up/` subfolders. It is not included in this project directory; provide it there or pass its location to the baseline training script.

ArtAug images and approval records are stored in:

```text
data/
├── manifest.csv
├── artaug_review.csv
├── artaug_input/{peace,stop,thumbs_up}/
└── artaug_generated/{peace,stop,thumbs_up}/
```

The review CSV records manual approval; the training script validates approval and checks that validation and test remain original-only.

## Models and results

| Experiment | Test accuracy |
|---|---:|
| EfficientNetV2B0 baseline | 98.39% |
| EfficientNetV2B0 + ArtAug | 98.39% |

ArtAug did **not** improve test accuracy in this experiment. Both were evaluated on the same 62 original test images. Assignment 1 reference results are in `results/baseline_reference.csv` and are historical results copied from the Assignment 1 notebook.

The model uses a frozen EfficientNetV2B0 backbone, 128×128 RGB input, ImageNet weights, and an Adam optimizer with learning rate 0.001. The ArtAug experiment uses the same training-time augmentation as the baseline, batch size 16, up to 15 epochs, and EarlyStopping on validation loss with patience 4.

## Application

The Streamlit app supports image upload and camera capture, displays class probabilities, provides a Grad-CAM visualization, and reports mean inference latency (five predictions after one warm-up; Grad-CAM time is excluded). It defaults to `artifacts/efficientnetv2b0_with_augmentation.keras`. To select another checkpoint, set `GESTURE_MODEL_PATH`.

## Setup (Windows PowerShell)

Use Python 3.13 and the versions pinned in `requirements.txt`:

```powershell
cd "C:\path\to\Hand_Gesture_Project\Assignment_2"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The requirements are TensorFlow 2.20.0, Streamlit 1.50.0, NumPy 2.1.3, scikit-learn 1.7.2, Matplotlib 3.10.7, and Pillow 11.3.0.

## Train and evaluate

Run both baseline augmentation ablation arms using the original dataset at `../dataset/`:

```powershell
python scripts\train.py
```

To specify another original dataset location:

```powershell
python scripts\train.py --data-dir "C:\path\to\dataset"
```

Check the ArtAug approvals, files, counts, and fixed test split without training:

```powershell
python scripts\train_artaug.py --check-only
```

Run the separate ArtAug experiment:

```powershell
python scripts\train_artaug.py
```

It reads the original dataset from `../dataset/`, adds only the 30 approved ArtAug images to the original training split, and writes to `artifacts/artaug/` and `results/artaug/`. It refuses to overwrite existing ArtAug outputs unless `--overwrite` is explicitly supplied. Training from ImageNet weights may require downloading those weights on the first run.

Evaluate a saved model on the fixed original test split:

```powershell
python scripts\evaluate.py artifacts\efficientnetv2b0_with_augmentation.keras
```

The evaluator writes to `results/`; use it with care if you want to preserve existing result files.

## Run the application

```powershell
streamlit run app.py
```

To use the ArtAug checkpoint instead of the default baseline model:

```powershell
$env:GESTURE_MODEL_PATH = "artifacts\artaug\efficientnetv2b0_artaug.keras"
streamlit run app.py
```

## Project structure

```text
Assignment_2/
├── .streamlit/config.toml
├── app.py
├── artifacts/
│   ├── efficientnetv2b0_with_augmentation.keras
│   ├── efficientnetv2b0_without_augmentation.keras
│   └── artaug/efficientnetv2b0_artaug.keras
├── data/                         # ArtAug inputs, generated images, manifest, review
├── reports/                      # Report outline, demo outline, cleanup audit
├── results/                      # Metrics, histories, predictions, figures
├── scripts/                      # Train, evaluate, validate, and plot scripts
├── src/                           # Model, preprocessing, inference, explainability
├── tests/
├── requirements.txt
└── README.md
```

`artaug_colab_test.ipynb` is also present as a legacy Colab/DiffSynth workflow. It is not used by the current ModelScope-generated dataset or the training/application commands above.

## Reproducibility and limitations

- The original dataset must be obtained separately and placed at the documented path; it is not included here.
- ArtAug outputs are included in `data/artaug_generated/`, and their approval/provenance is recorded in the manifest and review CSV. The project does not include ModelScope service logs or enough generation-session details to reproduce each image exactly.
- The dataset is small; results may not generalize to new people, cameras, lighting, or backgrounds. ArtAug did not improve test accuracy in this run.
- Class probabilities are not calibrated certainty, and Grad-CAM is an approximate explanation.
- Avoid capturing bystanders. The local app does not save uploaded images; a hosted deployment sends them to its server and should disclose that.
- Check dataset and image redistribution permissions before publishing the images.
