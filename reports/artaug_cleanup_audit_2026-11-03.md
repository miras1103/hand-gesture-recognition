# ArtAug cleanup and dataset preparation audit

Audit date: 2026-11-03

## Cleanup outcome

Removed one confirmed obsolete Colab smoke-test notebook:

- `artaug_colab_test.backup_2026-11-03.ipynb` — the old single-image smoke test used `%pip install --upgrade`, `FluxPipeline.from_pretrained()`, and Diffusers `load_lora_weights()`. This is the previously failed full-pipeline loading approach; it is superseded by the current component-based Drive/NF4 notebook.

After confirmation, removed the three remaining Colab notebooks because the project now uses ModelScope online and no longer needs a local Colab generation workflow:

- `artaug_colab_test.ipynb` — previous Drive/NF4 Colab workflow.
- `artaug_colab_test.backup_2026-11-03_audit2.ipynb` — earlier audit backup of that workflow.
- `artaug_colab_test.backup_2026-11-03_drive.ipynb` — Drive-persistence backup of that workflow.

During the subsequent training preparation, an `artaug_colab_test.ipynb` appeared again (created 2026-11-03 22:40); it was inspected and not deleted in that task. It is not required by the training workflow.

No failed, partial, or zero-byte ArtAug outputs were found. All 30 existing `*_artaug.png` images are retained. Python bytecode and `.pytest_cache` files were left in place because they are unrelated to the old Colab loading attempts; no cache cleanup was performed. No other deletion candidates were confirmed.

## Preserve

- Original dataset `..\dataset` (100 `peace`, 104 `stop`, 103 `thumbs_up` images), `dataset.zip`, and the Assignment 1 notebook.
- All 30 training-only source images under `data\artaug_input\`.
- All 30 existing ArtAug outputs under `data\artaug_generated\` (10 per class).
- Existing `data\manifest.csv` and its 30 approved original training records.
- Both EfficientNet checkpoints, all metrics, predictions, metadata, histories, figures, application code, existing training/evaluation scripts, tests, reports, documentation, and Streamlit config.
- `.venv`; it is the current project environment, not an ArtAug download cache.

## Uncertain-purpose files

No uncertain or incomplete image files were identified. The removed notebooks were all confirmed Colab workflows or backups of that workflow; the smoke-test notebook's obsolete loading code was also inspected directly.

## ArtAug technical validation

`scripts\validate_artaug.py` checked the generated images for decodability, dimensions, supported image format/extension, expected class folder, `_artaug` naming, an approved training-source image in the manifest, duplicate names/content, and exact pixel duplicates against the original dataset and other ArtAug outputs.

| Class | Files found | Technical checks passed | Manual review pending | Accepted |
|---|---:|---:|---:|---:|
| `peace` | 10 | 10 | 0 | 10 |
| `stop` | 10 | 10 | 0 | 10 |
| `thumbs_up` | 10 | 10 | 0 | 10 |
| **Total** | **30** | **30** | **0** | **30** |

The project owner manually reviewed and approved all 30 images. No byte-identical or pixel-identical outputs were found among the original dataset or the generated images. Technical checks alone do not verify gesture identity or visual quality. The decisions are recorded in `data\artaug_review.csv` and `data\manifest.csv`.

## Split and experiment status

`data\manifest.csv` contains the 30 original plus 30 approved ArtAug `train` records. `scripts\train_artaug.py` preserves the seed-42 split: 226 train (196 original plus 30 ArtAug), 49 original-only validation, and 62 original-only test. Before training, it verified that its test paths exactly match the saved original baseline predictions.

The separate EfficientNetV2B0 ArtAug experiment completed 15 epochs with the baseline settings and saved its checkpoint under `artifacts\artaug\` and metrics, predictions, metadata, classification report, confusion matrix, and training curves under `results\artaug\`. Its test accuracy was 98.39% and weighted F1 was 98.39%, matching the existing with-augmentation baseline on the same 62 test samples. No performance improvement was observed in this run. Original baseline artifacts and results were preserved.
