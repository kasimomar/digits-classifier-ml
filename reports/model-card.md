# Model card

## Intended use

An educational, reproducible handwritten-digit benchmark demonstrating preprocessing, model selection, held-out evaluation, model persistence, and validated batch inference. Not validated for production OCR or decisions about people.

## Data and training

Source: `sklearn.datasets.load_digits`, 1,797 examples, ten classes, 64 intensity features in [0,16]. The committed seed-42 split contains 1,437 training rows and 360 held-out rows. The report contains their exact dataset indices.

Two candidates are compared with five-fold stratified training-only cross-validation. Each candidate has a StandardScaler inside its pipeline. Mean macro F1 selects the RBF SVM (`C=3`, `gamma=scale`). The selected pipeline is then fitted to the full training partition. The saved artifact contains this preprocessing and model together; it is not refitted on the test examples.

## Evaluation and errors

Held-out accuracy is 98.0556% and macro F1 is 0.9805. A most-frequent baseline obtains 10% accuracy. The selected model makes seven errors: 8→1, 9→7, 4→7, 9→6, 7→5, 1→4, and 8→4. The per-class report and original example indices are in `metrics.json`; the confusion matrix makes the class-level errors visible.

The 8 class has two errors among 35 held-out examples, and the 9 class has two among 36. The counts are small, so this alone is not evidence of a stable class-specific performance gap. Error inspection should guide hypotheses, not repeated tuning on the same held-out labels.

## Reproducibility

Reference run: Python 3.12.14, scikit-learn 1.7.2, NumPy 2.2.6. Full package versions are pinned in `requirements.txt`. Regenerate with `python -m digits_ml train --output-dir artifacts`. Numerical results may vary slightly with platform libraries. No training-time dataset download or API credential is required.

## Limits

A random image split is not a writer-disjoint split. The benchmark does not test robustness to rotated digits, modern photographs, new handwriting styles, compression, or a shifted class distribution. The classifier always predicts a class; confidence calibration and abstention are not implemented. Performance on this dataset must not be presented as deployment performance.

Only load model artifacts you trust. Model files are deliberately excluded from the repository; recreate them from the source and pinned environment.
