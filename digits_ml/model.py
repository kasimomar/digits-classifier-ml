"""Training-only model selection followed by one held-out evaluation."""
from pathlib import Path
import json
import platform

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def split_indices(labels, seed=42):
    return train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed
    )


def validate_pixels(values):
    """Accept a batch of flattened 8x8 images, each with intensities 0..16."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 64 or len(array) == 0:
        raise ValueError("Expected a nonempty array of rows, each containing 64 pixels.")
    if not np.isfinite(array).all() or (array < 0).any() or (array > 16).any():
        raise ValueError("Pixels must be finite numbers between 0 and 16 inclusive.")
    return array


def metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def train(output_dir, seed=42):
    dataset = load_digits()
    train_ids, test_ids = split_indices(dataset.target, seed)
    x_train, y_train = dataset.data[train_ids], dataset.target[train_ids]
    x_test, y_test = dataset.data[test_ids], dataset.target[test_ids]
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale")),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_results = {}
    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, x_train, y_train, cv=folds, scoring="f1_macro", n_jobs=1)
        cv_results[name] = {"fold_macro_f1": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}
    # Selection sees only training-fold scores. Test labels never affect selection.
    selected = max(cv_results, key=lambda name: cv_results[name]["mean"])
    model = candidates[selected].fit(x_train, y_train)
    predicted = model.predict(x_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    matrix = confusion_matrix(y_test, predicted, labels=np.arange(10))
    report = {
        "dataset": "sklearn.datasets.load_digits",
        "seed": seed,
        "train_size": len(train_ids), "test_size": len(test_ids),
        "train_indices": train_ids.tolist(), "test_indices": test_ids.tolist(),
        "selection_metric": "5-fold training-only macro F1",
        "cross_validation": cv_results, "selected_model": selected,
        "baseline_test": metrics(y_test, baseline.predict(x_test)),
        "selected_test": metrics(y_test, predicted),
        "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "errors": [{"dataset_index": int(i), "actual": int(y), "predicted": int(p)}
                   for i, y, p in zip(test_ids, y_test, predicted) if y != p],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    joblib.dump(model, output / "model.joblib")
    (output / "example.json").write_text(json.dumps(x_test[:1].tolist()) + "\n")
    return report, model, matrix


def save_confusion_plot(matrix, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=np.arange(10)).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Held-out digits: selected model\nRows = actual, columns = predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
