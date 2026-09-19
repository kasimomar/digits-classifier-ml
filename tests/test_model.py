import json
import joblib
import numpy as np
import pytest
from sklearn.datasets import load_digits
from digits_ml.model import split_indices, train, validate_pixels


def test_split_is_disjoint_stratified_and_repeatable():
    labels = load_digits().target
    a, b = split_indices(labels)
    c, d = split_indices(labels)
    assert set(a).isdisjoint(b)
    assert len(a) + len(b) == len(labels)
    assert set(labels[a]) == set(labels[b]) == set(range(10))
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)


@pytest.mark.parametrize("pixels", [[], [0] * 64, [[0] * 63], [[17] * 64], [[-1] * 64], [[float('nan')] * 64], [[float('inf')] * 64], [["no"] * 64]])
def test_invalid_inference_input_is_rejected(pixels):
    with pytest.raises(ValueError):
        validate_pixels(pixels)


def test_valid_batch_preserves_values():
    batch = np.zeros((2, 64))
    batch[0, 0] = 16
    np.testing.assert_array_equal(validate_pixels(batch), batch)


def test_end_to_end_training_and_reload(tmp_path):
    report, model, matrix = train(tmp_path)
    data = load_digits()
    # The fitted scaler must see only the training partition.
    np.testing.assert_allclose(model.steps[0][1].mean_, data.data[report['train_indices']].mean(axis=0))
    assert matrix.sum() == report['test_size'] == 360
    assert report['selected_test']['macro_f1'] > 0.9
    assert report['selected_test']['macro_f1'] > report['baseline_test']['macro_f1']
    assert len(report['errors']) == report['test_size'] - np.trace(matrix)
    restored = joblib.load(tmp_path / 'model.joblib')
    example = validate_pixels(json.loads((tmp_path / 'example.json').read_text()))
    np.testing.assert_array_equal(restored.predict(example), model.predict(example))
    assert json.loads((tmp_path / 'metrics.json').read_text())['selected_model'] == report['selected_model']
