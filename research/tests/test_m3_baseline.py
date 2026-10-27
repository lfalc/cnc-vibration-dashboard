import numpy as np
import pandas as pd
import pytest

from research.m3_baseline import FEATURES, fit_predict, select_visible_labels


def _train(n=40):
    rows = []
    for i in range(n):
        label = i % 2
        row = {"window_id": f"w{i}", "split": "train", "label": label}
        row.update({feature: (10 * label + i / 100) for feature in FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def test_synthetic_data_has_separable_signal():
    train = _train()
    scored = train.assign(window_id=lambda x: "s" + x.window_id).iloc[:10].copy()
    predicted, probability = fit_predict(train, scored)
    assert np.array_equal(predicted, scored.label.to_numpy())
    assert np.all((probability >= 0) & (probability <= 1))


def test_selection_is_deterministic_and_seed_specific():
    train = _train()
    first = select_visible_labels(train.rename(columns={"label": "reference_label"}), 0.25, 856)
    second = select_visible_labels(train.rename(columns={"label": "reference_label"}), 0.25, 856)
    third = select_visible_labels(train.rename(columns={"label": "reference_label"}), 0.25, 857)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(third)
    assert set(first.visible_label) == {0, 1}


def test_fit_rejects_overlap_and_non_numeric_metadata_is_not_needed():
    train = _train()
    with pytest.raises(ValueError, match="overlap"):
        fit_predict(train, train.copy())
    scored = train.assign(window_id=lambda x: "s" + x.window_id).iloc[:2].copy()
    scored["identifier_metadata"] = "must not be used"
    predicted, _ = fit_predict(train, scored)
    assert len(predicted) == 2
