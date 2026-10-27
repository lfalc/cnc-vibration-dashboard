import numpy as np
import pandas as pd
import pytest

from research.m3_baseline import FEATURES
from research.m4_pseudolabel import generate_pseudo_labels


def _frame(prefix, labels):
    rows = []
    for i, label in enumerate(labels):
        row = {"window_id": f"{prefix}{i}", "label": label, "split": "train"}
        row.update({feature: 10 * label + i / 100 for feature in FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def test_confidence_filter_only_emits_high_confidence_labels():
    fit = _frame("f", [0, 1] * 20)
    candidates = _frame("c", [0, 1] * 10)
    accepted, probabilities = generate_pseudo_labels(fit, candidates, confidence=0.8)
    assert len(accepted) > 0
    assert (accepted.pseudo_confidence >= 0.8).all()
    assert set(accepted.label) <= {0, 1}
    assert len(probabilities) == len(candidates)


def test_confidence_and_overlap_guards():
    fit = _frame("f", [0, 1] * 5)
    candidates = _frame("c", [0, 1] * 5)
    with pytest.raises(ValueError, match="between"):
        generate_pseudo_labels(fit, candidates, confidence=0.5)
    with pytest.raises(ValueError, match="overlap"):
        generate_pseudo_labels(fit, fit.copy(), confidence=0.8)
