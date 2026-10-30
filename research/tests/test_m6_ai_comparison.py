import pandas as pd
import pytest

from research.m6_ai_comparison import _aggregate, _deltas


def test_ai_methods_are_included_in_aggregate_and_limited_label_deltas():
    rows = []
    for model, score in [("limited_label", 0.5), ("ai_direct", 0.7), ("ai_hybrid", 0.6)]:
        rows.append({"model": model, "budget": 0.1, "seed": 856, "split": "validation", "macro_f1": score,
                     "cutting_precision": score, "cutting_recall": score,
                     "non_cutting_precision": score, "non_cutting_recall": score})
    metrics = pd.DataFrame(rows)
    assert set(_aggregate(metrics).model) == {"limited_label", "ai_direct", "ai_hybrid"}
    deltas = _deltas(metrics)
    assert deltas.set_index("model").loc["ai_direct", "macro_f1_delta"] == pytest.approx(0.2)
    assert deltas.set_index("model").loc["ai_hybrid", "macro_f1_delta"] == pytest.approx(0.1)
