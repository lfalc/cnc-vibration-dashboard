import pandas as pd
import pytest

from research.m6_comparison import _deltas


def test_m6_deltas_are_matched_by_budget_seed_and_split():
    rows = []
    for model, score in [("limited_label", 0.5), ("self_training", 0.6), ("prototype_agent", 0.4)]:
        rows.append({"model": model, "budget": 0.1, "seed": 856, "split": "validation", "macro_f1": score})
    result = _deltas(pd.DataFrame(rows)).set_index("model")
    assert result.loc["self_training", "macro_f1_delta"] == pytest.approx(0.1)
    assert result.loc["prototype_agent", "macro_f1_delta"] == pytest.approx(-0.1)
