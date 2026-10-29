import pandas as pd
import pytest

from research.m7_reliability import _correlations


def test_coverage_correlation_is_computed_per_method_and_budget():
    quality = pd.DataFrame({"model": ["prototype_agent"] * 3, "budget": [0.1] * 3, "seed": [1, 2, 3], "coverage": [0.1, 0.2, 0.3], "pseudo_macro_f1": [0.9, 0.8, 0.7]})
    deltas = pd.DataFrame({"model": ["prototype_agent"] * 3, "budget": [0.1] * 3, "seed": [1, 2, 3], "macro_f1_delta": [0.1, 0.2, 0.3]})
    result = _correlations(quality, deltas)
    assert len(result) == 1
    assert result.coverage_downstream_delta_pearson.iloc[0] == pytest.approx(1.0)
