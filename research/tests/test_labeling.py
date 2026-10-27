import pandas as pd
import pytest

from research.labeling import label_times, load_intervals, union_duration


def intervals():
    return pd.DataFrame({"start": [1.0, 4.0], "end": [3.0, 5.0], "label": [0, 2]})


def test_label_times_uses_left_closed_right_open_intervals():
    result = label_times([0.0, 1.0, 2.999, 3.0, 4.0, 5.0], intervals())
    assert result.binary_state.tolist() == ["non_cutting", "cutting", "cutting", "non_cutting", "cutting", "non_cutting"]
    assert result.anomaly_label.iloc[1] == 0
    assert result.anomaly_label.iloc[4] == 2


def test_union_duration():
    assert union_duration(intervals()) == 3.0


def test_overlapping_intervals_are_rejected(tmp_path):
    path = tmp_path / "label.csv"
    pd.DataFrame({"start": [1, 2], "end": [3, 4], "label": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlapping"):
        load_intervals(path)
