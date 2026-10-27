import pandas as pd

from research.windowing import make_windows


def test_windowing_marks_transitions_and_preserves_split():
    datasets = pd.DataFrame({"dataset_id": ["m/d1"], "timeline_seconds": [6.0], "split": ["train"]})
    intervals = pd.DataFrame(
        {"dataset_id": ["m/d1"], "start": [2.0], "end": [4.0], "label": [1]}
    )
    windows = make_windows(datasets, intervals, window_seconds=2.0, stride_seconds=1.0)
    states = dict(zip(windows.window_start, windows.state))
    assert states == {0.0: "non_cutting", 1.0: "transition", 2.0: "cutting", 3.0: "transition", 4.0: "non_cutting"}
    assert windows[windows.state == "cutting"].source_label.tolist() == [1]
    assert set(windows.split) == {"train"}


def test_windowing_does_not_force_mixed_source_labels():
    datasets = pd.DataFrame({"dataset_id": ["m/d1"], "timeline_seconds": [4.0]})
    intervals = pd.DataFrame(
        {
            "dataset_id": ["m/d1", "m/d1"],
            "start": [0.0, 1.0],
            "end": [1.0, 2.0],
            "label": [0, 2],
        }
    )
    windows = make_windows(datasets, intervals, window_seconds=2.0, stride_seconds=2.0)
    assert windows.iloc[0].state == "cutting"
    assert pd.isna(windows.iloc[0].source_label)
