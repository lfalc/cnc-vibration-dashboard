import numpy as np
import pandas as pd

from research.features import extract_features


def test_extract_features_from_chunked_accelerometer_csv(tmp_path):
    dataset = tmp_path / "machine" / "dataset1"
    dataset.mkdir(parents=True)
    time = np.arange(0, 4, 0.25)
    pd.DataFrame({"time": time, "acc0": time, "acc1": -time}).to_csv(dataset / "run_acc.csv", index=False)
    windows = pd.DataFrame(
        {
            "dataset_id": ["machine/dataset1", "machine/dataset1"],
            "window_start": [0.0, 1.0],
            "window_end": [1.0, 2.0],
            "state": ["non_cutting", "cutting"],
        }
    )
    result = extract_features(windows, tmp_path, chunk_size=3)
    assert len(result) == 2
    assert np.isclose(result.loc[0, "acc0_mean"], 0.375)
    assert np.isclose(result.loc[1, "acc0_mean"], 1.375)
    assert set(result.state) == {"non_cutting", "cutting"}
