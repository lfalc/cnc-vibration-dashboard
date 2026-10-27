import pandas as pd

from preprocess import compute_daily_stats, compute_feature_bins, compute_spectral_bins


def sample_frame():
    return pd.DataFrame({
        "time": pd.to_datetime([
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:00:05Z",
            "2025-01-02T00:00:00Z",
        ]),
        "x_rms": [1.0, 3.0, 5.0],
        "x_band_01_db": [10.0, None, 20.0],
    })


def test_daily_stats_preserve_nan_counts():
    result = compute_daily_stats(sample_frame())
    first_day = result.iloc[0]
    assert first_day["x_rms_median"] == 2.0
    assert first_day["x_rms_count"] == 2
    assert first_day["x_band_01_db_count"] == 1


def test_minute_bins_are_time_indexed_medians():
    result = compute_feature_bins(sample_frame())
    assert list(result.columns) == ["time_bin", "x_rms", "x_band_01_db"]
    assert result.loc[0, "x_rms"] == 2.0


def test_spectral_bins_include_recorded_band_columns_only():
    result = compute_spectral_bins(sample_frame())
    assert "x_band_01_db" in result.columns
    assert "y_band_01_db" not in result.columns
