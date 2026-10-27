from vibration_dashboard.config import ALL_FEATURES, AXES, BANDS, MACHINE_NAMES


def test_feature_schema_matches_dataset():
    assert len(ALL_FEATURES) == 28
    assert len(BANDS) == 16
    assert AXES == ["x", "y", "z"]


def test_all_machine_ids_have_display_names():
    assert set(MACHINE_NAMES) == {f"machine_{i:02d}" for i in range(1, 7)}
