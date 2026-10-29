from research.m8_finalize import REQUIRED


def test_m8_requires_all_milestone_evidence():
    assert list(REQUIRED) == ["m2", "m3", "m4", "m5", "m6", "m7"]
    assert all(paths for paths in REQUIRED.values())
