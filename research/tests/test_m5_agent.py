import pandas as pd
import pytest

from research.m3_baseline import FEATURES
from research.m5_agent import AGENT_VERSION, agent_label


def _frame(prefix, labels):
    rows = []
    for i, label in enumerate(labels):
        row = {"window_id": f"{prefix}{i}", "label": label, "split": "train"}
        row.update({feature: 10 * label + i / 100 for feature in FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def test_agent_emits_traceable_decisions_and_abstains_when_margin_is_strict():
    fit = _frame("f", [0, 1] * 10)
    candidates = _frame("c", [0, 1] * 5)
    decisions = agent_label(fit, candidates, margin_threshold=0.3)
    assert len(decisions) == len(candidates)
    assert decisions.agent_version.eq(AGENT_VERSION).all()
    assert decisions.decision.isin(["accept", "abstain"]).all()
    assert decisions.accepted.equals(decisions.decision.eq("accept"))
    assert decisions.evidence_margin.notna().all()


def test_agent_rejects_overlap_and_invalid_threshold():
    fit = _frame("f", [0, 1] * 4)
    with pytest.raises(ValueError, match="overlap"):
        agent_label(fit, fit.copy())
    with pytest.raises(ValueError, match="between"):
        agent_label(fit, _frame("c", [0, 1]), margin_threshold=1.0)
