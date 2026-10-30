import json

import pandas as pd
import pytest

from research.ai_agent import AI_AGENT_VERSION, AbstentionPolicy, OpenRouterClient, apply_policy, label_candidates
from research.m3_baseline import FEATURES


class ReplayProvider:
    def __init__(self, payload):
        self.payload = json.dumps(payload)
        self.user = None

    def complete(self, system, user):
        self.user = user
        return self.payload


def _frame(prefix, labels):
    rows = []
    for i, label in enumerate(labels):
        row = {"window_id": f"{prefix}{i}", "label": label}
        row.update({feature: float(label + i / 10) for feature in FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def test_ai_decisions_are_validated_and_labels_are_not_sent_for_candidates():
    fit = _frame("fit", [0, 1])
    candidates = _frame("candidate", [0, 1]).drop(columns="label")
    provider = ReplayProvider({"decisions": [
        {"window_id": "candidate0", "decision": "accept", "label": 0, "probabilities": {"cutting": 0.05, "non_cutting": 0.90, "uncertain": 0.05}, "rationale": "near class 0"},
        {"window_id": "candidate1", "decision": "abstain", "label": None, "probabilities": {"cutting": 0.45, "non_cutting": 0.45, "uncertain": 0.10}, "rationale": "ambiguous"},
    ]})
    result = label_candidates(fit, candidates, provider)
    assert result.agent_version.eq(AI_AGENT_VERSION).all()
    assert result.accepted.tolist() == [True, False]
    sent = json.loads(provider.user)
    assert "label" not in sent["candidates"][0]
    assert sent["visible_examples"][0]["label"] in {0, 1}


def test_ai_response_must_cover_every_candidate():
    fit = _frame("fit", [0, 1])
    candidates = _frame("candidate", [0, 1]).drop(columns="label")
    provider = ReplayProvider({"decisions": [
        {"window_id": "candidate0", "decision": "accept", "label": 0, "probabilities": {"cutting": 0.05, "non_cutting": 0.90, "uncertain": 0.05}},
    ]})
    with pytest.raises(ValueError, match="omitted"):
        label_candidates(fit, candidates, provider)


def test_policy_applies_probability_and_transition_gates():
    decisions = pd.DataFrame([
        {"window_id": "a", "accepted": True, "predicted_label": 1, "probability_cutting": 0.90, "probability_non_cutting": 0.05, "near_transition": False},
        {"window_id": "b", "accepted": True, "predicted_label": 0, "probability_cutting": 0.10, "probability_non_cutting": 0.80, "near_transition": True},
    ])
    result = apply_policy(decisions, AbstentionPolicy(top_probability=0.85, minimum_margin=0.10))
    assert result.accepted_by_policy.tolist() == [True, False]
    assert pd.isna(result.loc[1, "predicted_label"])


def test_openrouter_cache_prevents_duplicate_provider_calls(tmp_path):
    client = OpenRouterClient(model="test/model", cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl")
    calls = []
    client._post_with_retries = lambda payload: calls.append(payload) or '{"decisions": []}'
    old_key = __import__("os").environ.get("OPENROUTER_API_KEY")
    __import__("os").environ["OPENROUTER_API_KEY"] = "test-only"
    try:
        assert client.complete("system", "user") == client.complete("system", "user")
    finally:
        if old_key is None:
            __import__("os").environ.pop("OPENROUTER_API_KEY", None)
        else:
            __import__("os").environ["OPENROUTER_API_KEY"] = old_key
    assert len(calls) == 1
    assert len((tmp_path / "trace.jsonl").read_text().splitlines()) == 2
