"""Cost-aware, provider-backed AI labeling agent.

The module sends visible examples and public candidate features to a configured
model, validates a typed response, caches the request/response, and applies a
local abstention policy. The deterministic prototype agent remains in
``research.m5_agent`` as a comparison method.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib import error, request

import pandas as pd

from research.m3_baseline import FEATURES


AI_AGENT_VERSION = "ai-evidence-agent-v1"
DECISION_SCHEMA_VERSION = "decision-v1"
LABELS = {0, 1}


class Provider(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw JSON response."""


@dataclass(frozen=True)
class AbstentionPolicy:
    top_probability: float = 0.85
    minimum_margin: float = 0.10
    reject_near_transition: bool = True
    max_rows_per_group: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.top_probability <= 1.0:
            raise ValueError("top_probability must be in (0, 1]")
        if not 0.0 <= self.minimum_margin <= 1.0:
            raise ValueError("minimum_margin must be in [0, 1]")
        if self.max_rows_per_group is not None and self.max_rows_per_group < 1:
            raise ValueError("max_rows_per_group must be positive")


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    estimated_usd: float


DECISION_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["decisions"],
    "properties": {"decisions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["window_id", "decision", "label", "probabilities", "rationale"],
        "properties": {
            "window_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["accept", "abstain"]},
            "label": {"type": ["integer", "null"], "enum": [0, 1, None]},
            "probabilities": {"type": "object", "additionalProperties": False,
                "required": ["cutting", "non_cutting", "uncertain"], "properties": {
                    "cutting": {"type": "number"}, "non_cutting": {"type": "number"}, "uncertain": {"type": "number"},
                }},
            "rationale": {"type": "string"},
        },
    }}}
}


@dataclass
class OpenRouterClient:
    """OpenRouter chat client with cache, bounded retries, and JSONL trace."""

    model: str
    cache_dir: Path = Path("research_outputs/ai_cache")
    trace_path: Path | None = Path("research_outputs/ai_traces.jsonl")
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    timeout: float = 120.0
    max_retries: int = 3
    input_usd_per_million: float = 0.01
    output_usd_per_million: float = 0.0
    schema_version: str = DECISION_SCHEMA_VERSION
    dry_run: bool = False

    def _request_payload(self, system: str, user: str) -> dict:
        return {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "vibration_decisions", "strict": True, "schema": DECISION_JSON_SCHEMA}}}

    def _cache_key(self, payload: dict) -> str:
        material = json.dumps({"schema_version": self.schema_version, "payload": payload}, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(material).hexdigest()

    def estimate(self, system: str, user: str) -> CostEstimate:
        input_tokens = max(1, (len(system) + len(user) + 3) // 4)
        output_tokens = 120 * max(1, user.count('"window_id"'))
        cost = (input_tokens * self.input_usd_per_million + output_tokens * self.output_usd_per_million) / 1_000_000
        return CostEstimate(input_tokens, output_tokens, cost)

    def complete(self, system: str, user: str) -> str:
        payload = self._request_payload(system, user)
        key = self._cache_key(payload)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{key}.json"
        estimate = self.estimate(system, user)
        started = time.monotonic()
        cache_hit = cache_path.exists()
        if cache_hit:
            raw = json.loads(cache_path.read_text())["response"]
        elif self.dry_run:
            raise RuntimeError("dry-run cannot produce labels; call estimate() instead")
        else:
            raw = self._post_with_retries(payload)
            cache_path.write_text(json.dumps({"request_hash": key, "response": raw}, sort_keys=True) + "\n")
        self._trace({"request_hash": key, "model": self.model, "schema_version": self.schema_version,
                     "cache_hit": cache_hit, "estimated_input_tokens": estimate.input_tokens,
                     "estimated_output_tokens": estimate.output_tokens, "estimated_usd": estimate.estimated_usd,
                     "latency_seconds": time.monotonic() - started, "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        return raw

    def _post_with_retries(self, payload: dict) -> str:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required for live AI labeling")
        req = request.Request(self.base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(), headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/", "X-Title": "CNC vibration pseudo-labeling thesis"}, method="POST")
        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    result = json.loads(response.read())
                return result["choices"][0]["message"]["content"]
            except error.HTTPError as exc:
                transient = exc.code == 408 or exc.code == 429 or exc.code >= 500
                detail = exc.read().decode(errors="replace")[:500]
                if not transient or attempt >= self.max_retries:
                    raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
            except error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"OpenRouter connection failed: {exc.reason}") from exc
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("OpenRouter response lacks valid message content") from exc
            time.sleep(min(2**attempt, 8))
        raise RuntimeError("OpenRouter retries exhausted")

    def _trace(self, row: dict) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


SYSTEM_PROMPT = """You are a conservative CNC vibration-labeling assistant.
Use only the visible labeled examples and candidate feature values supplied in
the user message. Label 1 as cutting and 0 as non_cutting. The uncertain score
is an abstention score, not a third ground-truth class. Return JSON only using
the required schema. Include exactly one decision for every candidate. Do not
invent IDs. Set label to null when decision is abstain. Do not claim that
non_cutting means idle; it means outside the labeled cutting interval."""


def _records(frame: pd.DataFrame, include_labels: bool) -> list[dict]:
    columns = ["window_id", *FEATURES]
    if include_labels:
        columns.append("label")
    return frame[columns].to_dict(orient="records")


def build_prompt(fit: pd.DataFrame, candidates: pd.DataFrame) -> str:
    """Serialize only allowed evidence; candidate labels are never included."""
    return json.dumps({"label_vocabulary": {"0": "non_cutting", "1": "cutting"},
                       "visible_examples": _records(fit[["window_id", *FEATURES, "label"]], True),
                       "candidates": _records(candidates[["window_id", *FEATURES]], False)}, sort_keys=True)


def _parse_response(raw: str, candidate_ids: set[str]) -> pd.DataFrame:
    try:
        decisions = json.loads(raw)["decisions"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("AI response must be JSON with a decisions array") from exc
    if not isinstance(decisions, list):
        raise ValueError("AI decisions must be a list")
    rows, seen = [], set()
    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("each AI decision must be an object")
        window_id, decision, label, probabilities = str(item.get("window_id", "")), item.get("decision"), item.get("label"), item.get("probabilities")
        if window_id not in candidate_ids or window_id in seen:
            raise ValueError("AI response contains an unknown or duplicate window_id")
        if decision not in {"accept", "abstain"}:
            raise ValueError("AI decision must be accept or abstain")
        if decision == "accept" and label not in LABELS:
            raise ValueError("accepted AI decisions require label 0 or 1")
        if decision == "abstain" and label is not None:
            raise ValueError("abstained AI decisions must have null label")
        if not isinstance(probabilities, dict) or set(probabilities) != {"cutting", "non_cutting", "uncertain"}:
            raise ValueError("AI probabilities must contain cutting, non_cutting, and uncertain")
        scores = {name: float(probabilities[name]) for name in probabilities}
        if any(score < 0 or score > 1 for score in scores.values()) or abs(sum(scores.values()) - 1.0) > 0.02:
            raise ValueError("AI probabilities must be in [0, 1] and sum to 1")
        seen.add(window_id)
        rows.append({"window_id": window_id, "agent_version": AI_AGENT_VERSION, "predicted_label": label,
                     "decision": decision, "probability_cutting": scores["cutting"],
                     "probability_non_cutting": scores["non_cutting"], "probability_uncertain": scores["uncertain"],
                     "rationale": str(item.get("rationale", ""))[:500], "accepted": decision == "accept"})
    if seen != candidate_ids:
        raise ValueError(f"AI response omitted candidate rows: {sorted(candidate_ids - seen)[:5]}")
    return pd.DataFrame(rows)


def apply_policy(decisions: pd.DataFrame, policy: AbstentionPolicy = AbstentionPolicy()) -> pd.DataFrame:
    """Apply score, transition, and optional group-cap gates locally."""
    result = decisions.copy()
    result["top_probability"] = result[["probability_cutting", "probability_non_cutting"]].max(axis=1)
    result["probability_margin"] = (result["probability_cutting"] - result["probability_non_cutting"]).abs()
    result["accepted_by_policy"] = result.accepted & (result.top_probability >= policy.top_probability) & (result.probability_margin >= policy.minimum_margin)
    if policy.reject_near_transition and "near_transition" in result:
        result.loc[result.near_transition.astype(bool), "accepted_by_policy"] = False
    result["decision"] = result["accepted_by_policy"].map({True: "accept", False: "abstain"})
    result.loc[~result.accepted_by_policy, "predicted_label"] = None
    if policy.max_rows_per_group is not None and "dataset_id" in result:
        accepted = result[result.accepted_by_policy].sort_values(["dataset_id", "top_probability"], ascending=[True, False])
        keep = accepted.groupby("dataset_id", sort=False).head(policy.max_rows_per_group).index
        drop = result.index.difference(keep)
        result.loc[drop, ["accepted_by_policy", "predicted_label", "decision"]] = [False, None, "abstain"]
    return result


def label_candidates(fit: pd.DataFrame, candidates: pd.DataFrame, provider: Provider) -> pd.DataFrame:
    """Ask a provider to label candidates and return validated decisions."""
    required_fit, required_candidates = {"window_id", "label", *FEATURES}, {"window_id", *FEATURES}
    if not required_fit.issubset(fit.columns):
        raise ValueError(f"fit missing columns: {sorted(required_fit - set(fit.columns))}")
    if not required_candidates.issubset(candidates.columns):
        raise ValueError(f"candidates missing columns: {sorted(required_candidates - set(candidates.columns))}")
    if set(fit.window_id.astype(str)) & set(candidates.window_id.astype(str)):
        raise ValueError("fit and candidate rows overlap")
    if candidates.window_id.duplicated().any():
        raise ValueError("candidate window_id values must be unique")
    return _parse_response(provider.complete(SYSTEM_PROMPT, build_prompt(fit, candidates)), set(candidates.window_id.astype(str)))


def export_policy(path: Path, policy: AbstentionPolicy, *, model: str, prompt_hash: str) -> None:
    payload = {"policy_version": "policy-v1", "model": model, "prompt_hash": prompt_hash, "policy": asdict(policy)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def import_policy(path: Path) -> AbstentionPolicy:
    payload = json.loads(path.read_text())
    if payload.get("policy_version") != "policy-v1":
        raise ValueError("unsupported policy version")
    return AbstentionPolicy(**payload["policy"])
