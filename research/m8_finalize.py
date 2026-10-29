"""Freeze the final M2-M7 research evidence and reproducibility manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED = {
    "m2": ["research_outputs/m2/protocol_summary.json", "research_outputs/m2/public_features.csv", "research_outputs/m2/evaluator_only/reference_labels.csv"],
    "m3": ["research_outputs/m3/experiment_config.json", "research_outputs/m3/per_run_metrics.csv", "research_outputs/m3/frozen_test_metrics.csv"],
    "m4": ["research_outputs/m4/experiment_config.json", "research_outputs/m4/validation_metrics.csv", "research_outputs/m4/frozen_test_metrics.csv", "research_outputs/m4/validation_pseudo_label_quality.csv"],
    "m5": ["research_outputs/m5/experiment_config.json", "research_outputs/m5/validation_metrics.csv", "research_outputs/m5/frozen_test_metrics.csv", "research_outputs/m5/validation_agent_decisions.csv"],
    "m6": ["research_outputs/m6/experiment_config.json", "research_outputs/m6/validation_augmented_pool.csv", "research_outputs/m6/frozen_test_metrics.csv", "research_outputs/m6/frozen_test_deltas.csv"],
    "m7": ["research_outputs/m7/experiment_config.json", "research_outputs/m7/seed_variability.csv", "research_outputs/m7/validation_ablation_metrics.csv", "research_outputs/m7/frozen_test_group_metrics.csv"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict:
    missing = [path for paths in REQUIRED.values() for path in paths if not (root / path).exists()]
    if missing:
        raise FileNotFoundError(f"missing required artifacts: {missing}")
    configs = {}
    files = {}
    for milestone, paths in REQUIRED.items():
        for relative in paths:
            path = root / relative
            files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        config_path = root / f"research_outputs/{milestone}/experiment_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if not config.get("frozen"):
                raise ValueError(f"{milestone} configuration is not frozen")
            configs[milestone] = {"configuration_sha256": config.get("configuration_sha256"), "protocol": config.get("protocol")}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        commit = "unavailable"
    return {"manifest_version": "M8-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "source_commit": commit, "python": sys.version, "platform": platform.platform(), "configs": configs, "files": files}


def finalize(root: Path, output_dir: Path) -> dict:
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "research/tests"], cwd=root, capture_output=True, text=True)
    if test.returncode != 0:
        raise RuntimeError(f"research tests failed:\n{test.stdout}\n{test.stderr}")
    manifest = build_manifest(root)
    manifest["test_command"] = "PYTHONPATH=. pytest -q research/tests"
    manifest["test_result"] = test.stdout.strip()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"milestones": list(manifest["configs"]), "artifact_count": len(manifest["files"]), "test_result": test.stdout.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("research_outputs/m8"))
    args = parser.parse_args()
    print(json.dumps(finalize(args.root.resolve(), args.output), indent=2))


if __name__ == "__main__":
    main()
