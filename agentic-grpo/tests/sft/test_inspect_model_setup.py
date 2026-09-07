from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = TRAINING_ROOT / "scripts/sft/inspect_model_setup.py"
MODEL_PATH = Path("/Users/projects/models/Qwen3-4B-Instruct-2507")


def test_metadata_only_model_setup_audit_does_not_load_weights() -> None:
    environment = os.environ.copy()
    environment["AGENTIC_RL_MODEL_PATH"] = str(MODEL_PATH)
    result = subprocess.run(
        [
            str(TRAINING_ROOT / ".venv/bin/python"),
            str(SCRIPT_PATH),
            "--metadata-only",
            "--skip-dataset-hash-verification",
        ],
        cwd=TRAINING_ROOT.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert report["mode"] == "metadata_only"
    assert report["dataset"]["trajectories"] == 256
    assert report["architecture"]["model_type"] == "qwen3"
    assert report["tokenizer_contract"]["maximum_dataset_token_id"] == 151_666
    assert "runtime" not in report
    assert "lora" not in report
    assert "model weights were not loaded" in result.stderr

