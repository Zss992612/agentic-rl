from __future__ import annotations

import subprocess
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = TRAINING_ROOT / "scripts/sft/train_sft.py"


def test_train_entrypoint_help_does_not_load_model() -> None:
    result = subprocess.run(
        [
            str(TRAINING_ROOT / ".venv/bin/python"),
            str(SCRIPT_PATH),
            "--help",
        ],
        cwd=TRAINING_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )

    normalized = " ".join(result.stdout.split())
    assert "assistant-only LoRA SFT" in normalized
    assert "--resume-from" in result.stdout
    assert "--disable-wandb" in result.stdout

