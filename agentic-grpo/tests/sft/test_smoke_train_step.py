from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest
import torch
from torch import nn

from agentic_rl.sft.trainer import TrainerError

TRAINING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = TRAINING_ROOT / "scripts/sft/smoke_train_step.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("smoke_train_step", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_script_help_does_not_load_model() -> None:
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

    assert "without saving weights" in " ".join(result.stdout.split())
    assert "--micro-batches" in result.stdout


def test_shifted_token_count_and_gradient_scaling() -> None:
    module = load_script_module()
    labels = torch.tensor(
        [
            [-100, -100, 2, 3],
            [-100, 5, -100, 7],
        ]
    )
    parameter = nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([8.0, 4.0])

    count = module.shifted_supervised_token_count(labels)
    module.scale_gradients([parameter], denominator=4)

    assert count == 4
    assert torch.equal(parameter.grad, torch.tensor([2.0, 1.0]))


def test_invalid_gradient_denominator_is_rejected() -> None:
    module = load_script_module()
    with pytest.raises(TrainerError, match="denominator"):
        module.scale_gradients([], denominator=0)
