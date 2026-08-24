from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from agentic_rl.sft.config import load_sft_config
from agentic_rl.sft.optimization import (
    OptimizationSetupError,
    build_optimization,
    calculate_training_schedule,
)

TRAINING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)
MODEL_PATH = Path("/Users/projects/models/Qwen3-4B-Instruct-2507")


def optimization_config():
    return load_sft_config(
        CONFIG_PATH,
        environ={"AGENTIC_RL_MODEL_PATH": str(MODEL_PATH)},
    ).optimization


def test_project_training_step_accounting() -> None:
    schedule = calculate_training_schedule(
        micro_batches_per_epoch=256,
        config=optimization_config(),
    )

    assert schedule.gradient_accumulation_steps == 8
    assert schedule.optimizer_steps_per_epoch == 32
    assert schedule.total_optimizer_steps == 96
    assert schedule.warmup_steps == 3


def test_optimizer_uses_only_trainable_parameters_and_scheduler_steps() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    model[0].weight.requires_grad = False
    bundle = build_optimization(
        model,
        optimization_config(),
        micro_batches_per_epoch=16,
    )

    optimized_ids = {
        id(parameter)
        for group in bundle.optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    initial_lr = bundle.optimizer.param_groups[0]["lr"]
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    bundle.optimizer.step()
    bundle.scheduler.step()

    assert optimized_ids == expected_ids
    assert "0.bias" in bundle.no_decay_parameter_names
    assert "1.weight" in bundle.no_decay_parameter_names
    assert bundle.optimizer.param_groups[0]["lr"] != initial_lr


def test_optimizer_rejects_fully_frozen_model() -> None:
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.requires_grad = False

    with pytest.raises(OptimizationSetupError, match="no trainable"):
        build_optimization(
            model,
            optimization_config(),
            micro_batches_per_epoch=1,
        )

