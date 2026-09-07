"""Optimizer, scheduler, and optimizer-step accounting for manual SFT."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler

from agentic_rl.sft.config import OptimizationConfig


class OptimizationSetupError(RuntimeError):
    """Raised when no valid trainable parameter or training step exists."""


@dataclass(frozen=True)
class TrainingSchedule:
    micro_batches_per_epoch: int
    gradient_accumulation_steps: int
    optimizer_steps_per_epoch: int
    total_optimizer_steps: int
    warmup_steps: int


@dataclass(frozen=True)
class OptimizationBundle:
    optimizer: Optimizer
    scheduler: LRScheduler
    schedule: TrainingSchedule
    decay_parameter_names: tuple[str, ...]
    no_decay_parameter_names: tuple[str, ...]


def calculate_training_schedule(
    *,
    micro_batches_per_epoch: int,
    config: OptimizationConfig,
) -> TrainingSchedule:
    if micro_batches_per_epoch <= 0:
        raise OptimizationSetupError("micro_batches_per_epoch must be positive")
    steps_per_epoch = math.ceil(
        micro_batches_per_epoch / config.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = math.ceil(total_steps * config.warmup_ratio)
    return TrainingSchedule(
        micro_batches_per_epoch=micro_batches_per_epoch,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optimizer_steps_per_epoch=steps_per_epoch,
        total_optimizer_steps=total_steps,
        warmup_steps=warmup_steps,
    )


def _parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> tuple[list[dict[str, object]], tuple[str, ...], tuple[str, ...]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias"):
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    if not decay and not no_decay:
        raise OptimizationSetupError("model has no trainable parameters")
    groups: list[dict[str, object]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups, tuple(decay_names), tuple(no_decay_names)


def build_optimizer(
    model: nn.Module,
    config: OptimizationConfig,
) -> tuple[Optimizer, tuple[str, ...], tuple[str, ...]]:
    groups, decay_names, no_decay_names = _parameter_groups(
        model,
        config.weight_decay,
    )
    optimizer = AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    return optimizer, decay_names, no_decay_names


def build_scheduler(
    optimizer: Optimizer,
    config: OptimizationConfig,
    schedule: TrainingSchedule,
) -> LRScheduler:
    scheduler_name = config.scheduler
    if scheduler_name == "constant" and schedule.warmup_steps > 0:
        scheduler_name = "constant_with_warmup"
    return get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=schedule.warmup_steps,
        num_training_steps=schedule.total_optimizer_steps,
    )


def build_optimization(
    model: nn.Module,
    config: OptimizationConfig,
    *,
    micro_batches_per_epoch: int,
) -> OptimizationBundle:
    schedule = calculate_training_schedule(
        micro_batches_per_epoch=micro_batches_per_epoch,
        config=config,
    )
    optimizer, decay_names, no_decay_names = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, schedule)
    return OptimizationBundle(
        optimizer=optimizer,
        scheduler=scheduler,
        schedule=schedule,
        decay_parameter_names=decay_names,
        no_decay_parameter_names=no_decay_names,
    )

