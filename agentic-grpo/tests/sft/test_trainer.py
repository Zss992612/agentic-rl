from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from agentic_rl.sft.config import load_sft_config
from agentic_rl.sft.optimization import build_optimization
from agentic_rl.sft.runtime import configure_runtime
from agentic_rl.sft.trainer import (
    OptimizerStepMetrics,
    SFTTrainer,
    SFTTrainerCallback,
    TrainerError,
    TrainerState,
)

TRAINING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)
MODEL_PATH = Path("/Users/projects/models/Qwen3-4B-Instruct-2507")


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids, attention_mask, labels):
        del attention_mask
        logits = self.lm_head(self.embedding(input_ids))
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)


class ToyDataPipeline:
    def __init__(self, batches):
        self.dataloader = batches
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class RecordingCallback(SFTTrainerCallback):
    def __init__(self) -> None:
        self.started = 0
        self.ended = 0
        self.epochs: list[int] = []
        self.steps: list[OptimizerStepMetrics] = []
        self.completed_epochs_at_step: list[int] = []

    def on_train_start(self, trainer: SFTTrainer) -> None:
        del trainer
        self.started += 1

    def on_optimizer_step(
        self,
        trainer: SFTTrainer,
        metrics: OptimizerStepMetrics,
    ) -> None:
        self.steps.append(metrics)
        self.completed_epochs_at_step.append(trainer.state.completed_epochs)

    def on_epoch_end(self, trainer: SFTTrainer, epoch: int) -> None:
        del trainer
        self.epochs.append(epoch)

    def on_train_end(self, trainer: SFTTrainer) -> None:
        del trainer
        self.ended += 1


def batch(tokens: list[int], supervised_positions: list[int]):
    input_ids = torch.tensor([tokens], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    for position in supervised_positions:
        labels[0, position] = input_ids[0, position]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
    }


def trainer_components(epochs: int = 2):
    config = load_sft_config(
        CONFIG_PATH,
        environ={"AGENTIC_RL_MODEL_PATH": str(MODEL_PATH)},
    )
    config = replace(
        config,
        model=replace(
            config.model,
            dtype="float32",
            attention_implementation="eager",
            gradient_checkpointing=False,
        ),
        data=replace(config.data, pin_memory=False, num_workers=0),
        optimization=replace(
            config.optimization,
            epochs=epochs,
            gradient_accumulation_steps=2,
            learning_rate=1e-2,
            warmup_ratio=0.0,
        ),
        runtime=replace(
            config.runtime,
            device="cpu",
            tf32=False,
            deterministic=False,
        ),
    )
    batches = [
        batch([1, 2, 3, 4, 5], [2, 3]),
        batch([2, 3, 4, 5, 6], [1, 2, 3, 4]),
        batch([3, 4, 5, 6, 7], [3]),
        batch([4, 5, 6, 7, 8], [2, 3, 4]),
        batch([5, 6, 7, 8, 9], [1, 4]),
    ]
    pipeline = ToyDataPipeline(batches)
    model = TinyCausalLM()
    runtime = configure_runtime(
        config.runtime,
        config.model,
        seed=config.experiment.seed,
    )
    optimization = build_optimization(
        model,
        config.optimization,
        micro_batches_per_epoch=len(batches),
    )
    return config, model, pipeline, runtime, optimization


def test_trainer_runs_epochs_full_windows_and_tail_windows() -> None:
    config, model, pipeline, runtime, optimization = trainer_components()
    callback = RecordingCallback()
    before = model.lm_head.weight.detach().clone()
    trainer = SFTTrainer(
        config=config,
        model=model,
        data_pipeline=pipeline,  # type: ignore[arg-type]
        optimization=optimization,
        runtime=runtime,
        callbacks=[callback],
    )

    state = trainer.train()

    assert state.completed_epochs == 2
    assert state.micro_step == 10
    assert state.optimizer_step == 6
    assert state.micro_batches_in_current_epoch == 0
    assert state.samples_seen == 10
    assert state.input_tokens_seen == 50
    assert state.supervised_tokens_seen == 24
    assert pipeline.epochs == [0, 1]
    assert callback.started == 1
    assert callback.ended == 1
    assert callback.epochs == [0, 1]
    assert [step.micro_steps_in_window for step in callback.steps] == [2, 2, 1] * 2
    assert [step.supervised_tokens_in_window for step in callback.steps] == [6, 4, 2] * 2
    assert callback.completed_epochs_at_step == [0, 0, 1, 1, 1, 2]
    assert all(math_is_finite(step.token_normalized_loss) for step in callback.steps)
    assert all(math_is_finite(step.gradient_norm_before_clip) for step in callback.steps)
    assert not torch.equal(before, model.lm_head.weight)


def math_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def test_max_optimizer_steps_stops_without_marking_partial_epoch_complete() -> None:
    config, model, pipeline, runtime, optimization = trainer_components()
    trainer = SFTTrainer(
        config=config,
        model=model,
        data_pipeline=pipeline,  # type: ignore[arg-type]
        optimization=optimization,
        runtime=runtime,
    )

    state = trainer.train(max_optimizer_steps=1)

    assert state.optimizer_step == 1
    assert state.micro_step == 2
    assert state.completed_epochs == 0
    assert state.micro_batches_in_current_epoch == 2

    resumed = trainer.train(max_optimizer_steps=2)

    assert resumed.optimizer_step == 2
    assert resumed.micro_step == 4
    assert resumed.micro_batches_in_current_epoch == 4


def test_trainer_state_round_trip_and_validation() -> None:
    state = TrainerState(
        completed_epochs=1,
        current_epoch=1,
        micro_step=10,
        optimizer_step=5,
        micro_batches_in_current_epoch=0,
        samples_seen=10,
        input_tokens_seen=100,
        supervised_tokens_seen=30,
    )

    assert TrainerState.from_state_dict(state.state_dict()) == state
    with pytest.raises(TrainerError, match="missing"):
        TrainerState.from_state_dict({"optimizer_step": 1})
