"""Console and Weights & Biases callbacks for the manual SFT trainer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import wandb

from agentic_rl.sft.config import SFTConfig
from agentic_rl.sft.trainer import (
    OptimizerStepMetrics,
    SFTTrainer,
    SFTTrainerCallback,
)


class TrackingError(RuntimeError):
    """Raised when an experiment tracking run cannot be initialized."""


def _gpu_metrics() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    mebibyte = 1024**2
    return {
        "gpu/memory_allocated_mib": torch.cuda.memory_allocated(device) / mebibyte,
        "gpu/memory_reserved_mib": torch.cuda.memory_reserved(device) / mebibyte,
        "gpu/max_memory_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / mebibyte
        ),
        "gpu/max_memory_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / mebibyte
        ),
    }


def step_metrics_payload(
    trainer: SFTTrainer,
    metrics: OptimizerStepMetrics,
) -> dict[str, float | int]:
    elapsed = max(metrics.elapsed_seconds, 1e-12)
    return {
        "train/optimizer_step": metrics.optimizer_step,
        "train/micro_step": trainer.state.micro_step,
        "train/epoch": metrics.epoch + 1,
        "train/loss": metrics.token_normalized_loss,
        "train/learning_rate_used": metrics.learning_rate_used,
        "train/learning_rate_next": metrics.learning_rate_next,
        "train/gradient_norm": metrics.gradient_norm_before_clip,
        "train/micro_batches_in_window": metrics.micro_steps_in_window,
        "train/samples_in_window": metrics.samples_in_window,
        "train/input_tokens_in_window": metrics.input_tokens_in_window,
        "train/supervised_tokens_in_window": metrics.supervised_tokens_in_window,
        "train/input_tokens_per_second": metrics.input_tokens_in_window / elapsed,
        "train/supervised_tokens_per_second": (
            metrics.supervised_tokens_in_window / elapsed
        ),
        "train/step_seconds": metrics.elapsed_seconds,
        "train/total_samples": trainer.state.samples_seen,
        "train/total_input_tokens": trainer.state.input_tokens_seen,
        "train/total_supervised_tokens": trainer.state.supervised_tokens_seen,
    }


class ConsoleMetricsCallback(SFTTrainerCallback):
    def __init__(self, *, log_steps: int = 1) -> None:
        if log_steps <= 0:
            raise TrackingError("console log_steps must be positive")
        self.log_steps = log_steps

    def on_optimizer_step(
        self,
        trainer: SFTTrainer,
        metrics: OptimizerStepMetrics,
    ) -> None:
        if metrics.optimizer_step % self.log_steps != 0:
            return
        payload = step_metrics_payload(trainer, metrics)
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


class WandbMetricsCallback(SFTTrainerCallback):
    """Log Trainer metrics while letting W&B monitor GPU utilization itself."""

    def __init__(
        self,
        *,
        config: SFTConfig,
        run_dir: str | Path,
        run_metadata: Mapping[str, Any] | None = None,
        wandb_module: Any = wandb,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_metadata = dict(run_metadata or {})
        self._wandb = wandb_module
        self._run: Any | None = None
        self._finished = False

    def _run_id(self) -> str:
        id_path = self.run_dir / "wandb_run_id.txt"
        if id_path.is_file():
            run_id = id_path.read_text(encoding="utf-8").strip()
            if run_id:
                return run_id
        run_id = self._wandb.util.generate_id()
        id_path.write_text(run_id + "\n", encoding="utf-8")
        return run_id

    def on_train_start(self, trainer: SFTTrainer) -> None:
        if self._run is not None:
            raise TrackingError("W&B run is already initialized")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        init_config = self.config.to_serializable_dict()
        init_config["run_metadata"] = self.run_metadata
        try:
            self._run = self._wandb.init(
                project=self.config.tracking.wandb.project,
                entity=self.config.tracking.wandb.entity,
                name=self.config.experiment.name,
                id=self._run_id(),
                resume="allow",
                dir=str(self.run_dir),
                config=init_config,
            )
        except Exception as error:
            raise TrackingError(f"failed to initialize W&B: {error}") from error
        if self._run is None:
            raise TrackingError("wandb.init returned no run")
        self._run.define_metric("train/optimizer_step")
        self._run.define_metric("train/*", step_metric="train/optimizer_step")
        self._run.define_metric("gpu/*", step_metric="train/optimizer_step")
        if self.config.tracking.wandb.watch_model:
            self._wandb.watch(
                trainer.model,
                log="gradients",
                log_freq=self.config.tracking.wandb.log_steps,
            )

    def on_optimizer_step(
        self,
        trainer: SFTTrainer,
        metrics: OptimizerStepMetrics,
    ) -> None:
        if self._run is None:
            raise TrackingError("W&B run has not been initialized")
        if metrics.optimizer_step % self.config.tracking.wandb.log_steps != 0:
            return
        payload = step_metrics_payload(trainer, metrics)
        payload.update(_gpu_metrics())
        self._run.log(payload)

    def on_train_end(self, trainer: SFTTrainer) -> None:
        del trainer
        self.finish(exit_code=0)

    def finish(self, *, exit_code: int) -> None:
        if self._run is not None and not self._finished:
            self._run.finish(exit_code=exit_code)
            self._finished = True
