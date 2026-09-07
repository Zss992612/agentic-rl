"""Atomic LoRA checkpoint persistence and exact single-device resume."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors

from agentic_rl.preprocessing.storage import sha256_file
from agentic_rl.sft.config import SFTConfig, save_resolved_config
from agentic_rl.sft.optimization import OptimizationBundle
from agentic_rl.sft.trainer import (
    OptimizerStepMetrics,
    SFTTrainer,
    SFTTrainerCallback,
    TrainerState,
)

CHECKPOINT_FORMAT = "agentic_rl_sft_lora_checkpoint"
CHECKPOINT_VERSION = 1
_STEP_DIRECTORY = re.compile(r"^step_(\d{8})$")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be safely saved or restored."""


def _unwrapped_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class CheckpointManager:
    def __init__(
        self,
        *,
        config: SFTConfig,
        run_dir: str | Path | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir or config.experiment.run_dir).expanduser().resolve()
        self.checkpoint_root = self.run_dir / "checkpoints"
        self.run_metadata = dict(run_metadata or {})
        self.last_checkpoint: Path | None = None

    def prepare_run(self) -> None:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        resolved_config = self.run_dir / "resolved_config.yaml"
        save_resolved_config(self.config, resolved_config)
        manifest = {
            "format": "agentic_rl_sft_run",
            "experiment": self.config.experiment.name,
            "resolved_config": resolved_config.name,
            "resolved_config_sha256": sha256_file(resolved_config),
            "metadata": self.run_metadata,
        }
        (self.run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _checkpoint_directories(self) -> list[tuple[int, Path]]:
        if not self.checkpoint_root.is_dir():
            return []
        checkpoints: list[tuple[int, Path]] = []
        for child in self.checkpoint_root.iterdir():
            match = _STEP_DIRECTORY.fullmatch(child.name)
            if match and child.is_dir():
                checkpoints.append((int(match.group(1)), child))
        return sorted(checkpoints)

    def latest_checkpoint(self) -> Path | None:
        checkpoints = self._checkpoint_directories()
        return checkpoints[-1][1] if checkpoints else None

    def _prune(self) -> None:
        checkpoints = self._checkpoint_directories()
        for _step, path in checkpoints[: -self.config.checkpoint.keep_last]:
            manifest_path = path / "checkpoint_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("format") == CHECKPOINT_FORMAT
                and manifest.get("experiment") == self.config.experiment.name
            ):
                shutil.rmtree(path)

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimization: OptimizationBundle,
        trainer_state: TrainerState,
    ) -> Path:
        self.prepare_run()
        step = trainer_state.optimizer_step
        target = self.checkpoint_root / f"step_{step:08d}"
        if target.is_dir():
            manifest_path = target / "checkpoint_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("format") == CHECKPOINT_FORMAT
                    and manifest.get("optimizer_step") == step
                ):
                    self.last_checkpoint = target
                    return target
            raise CheckpointError(f"checkpoint directory already exists: {target}")

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=self.checkpoint_root)
        )
        try:
            save_model = _unwrapped_model(model)
            save_pretrained = getattr(save_model, "save_pretrained", None)
            if save_pretrained is None:
                raise CheckpointError("model does not support save_pretrained")
            save_pretrained(temporary, safe_serialization=True)
            (temporary / "trainer_state.json").write_text(
                json.dumps(trainer_state.state_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
            if self.config.checkpoint.save_optimizer:
                torch.save(optimization.optimizer.state_dict(), temporary / "optimizer.pt")
                torch.save(optimization.scheduler.state_dict(), temporary / "scheduler.pt")
            torch.save(capture_rng_state(), temporary / "rng_state.pt")
            files = sorted(
                path.name
                for path in temporary.iterdir()
                if path.is_file() and path.name != "checkpoint_manifest.json"
            )
            manifest = {
                "format": CHECKPOINT_FORMAT,
                "version": CHECKPOINT_VERSION,
                "experiment": self.config.experiment.name,
                "optimizer_step": step,
                "trainer_state": trainer_state.state_dict(),
                "save_optimizer": self.config.checkpoint.save_optimizer,
                "files": files,
            }
            (temporary / "checkpoint_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        self.last_checkpoint = target
        self._prune()
        return target

    def load(
        self,
        checkpoint: str | Path,
        *,
        model: torch.nn.Module,
        optimization: OptimizationBundle,
        restore_rng: bool = True,
    ) -> TrainerState:
        path = Path(checkpoint).expanduser().resolve()
        manifest_path = path / "checkpoint_manifest.json"
        if not manifest_path.is_file():
            raise CheckpointError(f"checkpoint manifest does not exist: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != CHECKPOINT_FORMAT:
            raise CheckpointError("unsupported checkpoint format")
        if manifest.get("version") != CHECKPOINT_VERSION:
            raise CheckpointError("unsupported checkpoint version")
        if manifest.get("experiment") != self.config.experiment.name:
            raise CheckpointError("checkpoint experiment does not match config")

        adapter_path = path / "adapter_model.safetensors"
        if not adapter_path.is_file():
            raise CheckpointError("checkpoint has no adapter_model.safetensors")
        adapter_state = load_safetensors(adapter_path, device="cpu")
        result = set_peft_model_state_dict(
            _unwrapped_model(model),
            adapter_state,
            adapter_name="default",
        )
        unexpected = list(getattr(result, "unexpected_keys", []))
        if unexpected:
            raise CheckpointError(
                "unexpected adapter keys while restoring: " + ", ".join(unexpected)
            )

        if not manifest.get("save_optimizer"):
            raise CheckpointError(
                "checkpoint has no optimizer state and cannot exactly resume training"
            )
        optimizer_path = path / "optimizer.pt"
        scheduler_path = path / "scheduler.pt"
        if not optimizer_path.is_file() or not scheduler_path.is_file():
            raise CheckpointError("checkpoint optimizer or scheduler state is missing")
        optimization.optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=True)
        )
        optimization.scheduler.load_state_dict(
            torch.load(scheduler_path, map_location="cpu", weights_only=True)
        )
        state = TrainerState.from_state_dict(
            json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        )
        if restore_rng:
            rng_state = torch.load(
                path / "rng_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            restore_rng_state(rng_state)
        self.last_checkpoint = path
        return state


class CheckpointCallback(SFTTrainerCallback):
    def __init__(self, manager: CheckpointManager) -> None:
        self.manager = manager

    def on_train_start(self, trainer: SFTTrainer) -> None:
        del trainer
        self.manager.prepare_run()

    def on_optimizer_step(
        self,
        trainer: SFTTrainer,
        metrics: OptimizerStepMetrics,
    ) -> None:
        if metrics.optimizer_step % trainer.config.checkpoint.save_steps == 0:
            self.manager.save(
                model=trainer.model,
                optimization=trainer.optimization,
                trainer_state=trainer.state,
            )

    def on_train_end(self, trainer: SFTTrainer) -> None:
        if trainer.state.optimizer_step <= 0:
            return
        if (
            self.manager.last_checkpoint is None
            or self.manager.last_checkpoint.name
            != f"step_{trainer.state.optimizer_step:08d}"
        ):
            self.manager.save(
                model=trainer.model,
                optimization=trainer.optimization,
                trainer_state=trainer.state,
            )

