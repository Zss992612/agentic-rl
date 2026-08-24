"""Formal entry point for manual Qwen LoRA supervised fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from agentic_rl.sft.checkpoint import CheckpointCallback, CheckpointManager
from agentic_rl.sft.config import load_sft_config
from agentic_rl.sft.dataloader import build_train_dataloader
from agentic_rl.sft.lora import apply_lora
from agentic_rl.sft.modeling import (
    load_architecture_config,
    load_base_model,
    load_tokenizer,
    validate_tokenizer_contract,
)
from agentic_rl.sft.optimization import build_optimization
from agentic_rl.sft.runtime import configure_runtime
from agentic_rl.sft.tracking import ConsoleMetricsCallback, WandbMetricsCallback
from agentic_rl.sft.trainer import SFTTrainer, TrainerState

TRAINING_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the configured causal LM with assistant-only LoRA SFT."
    )
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Override checkpoint.resume_from with a checkpoint directory",
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        help="Absolute optimizer-step ceiling for a bounded run",
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        help="Optionally round dynamic padding width to this multiple",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B for this run while retaining console metrics",
    )
    parser.add_argument(
        "--skip-dataset-hash-verification",
        action="store_true",
        help="Skip dataset hashes while retaining shape and label checks",
    )
    return parser.parse_args()


def status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main() -> None:
    args = parse_args()
    if args.max_optimizer_steps is not None and args.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be positive")
    if args.pad_to_multiple_of is not None and args.pad_to_multiple_of <= 0:
        raise ValueError("--pad-to-multiple-of must be positive")

    status(f"Loading SFT config: {args.config.resolve()}")
    config = load_sft_config(args.config)
    if args.resume_from is not None:
        config = replace(
            config,
            checkpoint=replace(
                config.checkpoint,
                resume_from=args.resume_from.expanduser().resolve(),
            ),
        )
    if args.disable_wandb:
        config = replace(
            config,
            tracking=replace(
                config.tracking,
                wandb=replace(config.tracking.wandb, enabled=False),
            ),
        )

    status("Configuring CUDA runtime and deterministic state")
    runtime = configure_runtime(
        config.runtime,
        config.model,
        seed=config.experiment.seed,
    )
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)

    status("Loading tokenizer, model metadata, and training dataset")
    tokenizer = load_tokenizer(config.model)
    architecture = load_architecture_config(config.model)
    if tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer has no pad_token_id")
    data_pipeline = build_train_dataloader(
        config,
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=args.pad_to_multiple_of,
        verify_hashes=not args.skip_dataset_hash_verification,
    )
    tokenizer_contract = validate_tokenizer_contract(
        tokenizer,
        architecture,
        data_pipeline.dataset,
        model_path=config.model.path,
    )

    status(f"Loading base model on {runtime.device}")
    model = load_base_model(config.model, device=runtime.device)
    status("Injecting LoRA adapters")
    model, lora_report = apply_lora(
        model,
        config.lora,
        gradient_checkpointing=config.model.gradient_checkpointing,
    )
    optimization = build_optimization(
        model,
        config.optimization,
        micro_batches_per_epoch=len(data_pipeline.dataloader),
    )

    run_metadata: dict[str, Any] = {
        "tokenizer_contract": asdict(tokenizer_contract),
        "dataset_manifest_sha256": data_pipeline.dataset.manifest_sha256,
        "dataset_trajectories": len(data_pipeline.dataset),
        "dataset_total_tokens": data_pipeline.dataset.total_tokens,
        "dataset_supervised_tokens": data_pipeline.dataset.total_supervised_tokens,
        "lora_matched_modules": lora_report.matched_modules,
        "total_parameters": lora_report.total_parameters,
        "trainable_parameters": lora_report.trainable_parameters,
        "trainable_percentage": lora_report.trainable_fraction * 100,
        "training_schedule": asdict(optimization.schedule),
    }
    checkpoint_manager = CheckpointManager(
        config=config,
        run_metadata=run_metadata,
    )
    trainer_state = TrainerState()
    if config.checkpoint.resume_from is not None:
        status(f"Restoring checkpoint: {config.checkpoint.resume_from}")
        trainer_state = checkpoint_manager.load(
            config.checkpoint.resume_from,
            model=model,
            optimization=optimization,
            restore_rng=True,
        )

    if runtime.compile_enabled:
        status("Applying torch.compile")
        model = torch.compile(model)

    callbacks = [
        CheckpointCallback(checkpoint_manager),
        ConsoleMetricsCallback(log_steps=config.tracking.wandb.log_steps),
    ]
    wandb_callback: WandbMetricsCallback | None = None
    if config.tracking.wandb.enabled:
        wandb_callback = WandbMetricsCallback(
            config=config,
            run_dir=config.experiment.run_dir,
            run_metadata=run_metadata,
        )
        callbacks.append(wandb_callback)

    trainer = SFTTrainer(
        config=config,
        model=model,
        data_pipeline=data_pipeline,
        optimization=optimization,
        runtime=runtime,
        callbacks=callbacks,
        state=trainer_state,
    )
    status(
        f"Starting training at optimizer_step={trainer.state.optimizer_step}; "
        f"planned total={optimization.schedule.total_optimizer_steps}"
    )
    try:
        final_state = trainer.train(
            max_optimizer_steps=args.max_optimizer_steps,
        )
    except BaseException:
        if wandb_callback is not None:
            wandb_callback.finish(exit_code=1)
        raise

    report = {
        "status": "completed",
        "experiment": config.experiment.name,
        "run_dir": str(config.experiment.run_dir),
        "trainer_state": final_state.state_dict(),
        "last_checkpoint": (
            str(checkpoint_manager.last_checkpoint)
            if checkpoint_manager.last_checkpoint is not None
            else None
        ),
        "wandb_enabled": config.tracking.wandb.enabled,
    }
    status("SFT training process finished")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

