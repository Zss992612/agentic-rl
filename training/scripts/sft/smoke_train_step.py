"""Run a bounded real-data SFT forward/backward/optimizer smoke test.

This script mutates LoRA parameters only in process memory and never writes a
checkpoint.  Its purpose is to prove that the finalized data, Qwen model,
assistant-only labels, PEFT adapters, optimizer, and CUDA runtime work together
before a long-running Trainer is introduced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
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
from agentic_rl.sft.runtime import configure_runtime, move_batch_to_device
from agentic_rl.sft.trainer import (
    scale_gradients,
    shifted_supervised_token_count,
)
from torch import nn

TRAINING_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)
MEBIBYTE = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded Qwen SFT forward/backward/optimizer smoke test "
            "without saving weights."
        )
    )
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--micro-batches",
        type=int,
        default=1,
        help=(
            "Number of micro-batches accumulated before the smoke optimizer "
            "step (default: 1; project training config uses 8)"
        ),
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        help="Optionally round dynamic batch padding to this token multiple",
    )
    parser.add_argument(
        "--skip-dataset-hash-verification",
        action="store_true",
        help="Skip dataset hashes while retaining shape and label checks",
    )
    return parser.parse_args()


def status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_memory_report(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda":
        return None
    synchronize(device)
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / MEBIBYTE,
        "reserved_mib": torch.cuda.memory_reserved(device) / MEBIBYTE,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MEBIBYTE,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MEBIBYTE,
    }


def gradient_report(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> dict[str, Any]:
    gradient_tensors = 0
    nonzero_gradient_tensors = 0
    nonfinite_names: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad or parameter.grad is None:
            continue
        gradient_tensors += 1
        if not torch.isfinite(parameter.grad).all().item():
            nonfinite_names.append(name)
        if torch.count_nonzero(parameter.grad).item() > 0:
            nonzero_gradient_tensors += 1
    return {
        "gradient_tensors": gradient_tensors,
        "nonzero_gradient_tensors": nonzero_gradient_tensors,
        "nonfinite_parameter_names": nonfinite_names,
    }


def choose_update_probe(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> tuple[str, nn.Parameter]:
    trainable = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("model has no trainable parameters for an update probe")
    # LoRA B is initialized to zero and normally receives a non-zero gradient
    # on the first backward pass, making it the clearest update probe.
    return next(
        (
            (name, parameter)
            for name, parameter in trainable
            if "lora_B" in name
        ),
        trainable[0],
    )


def main() -> None:
    args = parse_args()
    if args.micro_batches <= 0:
        raise ValueError("--micro-batches must be positive")
    if args.pad_to_multiple_of is not None and args.pad_to_multiple_of <= 0:
        raise ValueError("--pad-to-multiple-of must be positive")

    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    status(f"Loading SFT config: {args.config.resolve()}")
    config = load_sft_config(args.config)
    if args.micro_batches > config.optimization.gradient_accumulation_steps:
        raise ValueError(
            "--micro-batches cannot exceed configured "
            f"gradient_accumulation_steps={config.optimization.gradient_accumulation_steps}"
        )

    stage_start = time.perf_counter()
    status("Loading tokenizer, model metadata, and tokenized dataset")
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
    timings["data_and_metadata_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    status("Configuring training runtime")
    runtime = configure_runtime(
        config.runtime,
        config.model,
        seed=config.experiment.seed,
    )
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)
    memory_before_model = cuda_memory_report(runtime.device)
    timings["runtime_setup_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    status(f"Loading the base model on {runtime.device}")
    model = load_base_model(config.model, device=runtime.device)
    memory_after_base_model = cuda_memory_report(runtime.device)
    timings["base_model_load_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    status("Injecting and auditing LoRA adapters")
    model, lora_report = apply_lora(
        model,
        config.lora,
        gradient_checkpointing=config.model.gradient_checkpointing,
    )
    if runtime.compile_enabled:
        status("Applying torch.compile as requested by runtime.compile")
        model = torch.compile(model)
    model.train()
    memory_after_lora = cuda_memory_report(runtime.device)
    timings["lora_setup_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    optimization = build_optimization(
        model,
        config.optimization,
        micro_batches_per_epoch=len(data_pipeline.dataloader),
    )
    optimizer = optimization.optimizer
    scheduler = optimization.scheduler
    optimizer.zero_grad(set_to_none=True)
    probe_name, probe_parameter = choose_update_probe(model.named_parameters())
    probe_before = probe_parameter.detach().float().cpu().clone()
    timings["optimization_setup_seconds"] = time.perf_counter() - stage_start

    batch_reports: list[dict[str, Any]] = []
    total_supervised_tokens = 0
    total_token_loss = 0.0
    data_iterator = iter(data_pipeline.dataloader)
    memory_after_forward: dict[str, float] | None = None
    memory_after_backward: dict[str, float] | None = None

    status(
        f"Running {args.micro_batches} real micro-batch(es) with token-normalized gradients"
    )
    stage_start = time.perf_counter()
    for micro_batch_index in range(args.micro_batches):
        batch = next(data_iterator)
        batch = move_batch_to_device(
            batch,
            runtime.device,
            non_blocking=config.data.pin_memory,
        )
        supervised_tokens = shifted_supervised_token_count(batch["labels"])
        if supervised_tokens <= 0:
            raise RuntimeError(
                f"micro-batch {micro_batch_index} has no shifted supervised tokens"
            )

        forward_start = time.perf_counter()
        with runtime.autocast():
            outputs = model(**batch)
            loss = outputs.loss
        synchronize(runtime.device)
        forward_seconds = time.perf_counter() - forward_start
        if loss is None or not torch.isfinite(loss).item():
            raise RuntimeError(
                f"micro-batch {micro_batch_index} produced a non-finite loss: {loss}"
            )
        if memory_after_forward is None:
            memory_after_forward = cuda_memory_report(runtime.device)

        # HF returns a mean over valid shifted labels. Multiplication restores
        # the token-loss sum; gradients are divided by the complete window's
        # supervised-token count after all backward calls.
        token_loss_sum = loss * supervised_tokens
        backward_start = time.perf_counter()
        token_loss_sum.backward()
        synchronize(runtime.device)
        backward_seconds = time.perf_counter() - backward_start
        if memory_after_backward is None:
            memory_after_backward = cuda_memory_report(runtime.device)

        total_supervised_tokens += supervised_tokens
        total_token_loss += float(loss.detach().float().item()) * supervised_tokens
        batch_reports.append(
            {
                "micro_batch": micro_batch_index,
                "shape": list(batch["input_ids"].shape),
                "input_tokens": int(batch["attention_mask"].sum().item()),
                "shifted_supervised_tokens": supervised_tokens,
                "loss": float(loss.detach().float().item()),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
            }
        )

    scale_gradients(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        total_supervised_tokens,
    )
    gradients = gradient_report(model.named_parameters())
    if gradients["gradient_tensors"] == 0:
        raise RuntimeError("backward produced no trainable gradients")
    if gradients["nonzero_gradient_tensors"] == 0:
        raise RuntimeError("all trainable gradients are zero")
    if gradients["nonfinite_parameter_names"]:
        raise RuntimeError(
            "non-finite gradients: "
            + ", ".join(gradients["nonfinite_parameter_names"])
        )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    gradient_norm = nn.utils.clip_grad_norm_(
        trainable_parameters,
        config.optimization.max_grad_norm,
        error_if_nonfinite=True,
    )
    learning_rate_before = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    probe_optimizer_state = optimizer.state.get(probe_parameter, {})
    optimizer_state_initialized = bool(probe_optimizer_state)
    optimizer_state_step = probe_optimizer_state.get("step")
    if isinstance(optimizer_state_step, torch.Tensor):
        optimizer_state_step = float(optimizer_state_step.item())
    elif optimizer_state_step is not None:
        optimizer_state_step = float(optimizer_state_step)
    if not optimizer_state_initialized:
        raise RuntimeError(
            f"optimizer did not initialize state for probe parameter {probe_name}"
        )
    optimizer.zero_grad(set_to_none=True)
    synchronize(runtime.device)
    timings["forward_backward_optimizer_seconds"] = (
        time.perf_counter() - stage_start
    )

    probe_after = probe_parameter.detach().float().cpu()
    probe_delta_l2 = float(torch.linalg.vector_norm(probe_after - probe_before).item())
    parameter_update_expected = learning_rate_before > 0
    if parameter_update_expected and probe_delta_l2 == 0:
        raise RuntimeError(
            f"optimizer used lr={learning_rate_before} but probe {probe_name} did not change"
        )
    memory_after_optimizer = cuda_memory_report(runtime.device)

    timings["total_seconds"] = time.perf_counter() - total_start
    report = {
        "status": "passed",
        "safety": {
            "checkpoint_written": False,
            "weights_persisted": False,
            "process_memory_only_update": True,
        },
        "config": str(config.source_path),
        "experiment": config.experiment.name,
        "runtime": {
            "device": str(runtime.device),
            "device_name": (
                torch.cuda.get_device_name(runtime.device)
                if runtime.device.type == "cuda"
                else runtime.device.type
            ),
            "model_dtype": str(runtime.model_dtype),
            "tf32_enabled": runtime.tf32_enabled,
            "gradient_checkpointing": bool(
                getattr(model, "is_gradient_checkpointing", False)
            ),
            "torch_compile": runtime.compile_enabled,
        },
        "tokenizer_contract": asdict(tokenizer_contract),
        "data": {
            "dataset_trajectories": len(data_pipeline.dataset),
            "micro_batch_size": config.optimization.micro_batch_size,
            "smoke_micro_batches": args.micro_batches,
            "configured_gradient_accumulation_steps": (
                config.optimization.gradient_accumulation_steps
            ),
            "partial_accumulation_window": (
                args.micro_batches
                < config.optimization.gradient_accumulation_steps
            ),
            "total_shifted_supervised_tokens": total_supervised_tokens,
            "token_normalized_loss": total_token_loss / total_supervised_tokens,
            "batches": batch_reports,
        },
        "lora": {
            "matched_modules": lora_report.matched_modules,
            "total_parameters": lora_report.total_parameters,
            "trainable_parameters": lora_report.trainable_parameters,
            "trainable_percentage": lora_report.trainable_fraction * 100,
            "trainable_parameter_tensors": len(
                lora_report.trainable_parameter_names
            ),
        },
        "optimization": {
            "schedule": asdict(optimization.schedule),
            "gradient_tensors": gradients["gradient_tensors"],
            "nonzero_gradient_tensors": gradients[
                "nonzero_gradient_tensors"
            ],
            "gradient_norm_before_clip": float(gradient_norm.item()),
            "max_grad_norm": config.optimization.max_grad_norm,
            "learning_rate_before_optimizer_step": learning_rate_before,
            "learning_rate_after_scheduler_step": optimizer.param_groups[0]["lr"],
            "optimizer_state_initialized": optimizer_state_initialized,
            "optimizer_state_step": optimizer_state_step,
            "update_probe_parameter": probe_name,
            "update_probe_delta_l2": probe_delta_l2,
            "parameter_update_expected_at_current_lr": parameter_update_expected,
        },
        "cuda_memory": {
            "before_model_load": memory_before_model,
            "after_base_model": memory_after_base_model,
            "after_lora": memory_after_lora,
            "after_first_forward": memory_after_forward,
            "after_first_backward": memory_after_backward,
            "after_optimizer": memory_after_optimizer,
        },
        "timings": timings,
    }
    status("Smoke train step passed; no checkpoint or model weights were saved")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        status(
            "CUDA out of memory during smoke test. No weights were saved; "
            "inspect the last printed stage and lower memory pressure before retrying."
        )
        raise
