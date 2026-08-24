"""Load and audit the configured base model plus LoRA adapters.

The default mode performs the real CUDA model load but never runs a forward
pass, updates parameters, or saves weights.  ``--metadata-only`` validates the
model/tokenizer/data contract without loading the 4B weight tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from agentic_rl.sft.config import load_sft_config
from agentic_rl.sft.dataset import TokenizedSFTDataset
from agentic_rl.sft.lora import apply_lora
from agentic_rl.sft.modeling import (
    load_architecture_config,
    load_base_model,
    load_tokenizer,
    validate_tokenizer_contract,
)
from agentic_rl.sft.runtime import configure_runtime

TRAINING_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)
MEBIBYTE = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the SFT tokenizer contract, load the base model, inject "
            "LoRA, and report trainable parameters without training."
        )
    )
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate config/tokenizer/data metadata without loading model weights",
    )
    parser.add_argument(
        "--skip-dataset-hash-verification",
        action="store_true",
        help="Skip persisted dataset file hashes (shape/contracts are still checked)",
    )
    return parser.parse_args()


def status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def architecture_report(architecture: Any) -> dict[str, Any]:
    return {
        "model_type": getattr(architecture, "model_type", None),
        "architectures": getattr(architecture, "architectures", None),
        "vocab_size": getattr(architecture, "vocab_size", None),
        "hidden_size": getattr(architecture, "hidden_size", None),
        "intermediate_size": getattr(architecture, "intermediate_size", None),
        "num_hidden_layers": getattr(architecture, "num_hidden_layers", None),
        "num_attention_heads": getattr(architecture, "num_attention_heads", None),
        "num_key_value_heads": getattr(
            architecture, "num_key_value_heads", None
        ),
        "max_position_embeddings": getattr(
            architecture, "max_position_embeddings", None
        ),
    }


def cuda_memory_report(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / MEBIBYTE,
        "reserved_mib": torch.cuda.memory_reserved(device) / MEBIBYTE,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MEBIBYTE,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MEBIBYTE,
    }


def main() -> None:
    args = parse_args()
    status(f"Loading SFT config: {args.config.resolve()}")
    config = load_sft_config(args.config)

    status("Opening tokenized dataset and validating its storage contract")
    dataset = TokenizedSFTDataset(
        config.data.path,
        expected_max_seq_length=config.data.max_seq_length,
        verify_hashes=not args.skip_dataset_hash_verification,
    )
    status("Loading tokenizer and model architecture metadata")
    tokenizer = load_tokenizer(config.model)
    architecture = load_architecture_config(config.model)
    tokenizer_contract = validate_tokenizer_contract(
        tokenizer,
        architecture,
        dataset,
        model_path=config.model.path,
    )

    report: dict[str, Any] = {
        "mode": "metadata_only" if args.metadata_only else "full_model_setup",
        "config": str(config.source_path),
        "experiment": config.experiment.name,
        "model_path": str(config.model.path),
        "dataset_path": str(config.data.path),
        "dataset": {
            "trajectories": len(dataset),
            "total_tokens": dataset.total_tokens,
            "total_supervised_tokens": dataset.total_supervised_tokens,
            "max_seq_length_contract": dataset.max_seq_length,
            "actual_max_seq_length": int(dataset.lengths.max()),
            "manifest_sha256": dataset.manifest_sha256,
            "hash_verification": not args.skip_dataset_hash_verification,
        },
        "architecture": architecture_report(architecture),
        "tokenizer_contract": asdict(tokenizer_contract),
        "requested_setup": {
            "dtype": config.model.dtype,
            "attention_implementation": config.model.attention_implementation,
            "use_cache": config.model.use_cache,
            "gradient_checkpointing": config.model.gradient_checkpointing,
            "runtime_device": config.runtime.device,
            "tf32": config.runtime.tf32,
            "compile": config.runtime.compile,
            "lora_enabled": config.lora.enabled,
            "lora_rank": config.lora.rank,
            "lora_alpha": config.lora.alpha,
            "lora_dropout": config.lora.dropout,
            "lora_bias": config.lora.bias,
            "lora_target_modules": list(config.lora.target_modules),
        },
    }

    if args.metadata_only:
        status("Metadata-only audit complete; model weights were not loaded")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    status("Configuring the requested training runtime")
    runtime = configure_runtime(
        config.runtime,
        config.model,
        seed=config.experiment.seed,
    )
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    memory_before = cuda_memory_report(runtime.device)

    status(
        f"Loading base model on {runtime.device} as {runtime.model_dtype}; "
        "this may take several minutes"
    )
    model = load_base_model(config.model, device=runtime.device)
    memory_after_base_model = cuda_memory_report(runtime.device)

    status("Validating LoRA targets and injecting adapters")
    model, lora_report = apply_lora(
        model,
        config.lora,
        gradient_checkpointing=config.model.gradient_checkpointing,
    )
    memory_after_lora = cuda_memory_report(runtime.device)

    first_parameter = next(model.parameters())
    report["runtime"] = {
        "device": str(runtime.device),
        "device_name": (
            torch.cuda.get_device_name(runtime.device)
            if runtime.device.type == "cuda"
            else runtime.device.type
        ),
        "model_parameter_dtype": str(first_parameter.dtype),
        "tf32_enabled": runtime.tf32_enabled,
        "deterministic": runtime.deterministic,
        "compile_requested": runtime.compile_enabled,
        "compile_applied": False,
    }
    report["model"] = {
        "class": model.__class__.__name__,
        "use_cache": getattr(model.config, "use_cache", None),
        "gradient_checkpointing": bool(
            getattr(model, "is_gradient_checkpointing", False)
        ),
    }
    report["lora"] = {
        "enabled": lora_report.enabled,
        "matched_modules": lora_report.matched_modules,
        "total_matched_modules": sum(lora_report.matched_modules.values()),
        "total_parameters": lora_report.total_parameters,
        "trainable_parameters": lora_report.trainable_parameters,
        "trainable_fraction": lora_report.trainable_fraction,
        "trainable_percentage": lora_report.trainable_fraction * 100,
        "trainable_parameter_tensors": len(
            lora_report.trainable_parameter_names
        ),
        "all_trainable_names_are_lora": all(
            "lora_" in name for name in lora_report.trainable_parameter_names
        ),
    }
    report["cuda_memory"] = {
        "before_model_load": memory_before,
        "after_base_model": memory_after_base_model,
        "after_lora": memory_after_lora,
    }

    status("Model and LoRA setup audit complete; no training step was executed")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

