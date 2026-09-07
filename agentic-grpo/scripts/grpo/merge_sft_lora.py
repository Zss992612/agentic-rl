"""Merge the immutable SFT LoRA into its base model for GRPO training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _required_path(argument: Path | None, environment_name: str) -> Path:
    value = argument or os.environ.get(environment_name)
    if not value:
        raise RuntimeError(f"Pass the path explicitly or set {environment_name}")
    return Path(value).expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(directory: Path, patterns: tuple[str, ...]) -> list[dict[str, object]]:
    paths = sorted({path for pattern in patterns for path in directory.glob(pattern)})
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
        if path.is_file()
    ]


def _version(package: str) -> str:
    return importlib.metadata.version(package)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    base_model = _required_path(args.base_model, "AGENTIC_RL_MODEL_PATH")
    adapter = _required_path(args.adapter, "AGENTIC_RL_LORA_PATH")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    print(f"Loading SFT adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter)

    print("Merging SFT adapter into the base model")
    merged_model = model.merge_and_unload(safe_merge=True)
    merged_model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output)

    print("Hashing source and merged model artifacts")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": str(base_model),
        "sft_adapter": str(adapter),
        "output": str(output),
        "dtype": "bfloat16",
        "packages": {
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "peft": _version("peft"),
        },
        "source_files": _records(
            base_model,
            ("config.json", "model*.safetensors", "tokenizer_config.json"),
        )
        + _records(adapter, ("adapter_config.json", "adapter_model.safetensors")),
        "output_files": _records(
            output,
            ("config.json", "model*.safetensors", "tokenizer_config.json"),
        ),
    }
    manifest_path = output / "merge_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Merged SFT model: {output}")
    print(f"Merge manifest: {manifest_path}")


if __name__ == "__main__":
    main()
