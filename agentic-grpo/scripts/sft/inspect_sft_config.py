"""Resolve and validate an SFT YAML file without loading PyTorch or a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rl.sft.config import load_sft_config

TRAINING_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve and validate an SFT config without loading the model."
    )
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-path-validation",
        action="store_true",
        help="Validate values but do not require local model/data directories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_sft_config(
        args.config,
        validate_paths=not args.skip_path_validation,
    )
    print(json.dumps(config.to_serializable_dict(), ensure_ascii=False, indent=2))
    print("\nConfiguration summary")
    print(f"  experiment: {config.experiment.name}")
    print(f"  run dir: {config.experiment.run_dir}")
    print(f"  model: {config.model.path}")
    print(f"  dataset: {config.data.path}")
    print(f"  max sequence length: {config.data.max_seq_length}")
    print(f"  effective batch size: {config.optimization.effective_batch_size}")
    print(f"  LoRA targets: {', '.join(config.lora.target_modules)}")
    print(f"  W&B enabled: {config.tracking.wandb.enabled}")


if __name__ == "__main__":
    main()
