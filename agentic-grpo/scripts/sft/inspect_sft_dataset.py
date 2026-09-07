"""Inspect the final SFT dataset without PyTorch, padding, or model loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agentic_rl.sft.dataset import TokenizedSFTDataset

TRAINING_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    TRAINING_ROOT
    / "data/tokenized/retail_serial_v3_qwen3_4b_instruct_2507"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one unpadded memory-mapped SFT trajectory."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=17_408)
    parser.add_argument("--skip-hash-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TokenizedSFTDataset(
        args.data,
        expected_max_seq_length=args.max_seq_length,
        verify_hashes=not args.skip_hash_validation,
    )
    sample = dataset[args.index]
    summary = {
        "dataset_directory": str(dataset.directory),
        "trajectories": len(dataset),
        "total_tokens": dataset.total_tokens,
        "total_supervised_tokens": dataset.total_supervised_tokens,
        "max_seq_length_contract": dataset.max_seq_length,
        "actual_min_length": int(dataset.lengths.min()),
        "actual_max_length": int(dataset.lengths.max()),
        "manifest_sha256": dataset.manifest_sha256,
        "sample": {
            "dataset_index": sample.dataset_index,
            "trajectory_id": sample.trajectory_id,
            "task_id": sample.task_id,
            "token_count": sample.token_count,
            "supervised_token_count": sample.supervised_token_count,
            "masked_token_count": sample.token_count
            - sample.supervised_token_count,
            "input_dtype": str(sample.input_ids.dtype),
            "label_dtype": str(sample.labels.dtype),
            "input_shape": list(sample.input_ids.shape),
            "label_shape": list(sample.labels.shape),
            "supervision_contract_valid": bool(
                np.all((sample.labels == -100) | (sample.labels == sample.input_ids))
            ),
            "contains_padding": False,
            "first_12_input_ids": sample.input_ids[:12].tolist(),
            "last_12_input_ids": sample.input_ids[-12:].tolist(),
        },
        "pytorch_loaded": False,
        "padding_applied": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
