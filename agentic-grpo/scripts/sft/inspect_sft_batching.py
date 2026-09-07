"""Audit length grouping and dynamic padding on the real SFT dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from agentic_rl.sft.collator import DynamicPaddingCollator
from agentic_rl.sft.config import load_sft_config
from agentic_rl.sft.dataset import TokenizedSFTDataset
from agentic_rl.sft.sampler import LengthGroupedBatchSampler

TRAINING_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect real length-grouped, dynamically padded SFT batches."
    )
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override optimization.micro_batch_size for this audit only",
    )
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--pad-to-multiple-of", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_batches <= 0:
        raise ValueError("--num-batches must be positive")

    config = load_sft_config(args.config)
    dataset = TokenizedSFTDataset(
        config.data.path,
        expected_max_seq_length=config.data.max_seq_length,
        verify_hashes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.path,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer has no pad_token_id")

    batch_size = args.batch_size or config.optimization.micro_batch_size
    sampler = LengthGroupedBatchSampler(
        dataset.lengths,
        batch_size=batch_size,
        bucket_size=config.data.bucket_size,
        shuffle=config.data.shuffle,
        seed=config.experiment.seed,
    )
    sampler.set_epoch(args.epoch)
    collator = DynamicPaddingCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_length=config.data.max_seq_length,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )

    batch_reports: list[dict[str, object]] = []
    total_real_tokens = 0
    total_tensor_slots = 0
    for batch_number, indices in enumerate(sampler):
        if batch_number >= args.num_batches:
            break
        samples = [dataset[index] for index in indices]
        batch = collator(samples)
        width = batch["input_ids"].shape[1]
        real_tokens = int(batch["attention_mask"].sum().item())
        tensor_slots = int(batch["attention_mask"].numel())
        total_real_tokens += real_tokens
        total_tensor_slots += tensor_slots
        batch_reports.append(
            {
                "batch": batch_number,
                "dataset_indices": indices,
                "trajectory_lengths": [sample.token_count for sample in samples],
                "tensor_shape": list(batch["input_ids"].shape),
                "padded_width": width,
                "real_tokens": real_tokens,
                "padding_tokens": tensor_slots - real_tokens,
                "supervised_tokens": int((batch["labels"] != -100).sum().item()),
            }
        )

    report = {
        "dataset": str(dataset.directory),
        "epoch": args.epoch,
        "batch_size": batch_size,
        "bucket_size": config.data.bucket_size,
        "pad_token_id": tokenizer.pad_token_id,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "inspected_batches": len(batch_reports),
        "padding_fraction": (
            0.0
            if total_tensor_slots == 0
            else (total_tensor_slots - total_real_tokens) / total_tensor_slots
        ),
        "batches": batch_reports,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

