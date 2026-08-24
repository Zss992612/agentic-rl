"""Checkpoint three: tokenize one Qwen trajectory and audit its round trip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rl.preprocessing.schema import SFTExample, iter_sft_examples
from agentic_rl.preprocessing.tokenization import tokenize_sft_example

TRAINING_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA = TRAINING_DIR / "data/sft/retail_serial_v3/sft_train_256.jsonl"
DEFAULT_TOKENIZER = TRAINING_DIR / "artifacts/tokenizers/Qwen3-4B-Instruct-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint three: encode and decode one rendered Qwen trajectory. "
            "No labels, padding, or truncation are applied."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--index", type=int, default=0)
    selector.add_argument("--trajectory-id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for decoded text and the JSON audit summary",
    )
    return parser.parse_args()


def load_local_tokenizer(path: Path):
    if not path.is_dir():
        raise SystemExit(f"local tokenizer directory does not exist: {path}")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "transformers is not installed; sync the preprocessing dependency group"
        ) from error
    return AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)


def select_example(
    data_path: Path,
    *,
    index: int,
    trajectory_id: str | None,
) -> tuple[int, SFTExample]:
    if index < 0:
        raise SystemExit("--index must be non-negative")
    for current_index, example in enumerate(iter_sft_examples(data_path)):
        if trajectory_id is not None:
            if example.trajectory_id == trajectory_id:
                return current_index, example
        elif current_index == index:
            return current_index, example
    if trajectory_id is not None:
        raise SystemExit(f"trajectory ID not found: {trajectory_id}")
    raise SystemExit(f"dataset index out of range: {index}")


def main() -> None:
    args = parse_args()
    tokenizer = load_local_tokenizer(args.tokenizer)
    index, example = select_example(
        args.data,
        index=args.index,
        trajectory_id=args.trajectory_id,
    )
    tokenized = tokenize_sft_example(example, tokenizer)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    summary = {
        "checkpoint": "qwen_tokenize_decode_round_trip",
        "dataset_index": index,
        "trajectory_id": tokenized.rendered.trajectory_id,
        "task_id": tokenized.rendered.task_id,
        "characters": tokenized.rendered.character_count,
        "tokens": tokenized.token_count,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "im_start_token_id": im_start_id,
        "im_end_token_id": im_end_id,
        "im_start_count": tokenized.input_ids.count(im_start_id),
        "im_end_count": tokenized.input_ids.count(im_end_id),
        "text_sha256": tokenized.text_sha256,
        "decoded_sha256": tokenized.decoded_sha256,
        "round_trip_exact": tokenized.round_trip_exact,
        "direct_chat_template_match": tokenized.direct_chat_template_match,
        "truncated": False,
        "padded": False,
        "labels_built": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"trajectory_{index:03d}"
        decoded_path = args.output_dir / f"{stem}_decoded.txt"
        summary_path = args.output_dir / f"{stem}_tokenization_summary.json"
        decoded_path.write_text(tokenized.decoded_text, encoding="utf-8")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Decoded text: {decoded_path}")
        print(f"Audit summary: {summary_path}")


if __name__ == "__main__":
    main()
