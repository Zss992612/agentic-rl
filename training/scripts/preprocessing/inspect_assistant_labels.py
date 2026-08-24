"""Checkpoint four: build and audit assistant-only labels for one trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from agentic_rl.preprocessing.audit import iter_token_audit_rows, render_span_audit
from agentic_rl.preprocessing.masking import build_assistant_labels
from agentic_rl.preprocessing.schema import SFTExample, iter_sft_examples
from agentic_rl.preprocessing.tokenization import tokenize_sft_example

TRAINING_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA = TRAINING_DIR / "data/sft/retail_serial_v3/sft_train_256.jsonl"
DEFAULT_TOKENIZER = TRAINING_DIR / "artifacts/tokenizers/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "results/preprocessing/checkpoint4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint four: construct assistant-only labels for one trajectory "
            "and write human-readable audits. No final NumPy dataset is produced."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--index", type=int, default=0)
    selector.add_argument("--trajectory-id")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    masked = build_assistant_labels(example, tokenized, tokenizer)
    kind_counts = Counter(span.kind for span in masked.assistant_spans)
    summary = {
        "checkpoint": "qwen_assistant_only_labels",
        "dataset_index": index,
        "trajectory_id": tokenized.rendered.trajectory_id,
        "task_id": tokenized.rendered.task_id,
        "total_tokens": tokenized.token_count,
        "supervised_tokens": masked.supervised_token_count,
        "masked_tokens": masked.masked_token_count,
        "supervised_fraction": masked.supervised_fraction,
        "assistant_spans": len(masked.assistant_spans),
        "assistant_span_kinds": dict(kind_counts),
        "boundary_merged_assistant_spans": sum(
            span.masked_leading_character_count > 0
            for span in masked.assistant_spans
        ),
        "masked_leading_assistant_characters": sum(
            span.masked_leading_character_count
            for span in masked.assistant_spans
        ),
        "ignore_index": masked.ignore_index,
        "assistant_headers_masked": True,
        "assistant_im_end_supervised": True,
        "manual_label_shift": False,
        "truncated": False,
        "padded": False,
        "final_numpy_dataset_written": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trajectory_{index:03d}"
    summary_path = args.output_dir / f"{stem}_label_summary.json"
    spans_path = args.output_dir / f"{stem}_assistant_targets.txt"
    tokens_path = args.output_dir / f"{stem}_token_audit.tsv"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    spans_path.write_text(render_span_audit(masked, tokenizer), encoding="utf-8")
    with tokens_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        writer.writerow(["index", "input_id", "label", "loss", "region", "token"])
        for row in iter_token_audit_rows(masked, tokenizer):
            writer.writerow(
                [
                    row.index,
                    row.input_id,
                    row.label,
                    "yes" if row.loss else "no",
                    row.region,
                    row.token,
                ]
            )
    print(f"Label summary: {summary_path}")
    print(f"Assistant targets: {spans_path}")
    print(f"Per-token audit: {tokens_path}")


if __name__ == "__main__":
    main()
