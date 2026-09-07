"""Render one SFT trajectory with Qwen and inspect the exact input text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agentic_rl.preprocessing.rendering import render_sft_example
from agentic_rl.preprocessing.schema import SFTExample, iter_sft_examples

TRAINING_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA = TRAINING_DIR / "data/sft/retail_serial_v3/sft_train_256.jsonl"
DEFAULT_TOKENIZER = (
    TRAINING_DIR / "artifacts/tokenizers/Qwen3-4B-Instruct-2507"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint two: render one validated trajectory using the official "
            "Qwen chat template. No token IDs or labels are produced."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--index", type=int, default=0)
    selector.add_argument("--trajectory-id")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the rendered text; stdout is used otherwise",
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
    return AutoTokenizer.from_pretrained(
        path,
        use_fast=True,
        local_files_only=True,
    )


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
    rendered = render_sft_example(example, tokenizer)
    role_counts = Counter(message.role for message in example.messages)
    summary = {
        "checkpoint": "qwen_rendering_only",
        "dataset_index": index,
        "trajectory_id": rendered.trajectory_id,
        "task_id": rendered.task_id,
        "characters": rendered.character_count,
        "messages": rendered.message_count,
        "message_roles": dict(role_counts),
        "tool_definitions": rendered.tool_definition_count,
        "assistant_messages": rendered.assistant_message_count,
        "assistant_tool_calls": rendered.assistant_tool_call_count,
        "tool_results": rendered.tool_result_count,
        "chat_template_sha256": rendered.chat_template_sha256,
        "contains_transport_tool_call_ids": False,
        "tokenized": False,
        "labels_built": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered.text, encoding="utf-8")
        print(f"Rendered text: {args.output}")
        return
    print("\n--- QWEN RENDERED TEXT START ---\n")
    print(rendered.text, end="")
    print("\n--- QWEN RENDERED TEXT END ---")


if __name__ == "__main__":
    main()
