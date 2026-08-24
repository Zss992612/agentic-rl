"""Checkpoint six: build the final memory-mappable NumPy SFT dataset."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from agentic_rl.preprocessing.masking import IGNORE_INDEX, build_assistant_labels
from agentic_rl.preprocessing.rendering import chat_template_sha256
from agentic_rl.preprocessing.schema import iter_sft_examples
from agentic_rl.preprocessing.storage import (
    FORMAT_NAME,
    FORMAT_VERSION,
    INPUT_DTYPE,
    LABEL_DTYPE,
    OFFSET_DTYPE,
    sha256_file,
    validate_dataset_directory,
    validate_flat_arrays,
)
from agentic_rl.preprocessing.tokenization import tokenize_sft_example

TRAINING_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA = TRAINING_DIR / "data/sft/retail_serial_v3/sft_train_256.jsonl"
DEFAULT_TOKENIZER = TRAINING_DIR / "artifacts/tokenizers/Qwen3-4B-Instruct-2507"
DEFAULT_SCAN_SUMMARY = (
    TRAINING_DIR / "results/preprocessing/checkpoint5/dataset_scan_summary.json"
)
DEFAULT_OUTPUT_DIR = (
    TRAINING_DIR
    / "data/tokenized/retail_serial_v3_qwen3_4b_instruct_2507"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint six: convert every validated trajectory to flat, "
            "memory-mappable input_ids, labels, and offsets arrays."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--scan-summary", type=Path, default=DEFAULT_SCAN_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=17_408)
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


def load_scan_contract(path: Path, data_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"checkpoint-five scan summary does not exist: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("checkpoint") != "full_dataset_transient_scan":
        raise SystemExit("scan summary is not a checkpoint-five report")
    if summary.get("counts", {}).get("failed_trajectories") != 0:
        raise SystemExit("scan summary contains failed trajectories")
    expected_hash = summary.get("source", {}).get("sha256")
    actual_hash = sha256_file(data_path)
    if expected_hash != actual_hash:
        raise SystemExit(
            "source JSONL changed after checkpoint five; rerun the scan first"
        )
    return summary


def numeric_summary(values: list[int | float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def portable_training_path(path: Path) -> str:
    """Prefer a path relative to training/ so manifests survive server sync."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(TRAINING_DIR.resolve()))
    except ValueError:
        return str(resolved)


def main() -> None:
    args = parse_args()
    if args.max_seq_length <= 0:
        raise SystemExit("--max-seq-length must be positive")
    if args.output_dir.exists():
        raise SystemExit(
            f"output directory already exists; refusing to overwrite: {args.output_dir}"
        )

    scan = load_scan_contract(args.scan_summary, args.data)
    expected_trajectories = scan["counts"]["successful_trajectories"]
    expected_tokens = scan["totals"]["tokens"]
    scanned_max = scan["token_lengths"]["max"]
    if scanned_max > args.max_seq_length:
        raise SystemExit(
            f"scan contains {scanned_max} tokens but max is {args.max_seq_length}"
        )

    tokenizer = load_local_tokenizer(args.tokenizer)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.tmp-",
            dir=args.output_dir.parent,
        )
    )
    try:
        input_ids = np.lib.format.open_memmap(
            staging / "input_ids.npy",
            mode="w+",
            dtype=INPUT_DTYPE,
            shape=(expected_tokens,),
        )
        labels = np.lib.format.open_memmap(
            staging / "labels.npy",
            mode="w+",
            dtype=LABEL_DTYPE,
            shape=(expected_tokens,),
        )
        offsets = np.lib.format.open_memmap(
            staging / "offsets.npy",
            mode="w+",
            dtype=OFFSET_DTYPE,
            shape=(expected_trajectories + 1,),
        )
        offsets[0] = 0

        metadata_path = staging / "metadata.jsonl"
        cursor = 0
        trajectory_count = 0
        token_counts: list[int] = []
        supervised_counts: list[int] = []
        assistant_kind_counts: Counter[str] = Counter()
        boundary_merged_spans = 0
        boundary_masked_characters = 0
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            for index, example in enumerate(iter_sft_examples(args.data)):
                tokenized = tokenize_sft_example(example, tokenizer)
                masked = build_assistant_labels(example, tokenized, tokenizer)
                token_count = tokenized.token_count
                if token_count > args.max_seq_length:
                    raise RuntimeError(
                        f"trajectory {index} has {token_count} tokens, above "
                        f"the {args.max_seq_length} limit"
                    )
                next_cursor = cursor + token_count
                if next_cursor > expected_tokens:
                    raise RuntimeError("token count exceeds checkpoint-five contract")

                input_ids[cursor:next_cursor] = np.asarray(
                    tokenized.input_ids,
                    dtype=INPUT_DTYPE,
                )
                labels[cursor:next_cursor] = np.asarray(
                    masked.labels,
                    dtype=LABEL_DTYPE,
                )
                offsets[index + 1] = next_cursor
                text_spans = sum(
                    span.kind == "text" for span in masked.assistant_spans
                )
                tool_call_spans = sum(
                    span.kind == "tool_call" for span in masked.assistant_spans
                )
                assistant_kind_counts.update(
                    {"text": text_spans, "tool_call": tool_call_spans}
                )
                merged_spans = sum(
                    span.masked_leading_character_count > 0
                    for span in masked.assistant_spans
                )
                masked_characters = sum(
                    span.masked_leading_character_count
                    for span in masked.assistant_spans
                )
                boundary_merged_spans += merged_spans
                boundary_masked_characters += masked_characters
                metadata_record = {
                    "dataset_index": index,
                    "trajectory_id": example.trajectory_id,
                    "task_id": example.task_id,
                    "offset_start": cursor,
                    "offset_end": next_cursor,
                    "token_count": token_count,
                    "supervised_token_count": masked.supervised_token_count,
                    "masked_token_count": masked.masked_token_count,
                    "supervised_fraction": masked.supervised_fraction,
                    "message_count": tokenized.rendered.message_count,
                    "assistant_spans": [
                        {
                            "ordinal": span.ordinal,
                            "kind": span.kind,
                            "header_start": span.header_start,
                            "target_start": span.target_start,
                            "target_end": span.target_end,
                            "supervised_token_count": span.supervised_token_count,
                            "masked_leading_text": span.masked_leading_text,
                        }
                        for span in masked.assistant_spans
                    ],
                    "tool_definition_count": (
                        tokenized.rendered.tool_definition_count
                    ),
                    "tool_result_count": tokenized.rendered.tool_result_count,
                    "source_metadata": example.metadata,
                }
                metadata_file.write(
                    json.dumps(metadata_record, ensure_ascii=False) + "\n"
                )
                cursor = next_cursor
                trajectory_count += 1
                token_counts.append(token_count)
                supervised_counts.append(masked.supervised_token_count)
                if trajectory_count % 32 == 0:
                    print(
                        f"Built {trajectory_count}/{expected_trajectories} trajectories...",
                        flush=True,
                    )

        if trajectory_count != expected_trajectories:
            raise RuntimeError(
                "trajectory count changed after scan: "
                f"{trajectory_count} != {expected_trajectories}"
            )
        if cursor != expected_tokens:
            raise RuntimeError(
                f"token count changed after scan: {cursor} != {expected_tokens}"
            )
        input_ids.flush()
        labels.flush()
        offsets.flush()
        validate_flat_arrays(input_ids, labels, offsets)
        total_supervised = int(np.count_nonzero(labels != IGNORE_INDEX))
        del input_ids, labels, offsets

        stats = {
            "trajectories": trajectory_count,
            "tokens": cursor,
            "supervised_tokens": total_supervised,
            "masked_tokens": cursor - total_supervised,
            "weighted_supervised_fraction": total_supervised / cursor,
            "token_lengths": numeric_summary(token_counts),
            "supervised_token_lengths": numeric_summary(supervised_counts),
            "assistant_spans": dict(assistant_kind_counts),
            "boundary_merged_assistant_spans": boundary_merged_spans,
            "boundary_masked_characters": boundary_masked_characters,
            "max_seq_length": args.max_seq_length,
            "over_max_seq_length": 0,
        }
        stats_path = staging / "stats.json"
        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        tracked_files = [
            "input_ids.npy",
            "labels.npy",
            "offsets.npy",
            "metadata.jsonl",
            "stats.json",
        ]
        manifest = {
            "format_name": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "path": portable_training_path(args.data),
                "sha256": sha256_file(args.data),
                "checkpoint5_summary": portable_training_path(args.scan_summary),
                "checkpoint5_summary_sha256": sha256_file(args.scan_summary),
            },
            "tokenizer": {
                "path": portable_training_path(args.tokenizer),
                "class": type(tokenizer).__name__,
                "vocab_size": tokenizer.vocab_size,
                "chat_template_sha256": chat_template_sha256(tokenizer),
                "tokenizer_json_sha256": sha256_file(
                    args.tokenizer / "tokenizer.json"
                ),
                "tokenizer_config_sha256": sha256_file(
                    args.tokenizer / "tokenizer_config.json"
                ),
            },
            "sequence_contract": {
                "max_seq_length": args.max_seq_length,
                "truncation": False,
                "padding": False,
                "padding_owner": "dynamic_training_collator",
                "order": "source_jsonl_order",
                "labels": (
                    "input token ID for assistant payload and assistant <|im_end|>; "
                    f"{IGNORE_INDEX} everywhere else"
                ),
                "manual_causal_shift": False,
                "offsets": (
                    "cumulative flat-array boundaries; trajectory i is "
                    "array[offsets[i]:offsets[i+1]]"
                ),
            },
            "arrays": {
                "input_ids.npy": {
                    "dtype": str(INPUT_DTYPE),
                    "shape": [cursor],
                },
                "labels.npy": {
                    "dtype": str(LABEL_DTYPE),
                    "shape": [cursor],
                    "ignore_index": IGNORE_INDEX,
                },
                "offsets.npy": {
                    "dtype": str(OFFSET_DTYPE),
                    "shape": [trajectory_count + 1],
                },
            },
            "counts": {
                "trajectories": trajectory_count,
                "tokens": cursor,
                "supervised_tokens": total_supervised,
            },
            "software": {
                "python": platform.python_version(),
                "numpy": package_version("numpy"),
                "transformers": package_version("transformers"),
                "tokenizers": package_version("tokenizers"),
            },
            "files": {
                name: {
                    "bytes": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in tracked_files
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        validated = validate_dataset_directory(staging, verify_hashes=True)
        os.replace(staging, args.output_dir)
        print(json.dumps(validated, ensure_ascii=False, indent=2))
        print(f"Final tokenized dataset: {args.output_dir}")
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
