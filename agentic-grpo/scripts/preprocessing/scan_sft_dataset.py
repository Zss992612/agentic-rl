"""Checkpoint five: transiently audit every SFT trajectory and write overviews."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from agentic_rl.preprocessing.masking import build_assistant_labels
from agentic_rl.preprocessing.schema import iter_sft_examples
from agentic_rl.preprocessing.tokenization import tokenize_sft_example

TRAINING_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA = TRAINING_DIR / "data/sft/retail_serial_v3/sft_train_256.jsonl"
DEFAULT_TOKENIZER = TRAINING_DIR / "artifacts/tokenizers/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "results/preprocessing/checkpoint5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint five: transiently render, tokenize, and mask every "
            "trajectory, then write metrics only. No final token arrays are saved."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-model-len", type=int, default=16_384)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values: list[float | int], percentage: float) -> float:
    """Linearly interpolated percentile without adding a NumPy dependency here."""

    if not sorted_values:
        raise ValueError("cannot calculate a percentile of no values")
    position = (len(sorted_values) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def numeric_summary(values: list[float | int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p75": percentile(ordered, 75),
        "p90": percentile(ordered, 90),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
    }


def length_bin(token_count: int, max_model_len: int) -> str:
    if token_count <= 4_096:
        return "00000-04096"
    if token_count <= 8_192:
        return "04097-08192"
    if token_count <= 12_288:
        return "08193-12288"
    if token_count <= max_model_len:
        return f"12289-{max_model_len:05d}"
    return f">{max_model_len}"


def build_summary(
    records: list[dict[str, Any]],
    *,
    data_path: Path,
    tokenizer_path: Path,
    tokenizer: Any,
    max_model_len: int,
) -> dict[str, Any]:
    successful = [record for record in records if record["status"] == "ok"]
    failed = [record for record in records if record["status"] != "ok"]
    if not successful:
        raise RuntimeError("all trajectories failed checkpoint-five processing")
    task_counts = Counter(record["task_id"] for record in successful)
    bins = Counter(record["length_bin"] for record in successful)
    boundary_records = [
        record for record in successful if record["boundary_merged_spans"] > 0
    ]
    non_whitespace_boundary_records = [
        record
        for record in successful
        if record["boundary_non_whitespace_characters"] > 0
    ]
    over_limit = [
        record for record in successful if record["tokens"] > max_model_len
    ]
    longest = sorted(successful, key=lambda record: record["tokens"], reverse=True)
    lowest_supervision = sorted(
        successful,
        key=lambda record: record["supervised_fraction"],
    )
    total_tokens = sum(record["tokens"] for record in successful)
    total_supervised = sum(record["supervised_tokens"] for record in successful)
    return {
        "checkpoint": "full_dataset_transient_scan",
        "source": {
            "path": str(data_path.resolve()),
            "sha256": sha256_file(data_path),
        },
        "tokenizer": {
            "path": str(tokenizer_path.resolve()),
            "class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
        },
        "settings": {
            "max_model_len": max_model_len,
            "truncation": False,
            "padding": False,
            "final_token_arrays_written": False,
        },
        "counts": {
            "records_seen": len(records),
            "successful_trajectories": len(successful),
            "failed_trajectories": len(failed),
            "unique_trajectory_ids": len(
                {record["trajectory_id"] for record in successful}
            ),
            "unique_tasks": len(task_counts),
            "trajectories_per_task": dict(Counter(task_counts.values())),
            "total_assistant_spans": sum(
                record["assistant_spans"] for record in successful
            ),
            "total_assistant_text_spans": sum(
                record["assistant_text_spans"] for record in successful
            ),
            "total_assistant_tool_call_spans": sum(
                record["assistant_tool_call_spans"] for record in successful
            ),
            "total_tool_results": sum(
                record["tool_results"] for record in successful
            ),
        },
        "token_lengths": numeric_summary(
            [record["tokens"] for record in successful]
        ),
        "supervised_token_lengths": numeric_summary(
            [record["supervised_tokens"] for record in successful]
        ),
        "supervised_fractions": numeric_summary(
            [record["supervised_fraction"] for record in successful]
        ),
        "totals": {
            "tokens": total_tokens,
            "supervised_tokens": total_supervised,
            "masked_tokens": total_tokens - total_supervised,
            "weighted_supervised_fraction": total_supervised / total_tokens,
        },
        "length_bins": dict(sorted(bins.items())),
        "context_limit": {
            "within_limit": len(successful) - len(over_limit),
            "over_limit": len(over_limit),
            "over_limit_records": [
                {
                    "dataset_index": record["dataset_index"],
                    "trajectory_id": record["trajectory_id"],
                    "task_id": record["task_id"],
                    "tokens": record["tokens"],
                }
                for record in over_limit
            ],
        },
        "boundary_alignment": {
            "trajectories_with_merged_boundaries": len(boundary_records),
            "merged_assistant_spans": sum(
                record["boundary_merged_spans"] for record in successful
            ),
            "masked_leading_characters": sum(
                record["boundary_masked_characters"] for record in successful
            ),
            "non_whitespace_characters": sum(
                record["boundary_non_whitespace_characters"]
                for record in successful
            ),
            "trajectories_with_non_whitespace": [
                {
                    "dataset_index": record["dataset_index"],
                    "trajectory_id": record["trajectory_id"],
                    "task_id": record["task_id"],
                    "characters": record["boundary_non_whitespace_characters"],
                }
                for record in non_whitespace_boundary_records
            ],
        },
        "top_10_longest": [
            {
                "dataset_index": record["dataset_index"],
                "trajectory_id": record["trajectory_id"],
                "task_id": record["task_id"],
                "tokens": record["tokens"],
                "supervised_tokens": record["supervised_tokens"],
                "supervised_fraction": record["supervised_fraction"],
            }
            for record in longest[:10]
        ],
        "bottom_10_supervised_fraction": [
            {
                "dataset_index": record["dataset_index"],
                "trajectory_id": record["trajectory_id"],
                "task_id": record["task_id"],
                "tokens": record["tokens"],
                "supervised_tokens": record["supervised_tokens"],
                "supervised_fraction": record["supervised_fraction"],
            }
            for record in lowest_supervision[:10]
        ],
        "failures": failed,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lengths = summary["token_lengths"]
    supervised = summary["supervised_token_lengths"]
    totals = summary["totals"]
    context = summary["context_limit"]
    boundary = summary["boundary_alignment"]
    lines = [
        "# SFT dataset checkpoint-five overview",
        "",
        "This report is a transient audit. No final token or label arrays were written.",
        "",
        "## Dataset",
        "",
        f"- Trajectories: {counts['successful_trajectories']}",
        f"- Failed trajectories: {counts['failed_trajectories']}",
        f"- Unique tasks: {counts['unique_tasks']}",
        f"- Trajectories per task distribution: `{counts['trajectories_per_task']}`",
        f"- Total assistant spans: {counts['total_assistant_spans']}",
        f"- Assistant text spans: {counts['total_assistant_text_spans']}",
        f"- Assistant tool-call spans: {counts['total_assistant_tool_call_spans']}",
        f"- Tool results: {counts['total_tool_results']}",
        "",
        "## Token lengths",
        "",
        f"- Total tokens: {totals['tokens']:,}",
        f"- Min / median / mean / max: {lengths['min']:,} / {lengths['median']:,.1f} / {lengths['mean']:,.1f} / {lengths['max']:,}",
        f"- P90 / P95 / P99: {lengths['p90']:,.1f} / {lengths['p95']:,.1f} / {lengths['p99']:,.1f}",
        f"- Within {summary['settings']['max_model_len']:,}: {context['within_limit']}",
        f"- Over {summary['settings']['max_model_len']:,}: {context['over_limit']}",
        f"- Length bins: `{summary['length_bins']}`",
        "",
        "## Supervision",
        "",
        f"- Total supervised tokens: {totals['supervised_tokens']:,}",
        f"- Total masked tokens: {totals['masked_tokens']:,}",
        f"- Weighted supervised fraction: {totals['weighted_supervised_fraction']:.2%}",
        f"- Supervised min / median / mean / max: {supervised['min']:,} / {supervised['median']:,.1f} / {supervised['mean']:,.1f} / {supervised['max']:,}",
        "",
        "## Assistant boundary alignment",
        "",
        f"- Trajectories with merged boundaries: {boundary['trajectories_with_merged_boundaries']}",
        f"- Merged assistant spans: {boundary['merged_assistant_spans']}",
        f"- Masked leading characters: {boundary['masked_leading_characters']}",
        f"- Non-whitespace characters masked at boundaries: {boundary['non_whitespace_characters']}",
        "",
        "## Ten longest trajectories",
        "",
        "| Index | Task | Tokens | Supervised | Fraction | Trajectory ID |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for record in summary["top_10_longest"]:
        lines.append(
            f"| {record['dataset_index']} | {record['task_id']} | "
            f"{record['tokens']} | {record['supervised_tokens']} | "
            f"{record['supervised_fraction']:.2%} | {record['trajectory_id']} |"
        )
    lines.extend(
        [
            "",
            "## Ten lowest supervision fractions",
            "",
            "| Index | Task | Tokens | Supervised | Fraction | Trajectory ID |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in summary["bottom_10_supervised_fraction"]:
        lines.append(
            f"| {record['dataset_index']} | {record['task_id']} | "
            f"{record['tokens']} | {record['supervised_tokens']} | "
            f"{record['supervised_fraction']:.2%} | {record['trajectory_id']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.max_model_len <= 0:
        raise SystemExit("--max-model-len must be positive")
    tokenizer = load_local_tokenizer(args.tokenizer)
    records: list[dict[str, Any]] = []
    for index, example in enumerate(iter_sft_examples(args.data)):
        try:
            tokenized = tokenize_sft_example(example, tokenizer)
            masked = build_assistant_labels(example, tokenized, tokenizer)
            boundary_texts = [
                span.masked_leading_text
                for span in masked.assistant_spans
                if span.masked_leading_text
            ]
            non_whitespace_characters = sum(
                not character.isspace()
                for text in boundary_texts
                for character in text
            )
            text_spans = sum(
                span.kind == "text" for span in masked.assistant_spans
            )
            tool_call_spans = sum(
                span.kind == "tool_call" for span in masked.assistant_spans
            )
            records.append(
                {
                    "status": "ok",
                    "dataset_index": index,
                    "trajectory_id": example.trajectory_id,
                    "task_id": example.task_id,
                    "characters": tokenized.rendered.character_count,
                    "tokens": tokenized.token_count,
                    "length_bin": length_bin(
                        tokenized.token_count,
                        args.max_model_len,
                    ),
                    "over_max_model_len": tokenized.token_count > args.max_model_len,
                    "supervised_tokens": masked.supervised_token_count,
                    "masked_tokens": masked.masked_token_count,
                    "supervised_fraction": masked.supervised_fraction,
                    "messages": tokenized.rendered.message_count,
                    "assistant_spans": len(masked.assistant_spans),
                    "assistant_text_spans": text_spans,
                    "assistant_tool_call_spans": tool_call_spans,
                    "tool_results": tokenized.rendered.tool_result_count,
                    "boundary_merged_spans": sum(
                        span.masked_leading_character_count > 0
                        for span in masked.assistant_spans
                    ),
                    "boundary_masked_characters": sum(
                        span.masked_leading_character_count
                        for span in masked.assistant_spans
                    ),
                    "boundary_non_whitespace_characters": non_whitespace_characters,
                    "boundary_masked_texts": boundary_texts,
                }
            )
        except Exception as error:  # Preserve all failures in the audit output.
            records.append(
                {
                    "status": "error",
                    "dataset_index": index,
                    "trajectory_id": example.trajectory_id,
                    "task_id": example.task_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        if (index + 1) % 32 == 0:
            print(f"Scanned {index + 1} trajectories...", flush=True)

    summary = build_summary(
        records,
        data_path=args.data,
        tokenizer_path=args.tokenizer,
        tokenizer=tokenizer,
        max_model_len=args.max_model_len,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "dataset_scan_summary.json"
    records_path = args.output_dir / "trajectory_metrics.jsonl"
    csv_path = args.output_dir / "trajectory_metrics.csv"
    markdown_path = args.output_dir / "dataset_overview.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with records_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    fieldnames = sorted({key for record in records for key in record})
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            csv_record = dict(record)
            if isinstance(csv_record.get("boundary_masked_texts"), list):
                csv_record["boundary_masked_texts"] = json.dumps(
                    csv_record["boundary_masked_texts"],
                    ensure_ascii=False,
                )
            writer.writerow(csv_record)
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")

    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")
    print(f"Per-trajectory JSONL: {records_path}")
    print(f"Per-trajectory CSV: {csv_path}")
    print(f"Readable overview: {markdown_path}")
    if summary["counts"]["failed_trajectories"]:
        raise SystemExit("checkpoint-five scan completed with trajectory failures")


if __name__ == "__main__":
    main()
