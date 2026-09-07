"""Run the audited OpenAI-chat-to-Qwen SFT preprocessing pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

TRAINING_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = TRAINING_DIR / "scripts/preprocessing"


class PipelineConfigError(ValueError):
    """Raised when the preprocessing pipeline configuration is invalid."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    pipeline_name: str
    source_data: Path
    tokenizer_path: Path
    output_dir: Path
    audit_dir: Path
    expected_trajectories: int
    expected_tasks: int
    max_model_length: int
    max_seq_length: str | int
    length_rounding: int
    sample_index: int
    allow_truncation: bool


def _resolve_training_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PipelineConfigError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    return (TRAINING_DIR / path).resolve() if not path.is_absolute() else path.resolve()


def _positive_integer(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineConfigError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise PipelineConfigError(f"{field} must be >= {minimum}")
    return value


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Load and strictly validate one pipeline YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PipelineConfigError("pipeline config must contain a mapping")
    expected_keys = {
        "pipeline_name",
        "source_data",
        "tokenizer_path",
        "output_dir",
        "audit_dir",
        "expected_trajectories",
        "expected_tasks",
        "max_model_length",
        "max_seq_length",
        "length_rounding",
        "sample_index",
        "allow_truncation",
    }
    missing = expected_keys - raw.keys()
    extra = raw.keys() - expected_keys
    if missing:
        raise PipelineConfigError(f"missing config keys: {sorted(missing)}")
    if extra:
        raise PipelineConfigError(f"unknown config keys: {sorted(extra)}")

    name = raw["pipeline_name"]
    if not isinstance(name, str) or not name.strip():
        raise PipelineConfigError("pipeline_name must be a non-empty string")
    max_seq_length = raw["max_seq_length"]
    if max_seq_length != "auto":
        max_seq_length = _positive_integer(max_seq_length, "max_seq_length")
    allow_truncation = raw["allow_truncation"]
    if allow_truncation is not False:
        raise PipelineConfigError("allow_truncation must be false")

    config = PipelineConfig(
        pipeline_name=name,
        source_data=_resolve_training_path(raw["source_data"], "source_data"),
        tokenizer_path=_resolve_training_path(
            raw["tokenizer_path"], "tokenizer_path"
        ),
        output_dir=_resolve_training_path(raw["output_dir"], "output_dir"),
        audit_dir=_resolve_training_path(raw["audit_dir"], "audit_dir"),
        expected_trajectories=_positive_integer(
            raw["expected_trajectories"], "expected_trajectories"
        ),
        expected_tasks=_positive_integer(raw["expected_tasks"], "expected_tasks"),
        max_model_length=_positive_integer(
            raw["max_model_length"], "max_model_length"
        ),
        max_seq_length=max_seq_length,
        length_rounding=_positive_integer(
            raw["length_rounding"], "length_rounding"
        ),
        sample_index=_positive_integer(
            raw["sample_index"], "sample_index", allow_zero=True
        ),
        allow_truncation=allow_truncation,
    )
    if isinstance(config.max_seq_length, int):
        if config.max_seq_length > config.max_model_length:
            raise PipelineConfigError(
                "max_seq_length cannot exceed max_model_length"
            )
    return config


def round_up(value: int, multiple: int) -> int:
    """Round a positive length up to the requested storage contract."""
    if value <= 0 or multiple <= 0:
        raise ValueError("value and multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def select_max_seq_length(config: PipelineConfig, scanned_max: int) -> int:
    selected = (
        round_up(scanned_max, config.length_rounding)
        if config.max_seq_length == "auto"
        else config.max_seq_length
    )
    assert isinstance(selected, int)
    if scanned_max > selected:
        raise PipelineConfigError(
            f"scanned maximum {scanned_max} exceeds max_seq_length {selected}"
        )
    if selected > config.max_model_length:
        raise PipelineConfigError(
            f"selected max_seq_length {selected} exceeds model limit "
            f"{config.max_model_length}"
        )
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_commands(config: PipelineConfig) -> list[list[str]]:
    python = sys.executable
    data = str(config.source_data)
    tokenizer = str(config.tokenizer_path)
    audit = config.audit_dir
    index = str(config.sample_index)
    return [
        [
            python,
            str(SCRIPT_DIR / "inspect_rendered_trajectory.py"),
            "--data",
            data,
            "--tokenizer",
            tokenizer,
            "--index",
            index,
            "--output",
            str(audit / "checkpoint2/trajectory_000_rendered.txt"),
        ],
        [
            python,
            str(SCRIPT_DIR / "inspect_tokenized_trajectory.py"),
            "--data",
            data,
            "--tokenizer",
            tokenizer,
            "--index",
            index,
            "--output-dir",
            str(audit / "checkpoint3"),
        ],
        [
            python,
            str(SCRIPT_DIR / "inspect_assistant_labels.py"),
            "--data",
            data,
            "--tokenizer",
            tokenizer,
            "--index",
            index,
            "--output-dir",
            str(audit / "checkpoint4"),
        ],
        [
            python,
            str(SCRIPT_DIR / "scan_sft_dataset.py"),
            "--data",
            data,
            "--tokenizer",
            tokenizer,
            "--output-dir",
            str(audit / "checkpoint5"),
            "--max-model-len",
            str(config.max_model_length),
        ],
    ]


def build_command(config: PipelineConfig, max_seq_length: int) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "build_tokenized_dataset.py"),
        "--data",
        str(config.source_data),
        "--tokenizer",
        str(config.tokenizer_path),
        "--scan-summary",
        str(config.audit_dir / "checkpoint5/dataset_scan_summary.json"),
        "--output-dir",
        str(config.output_dir),
        "--max-seq-length",
        str(max_seq_length),
    ]


def print_command(command: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)


def run_command(command: list[str]) -> None:
    print_command(command)
    subprocess.run(command, cwd=TRAINING_DIR, check=True)


def preflight(config: PipelineConfig) -> None:
    """Validate source records and prove that the tokenizer renders tools."""
    if not config.source_data.is_file():
        raise PipelineConfigError(f"source data does not exist: {config.source_data}")
    if not config.tokenizer_path.is_dir():
        raise PipelineConfigError(
            f"tokenizer directory does not exist: {config.tokenizer_path}"
        )
    if config.output_dir.exists():
        raise PipelineConfigError(
            f"output directory already exists: {config.output_dir}"
        )
    if config.audit_dir.exists():
        raise PipelineConfigError(f"audit directory already exists: {config.audit_dir}")

    from agentic_rl.preprocessing.rendering import render_sft_example
    from agentic_rl.preprocessing.schema import iter_sft_examples
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path,
        use_fast=True,
        local_files_only=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise PipelineConfigError("tokenizer has no chat template")

    examples = list(iter_sft_examples(config.source_data))
    task_counts = Counter(example.task_id for example in examples)
    if len(examples) != config.expected_trajectories:
        raise PipelineConfigError(
            f"expected {config.expected_trajectories} trajectories, got {len(examples)}"
        )
    if len(task_counts) != config.expected_tasks:
        raise PipelineConfigError(
            f"expected {config.expected_tasks} tasks, got {len(task_counts)}"
        )
    if config.sample_index >= len(examples):
        raise PipelineConfigError("sample_index is outside the dataset")

    rendered = render_sft_example(examples[config.sample_index], tokenizer)
    if "<tools>" not in rendered.text or "<tool_call>" not in rendered.text:
        raise PipelineConfigError(
            "target tokenizer did not render the tools/tool-call protocol"
        )


def validate_scan(config: PipelineConfig) -> tuple[dict[str, Any], int]:
    path = config.audit_dir / "checkpoint5/dataset_scan_summary.json"
    scan = json.loads(path.read_text(encoding="utf-8"))
    counts = scan["counts"]
    if counts["successful_trajectories"] != config.expected_trajectories:
        raise PipelineConfigError("scan trajectory count differs from config")
    if counts["unique_tasks"] != config.expected_tasks:
        raise PipelineConfigError("scan task count differs from config")
    if counts["failed_trajectories"] != 0:
        raise PipelineConfigError("scan contains failed trajectories")
    if scan["context_limit"]["over_limit"] != 0:
        raise PipelineConfigError("one or more trajectories exceed the model limit")
    selected = select_max_seq_length(config, int(scan["token_lengths"]["max"]))
    return scan, selected


def write_pipeline_summary(
    *,
    config_path: Path,
    config: PipelineConfig,
    scan: dict[str, Any],
    max_seq_length: int,
) -> Path:
    from agentic_rl.preprocessing.storage import validate_dataset_directory

    validated = validate_dataset_directory(config.output_dir, verify_hashes=True)
    manifest_path = config.output_dir / "manifest.json"
    summary = {
        "pipeline": config.pipeline_name,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "source": {
            "path": str(config.source_data),
            "sha256": sha256_file(config.source_data),
        },
        "tokenizer_path": str(config.tokenizer_path),
        "counts": scan["counts"],
        "tokens": scan["totals"],
        "selected_max_seq_length": max_seq_length,
        "length_rounding": config.length_rounding,
        "truncation": False,
        "output": {
            "directory": str(config.output_dir),
            "manifest_sha256": sha256_file(manifest_path),
            "validation": validated,
        },
    }
    path = config.audit_dir / "pipeline_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the audited Qwen SFT preprocessing pipeline."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print planned stages without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_pipeline_config(config_path)
    commands = sample_commands(config)

    print(f"Pipeline: {config.pipeline_name}")
    print(f"Source: {config.source_data}")
    print(f"Tokenizer: {config.tokenizer_path}")
    print(f"Output: {config.output_dir}")
    if args.dry_run:
        print("Dry run: paths and tokenizer are not opened.")
        for command in commands:
            print_command(command)
        print("$ [select max_seq_length from checkpoint5 scan]")
        print_command(build_command(config, config.max_model_length))
        print("$ [validate final arrays, hashes, manifest, and write summary]")
        return

    preflight(config)
    for command in commands:
        run_command(command)
    scan, max_seq_length = validate_scan(config)
    print(f"Selected max_seq_length: {max_seq_length}")
    run_command(build_command(config, max_seq_length))
    summary_path = write_pipeline_summary(
        config_path=config_path,
        config=config,
        scan=scan,
        max_seq_length=max_seq_length,
    )
    print(f"Pipeline summary: {summary_path}")


if __name__ == "__main__":
    main()
