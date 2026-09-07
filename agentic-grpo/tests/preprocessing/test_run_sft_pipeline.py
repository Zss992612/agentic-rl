from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/preprocessing/run_sft_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("run_sft_pipeline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_config(path: Path, **overrides: object) -> Path:
    values: dict[str, object] = {
        "pipeline_name": "test_pipeline",
        "source_data": "data/source.jsonl",
        "tokenizer_path": "artifacts/tokenizer",
        "output_dir": "data/tokenized/output",
        "audit_dir": "results/preprocessing/test",
        "expected_trajectories": 256,
        "expected_tasks": 64,
        "max_model_length": 32768,
        "max_seq_length": "auto",
        "length_rounding": 512,
        "sample_index": 0,
        "allow_truncation": False,
    }
    values.update(overrides)
    import yaml

    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def test_auto_sequence_length_rounds_up() -> None:
    config = MODULE.PipelineConfig(
        pipeline_name="test",
        source_data=Path("source"),
        tokenizer_path=Path("tokenizer"),
        output_dir=Path("output"),
        audit_dir=Path("audit"),
        expected_trajectories=256,
        expected_tasks=64,
        max_model_length=32768,
        max_seq_length="auto",
        length_rounding=512,
        sample_index=0,
        allow_truncation=False,
    )

    assert MODULE.select_max_seq_length(config, 16904) == 17408


def test_config_resolves_paths_from_training_root(tmp_path: Path) -> None:
    config = MODULE.load_pipeline_config(write_config(tmp_path / "pipeline.yaml"))

    assert config.source_data == (MODULE.TRAINING_DIR / "data/source.jsonl").resolve()
    assert config.max_seq_length == "auto"
    assert config.allow_truncation is False


def test_rejects_truncation(tmp_path: Path) -> None:
    path = write_config(tmp_path / "pipeline.yaml", allow_truncation=True)

    with pytest.raises(MODULE.PipelineConfigError, match="must be false"):
        MODULE.load_pipeline_config(path)


def test_commands_use_model_specific_output_paths(tmp_path: Path) -> None:
    config = MODULE.load_pipeline_config(write_config(tmp_path / "pipeline.yaml"))
    commands = MODULE.sample_commands(config)

    flattened = "\n".join(" ".join(command) for command in commands)
    assert "inspect_rendered_trajectory.py" in flattened
    assert "inspect_tokenized_trajectory.py" in flattened
    assert "inspect_assistant_labels.py" in flattened
    assert "scan_sft_dataset.py" in flattened
    assert str(config.audit_dir) in flattened


def test_explicit_sequence_length_must_cover_scan(tmp_path: Path) -> None:
    config = MODULE.load_pipeline_config(
        write_config(tmp_path / "pipeline.yaml", max_seq_length=16384)
    )

    with pytest.raises(MODULE.PipelineConfigError, match="scanned maximum"):
        MODULE.select_max_seq_length(config, 16904)
