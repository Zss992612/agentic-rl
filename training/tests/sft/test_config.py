from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentic_rl.sft.config import (
    SFTConfigError,
    load_sft_config,
    save_resolved_config,
)

TRAINING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)
MODEL_PATH = Path("/Users/projects/models/Qwen3-4B-Instruct-2507")


def test_loads_project_sft_config() -> None:
    config = load_sft_config(
        CONFIG_PATH,
        environ={"AGENTIC_RL_MODEL_PATH": str(MODEL_PATH)},
    )

    assert config.experiment.name == "retail_qwen3_4b_instruct_2507_lora_v1"
    assert config.experiment.run_dir == (
        TRAINING_ROOT
        / "outputs/sft/retail_qwen3_4b_instruct_2507_lora_v1"
    )
    assert config.model.path == MODEL_PATH
    assert config.data.max_seq_length == 17_408
    assert config.optimization.effective_batch_size == 8
    assert config.lora.target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def test_missing_model_environment_variable_is_rejected() -> None:
    with pytest.raises(SFTConfigError, match="AGENTIC_RL_MODEL_PATH"):
        load_sft_config(CONFIG_PATH, environ={}, validate_paths=False)


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["optimization"]["typo_learning_rate"] = 1e-4
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(SFTConfigError, match="unexpected keys"):
        load_sft_config(
            path,
            environ={"AGENTIC_RL_MODEL_PATH": str(MODEL_PATH)},
            validate_paths=False,
        )


def test_dataset_sequence_contract_mismatch_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["data"]["max_seq_length"] = 16_384
    path = tmp_path / "invalid_length.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(SFTConfigError, match="dataset contract"):
        load_sft_config(
            path,
            environ={"AGENTIC_RL_MODEL_PATH": str(MODEL_PATH)},
        )


def test_gradient_checkpointing_rejects_use_cache(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["model"]["use_cache"] = True
    path = tmp_path / "invalid_cache.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(SFTConfigError, match="use_cache=false"):
        load_sft_config(
            path,
            environ={"AGENTIC_RL_MODEL_PATH": str(MODEL_PATH)},
        )


def test_resolved_config_can_be_reloaded_without_environment(tmp_path: Path) -> None:
    config = load_sft_config(
        CONFIG_PATH,
        environ={"AGENTIC_RL_MODEL_PATH": str(MODEL_PATH)},
    )
    resolved_path = tmp_path / "resolved.yaml"
    save_resolved_config(config, resolved_path)

    reloaded = load_sft_config(resolved_path, environ={})

    assert reloaded.model.path == config.model.path
    assert reloaded.data.path == config.data.path
    assert reloaded.optimization == config.optimization
