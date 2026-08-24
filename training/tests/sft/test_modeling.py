from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agentic_rl.sft.config import load_sft_config
from agentic_rl.sft.dataset import TokenizedSFTDataset
from agentic_rl.sft.modeling import (
    ModelSetupError,
    load_architecture_config,
    load_tokenizer,
    validate_tokenizer_contract,
)

TRAINING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)
DATA_PATH = (
    TRAINING_ROOT
    / "data/tokenized/retail_serial_v3_qwen3_4b_instruct_2507"
)
MODEL_PATH = Path("/Users/projects/models/Qwen3-4B-Instruct-2507")


@pytest.fixture(scope="module")
def project_components():
    config = load_sft_config(
        CONFIG_PATH,
        environ={"AGENTIC_RL_MODEL_PATH": str(MODEL_PATH)},
    )
    tokenizer = load_tokenizer(config.model)
    architecture = load_architecture_config(config.model)
    dataset = TokenizedSFTDataset(DATA_PATH, verify_hashes=False)
    return config, tokenizer, architecture, dataset


def test_real_tokenizer_matches_pretokenized_dataset(project_components) -> None:
    config, tokenizer, architecture, dataset = project_components

    report = validate_tokenizer_contract(
        tokenizer,
        architecture,
        dataset,
        model_path=config.model.path,
    )

    assert report.tokenizer_class == "Qwen2TokenizerFast"
    assert report.tokenizer_vocab_size == 151_643
    assert report.tokenizer_length == 151_669
    assert report.model_vocab_size == 151_936
    assert report.minimum_dataset_token_id == 0
    assert report.maximum_dataset_token_id == 151_666
    assert report.pad_token_id == 151_643
    assert report.eos_token_id == 151_645
    assert tokenizer.padding_side == "right"


def test_mismatched_tokenizer_files_are_rejected(
    project_components,
    tmp_path: Path,
) -> None:
    _config, tokenizer, architecture, dataset = project_components
    shutil.copy(MODEL_PATH / "tokenizer.json", tmp_path / "tokenizer.json")
    shutil.copy(
        MODEL_PATH / "tokenizer_config.json",
        tmp_path / "tokenizer_config.json",
    )
    with (tmp_path / "tokenizer_config.json").open("a", encoding="utf-8") as file:
        file.write("\n")

    with pytest.raises(ModelSetupError, match="differs from preprocessing"):
        validate_tokenizer_contract(
            tokenizer,
            architecture,
            dataset,
            model_path=tmp_path,
        )

