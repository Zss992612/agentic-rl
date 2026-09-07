"""Tokenizer/model loading and compatibility checks for pretokenized SFT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from agentic_rl.preprocessing.storage import sha256_file
from agentic_rl.sft.config import ModelConfig
from agentic_rl.sft.dataset import TokenizedSFTDataset
from agentic_rl.sft.runtime import torch_dtype


class ModelSetupError(RuntimeError):
    """Raised when model loading or tokenizer compatibility is unsafe."""


@dataclass(frozen=True)
class TokenizerContractReport:
    tokenizer_class: str
    tokenizer_vocab_size: int
    tokenizer_length: int
    model_vocab_size: int
    minimum_dataset_token_id: int
    maximum_dataset_token_id: int
    pad_token_id: int
    eos_token_id: int | None
    tokenizer_json_sha256: str
    tokenizer_config_sha256: str


def load_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.path,
        trust_remote_code=model_config.trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        raise ModelSetupError("model tokenizer has no pad_token_id")
    tokenizer.padding_side = "right"
    return tokenizer


def load_architecture_config(model_config: ModelConfig) -> PretrainedConfig:
    return AutoConfig.from_pretrained(
        model_config.path,
        trust_remote_code=model_config.trust_remote_code,
        local_files_only=True,
    )


def _required_file(path: Path, name: str) -> Path:
    file_path = path / name
    if not file_path.is_file():
        raise ModelSetupError(f"model directory is missing {name}: {path}")
    return file_path


def validate_tokenizer_contract(
    tokenizer: PreTrainedTokenizerBase,
    architecture_config: PretrainedConfig,
    dataset: TokenizedSFTDataset,
    *,
    model_path: str | Path,
) -> TokenizerContractReport:
    """Prove that persisted token IDs use the tokenizer shipped with the model."""

    resolved_model_path = Path(model_path).expanduser().resolve()
    tokenizer_json = _required_file(resolved_model_path, "tokenizer.json")
    tokenizer_config = _required_file(
        resolved_model_path, "tokenizer_config.json"
    )
    actual_tokenizer_hash = sha256_file(tokenizer_json)
    actual_config_hash = sha256_file(tokenizer_config)
    expected = dataset.manifest.get("tokenizer", {})
    if actual_tokenizer_hash != expected.get("tokenizer_json_sha256"):
        raise ModelSetupError(
            "model tokenizer.json differs from the tokenizer used for preprocessing"
        )
    if actual_config_hash != expected.get("tokenizer_config_sha256"):
        raise ModelSetupError(
            "model tokenizer_config.json differs from preprocessing"
        )

    model_vocab_size = getattr(architecture_config, "vocab_size", None)
    if not isinstance(model_vocab_size, int) or model_vocab_size <= 0:
        raise ModelSetupError("model config has no valid vocab_size")
    if len(tokenizer) > model_vocab_size:
        raise ModelSetupError(
            f"tokenizer length {len(tokenizer)} exceeds model vocab size "
            f"{model_vocab_size}"
        )

    token_array = np.load(
        dataset.directory / "input_ids.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    minimum_token_id = int(token_array.min())
    maximum_token_id = int(token_array.max())
    if minimum_token_id < 0:
        raise ModelSetupError("dataset contains a negative input token ID")
    if maximum_token_id >= len(tokenizer):
        raise ModelSetupError(
            f"dataset token ID {maximum_token_id} is outside tokenizer length "
            f"{len(tokenizer)}"
        )
    if maximum_token_id >= model_vocab_size:
        raise ModelSetupError(
            f"dataset token ID {maximum_token_id} is outside model vocab size "
            f"{model_vocab_size}"
        )
    if tokenizer.pad_token_id is None:
        raise ModelSetupError("tokenizer has no pad_token_id")

    return TokenizerContractReport(
        tokenizer_class=tokenizer.__class__.__name__,
        tokenizer_vocab_size=tokenizer.vocab_size,
        tokenizer_length=len(tokenizer),
        model_vocab_size=model_vocab_size,
        minimum_dataset_token_id=minimum_token_id,
        maximum_dataset_token_id=maximum_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tokenizer_json_sha256=actual_tokenizer_hash,
        tokenizer_config_sha256=actual_config_hash,
    )


def load_base_model(
    model_config: ModelConfig,
    *,
    device: torch.device | None = None,
) -> PreTrainedModel:
    """Load the causal LM and apply memory-related model settings."""

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.path,
            torch_dtype=torch_dtype(model_config.dtype),
            attn_implementation=model_config.attention_implementation,
            trust_remote_code=model_config.trust_remote_code,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    except Exception as error:
        raise ModelSetupError(f"failed to load base model: {error}") from error

    model.config.use_cache = model_config.use_cache
    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if device is not None:
        model.to(device)
    return model

