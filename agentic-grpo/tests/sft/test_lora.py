from __future__ import annotations

import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from agentic_rl.sft.config import LoraConfig
from agentic_rl.sft.lora import LoRASetupError, apply_lora


def tiny_qwen3() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
        )
    )


def lora_config(**overrides) -> LoraConfig:
    values = {
        "enabled": True,
        "rank": 4,
        "alpha": 8,
        "dropout": 0.05,
        "bias": "none",
        "target_modules": (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    }
    values.update(overrides)
    return LoraConfig(**values)


def test_lora_injects_all_qwen_targets_and_freezes_base_model() -> None:
    wrapped, report = apply_lora(
        tiny_qwen3(),
        lora_config(),
        gradient_checkpointing=True,
    )

    assert report.enabled
    assert report.matched_modules == {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 1,
        "gate_proj": 1,
        "up_proj": 1,
        "down_proj": 1,
    }
    assert 0 < report.trainable_parameters < report.total_parameters
    assert all("lora_" in name for name in report.trainable_parameter_names)
    assert all(
        parameter.requires_grad == ("lora_" in name)
        for name, parameter in wrapped.named_parameters()
    )


def test_missing_target_module_is_rejected_before_injection() -> None:
    with pytest.raises(LoRASetupError, match="not found"):
        apply_lora(
            tiny_qwen3(),
            lora_config(target_modules=("not_a_real_projection",)),
        )

