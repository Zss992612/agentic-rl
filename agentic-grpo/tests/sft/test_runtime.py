from __future__ import annotations

from pathlib import Path

import pytest
import torch

from agentic_rl.sft.config import ModelConfig, RuntimeConfig
from agentic_rl.sft.runtime import (
    RuntimeSetupError,
    configure_runtime,
    move_batch_to_device,
    seed_everything,
    torch_dtype,
)


def model_config(dtype: str = "float32") -> ModelConfig:
    return ModelConfig(
        path=Path("/unused"),
        dtype=dtype,  # type: ignore[arg-type]
        attention_implementation="eager",
        trust_remote_code=False,
        use_cache=False,
        gradient_checkpointing=False,
    )


def test_cpu_runtime_and_batch_movement() -> None:
    state = configure_runtime(
        RuntimeConfig(
            device="cpu",
            tf32=True,
            compile=False,
            deterministic=False,
        ),
        model_config(),
        seed=42,
    )

    batch = move_batch_to_device({"input_ids": torch.tensor([[1, 2]])}, state.device)

    assert state.device.type == "cpu"
    assert state.model_dtype == torch.float32
    assert not state.tf32_enabled
    assert batch["input_ids"].device.type == "cpu"
    with state.autocast():
        result = batch["input_ids"] + 1
    assert result.tolist() == [[2, 3]]


def test_seed_everything_is_reproducible() -> None:
    seed_everything(7)
    first = torch.rand(4)
    seed_everything(7)
    second = torch.rand(4)

    assert torch.equal(first, second)


def test_runtime_rejects_invalid_local_modes() -> None:
    assert torch_dtype("bfloat16") == torch.bfloat16
    with pytest.raises(RuntimeSetupError, match="float16 training on CPU"):
        configure_runtime(
            RuntimeConfig(
                device="cpu",
                tf32=False,
                compile=False,
                deterministic=False,
            ),
            model_config("float16"),
            seed=1,
        )
    with pytest.raises(RuntimeSetupError, match="seed"):
        seed_everything(-1)

