"""Process-wide runtime setup for reproducible single-device SFT."""

from __future__ import annotations

import random
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from agentic_rl.sft.config import ModelConfig, RuntimeConfig


class RuntimeSetupError(RuntimeError):
    """Raised when the configured device or numeric mode is unavailable."""


_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class RuntimeState:
    device: torch.device
    model_dtype: torch.dtype
    seed: int
    tf32_enabled: bool
    deterministic: bool
    compile_enabled: bool

    def autocast(self) -> AbstractContextManager[None]:
        """Return a fresh autocast context for one forward pass."""

        if self.model_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.model_dtype,
        )


def torch_dtype(name: str) -> torch.dtype:
    """Resolve the validated YAML dtype name to a PyTorch dtype."""

    try:
        return _DTYPES[name]
    except KeyError as error:
        raise RuntimeSetupError(f"unsupported model dtype: {name}") from error


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeSetupError("runtime.device=cuda but CUDA is unavailable")
        return torch.device("cuda", torch.cuda.current_device())
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeSetupError("runtime.device=mps but MPS is unavailable")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise RuntimeSetupError(f"unsupported runtime device: {name}")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and every available PyTorch device."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RuntimeSetupError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_data_worker(worker_id: int) -> None:
    """Seed Python/NumPy inside a DataLoader worker from PyTorch's seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_data_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def configure_runtime(
    runtime_config: RuntimeConfig,
    model_config: ModelConfig,
    *,
    seed: int,
) -> RuntimeState:
    """Validate and apply the process-wide runtime configuration."""

    device = resolve_device(runtime_config.device)
    dtype = torch_dtype(model_config.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise RuntimeSetupError("float16 training on CPU is unsupported")
    if (
        device.type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeSetupError("the selected CUDA device does not support bfloat16")

    seed_everything(seed)
    torch.use_deterministic_algorithms(runtime_config.deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = runtime_config.deterministic
        torch.backends.cudnn.benchmark = not runtime_config.deterministic

    tf32_enabled = device.type == "cuda" and runtime_config.tf32
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled

    return RuntimeState(
        device=device,
        model_dtype=dtype,
        seed=seed,
        tf32_enabled=tf32_enabled,
        deterministic=runtime_config.deterministic,
        compile_enabled=runtime_config.compile,
    )


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    non_blocking: bool = True,
) -> dict[str, torch.Tensor]:
    """Move only tensor batch values to the configured training device."""

    return {
        key: value.to(device=device, non_blocking=non_blocking)
        for key, value in batch.items()
    }

