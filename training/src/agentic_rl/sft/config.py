"""Typed configuration contract for supervised fine-tuning.

This module intentionally performs no model loading and imports neither
PyTorch nor PEFT.  Its job is to make every training choice explicit before
the data and model layers are constructed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

TRAINING_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class SFTConfigError(ValueError):
    """Raised when an SFT YAML file violates the configuration contract."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_root: Path

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.name


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    dtype: Literal["bfloat16", "float16", "float32"]
    attention_implementation: Literal["flash_attention_2", "sdpa", "eager"]
    trust_remote_code: bool
    use_cache: bool
    gradient_checkpointing: bool


@dataclass(frozen=True)
class DataConfig:
    path: Path
    max_seq_length: int
    num_workers: int
    pin_memory: bool
    shuffle: bool
    length_bucket: bool
    bucket_size: int


@dataclass(frozen=True)
class LoraConfig:
    enabled: bool
    rank: int
    alpha: int
    dropout: float
    bias: Literal["none", "all", "lora_only"]
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float
    warmup_ratio: float
    scheduler: Literal["linear", "cosine", "constant"]

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class RuntimeConfig:
    device: Literal["cuda", "cpu", "mps"]
    tf32: bool
    compile: bool
    deterministic: bool


@dataclass(frozen=True)
class CheckpointConfig:
    save_steps: int
    keep_last: int
    save_optimizer: bool
    resume_from: Path | None


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    entity: str | None
    log_steps: int
    watch_model: bool


@dataclass(frozen=True)
class TrackingConfig:
    wandb: WandbConfig


@dataclass(frozen=True)
class SFTConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig
    optimization: OptimizationConfig
    runtime: RuntimeConfig
    checkpoint: CheckpointConfig
    tracking: TrackingConfig
    source_path: Path

    def to_serializable_dict(self) -> dict[str, Any]:
        """Return a resolved representation safe for YAML/JSON manifests."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        result = convert(asdict(self))
        result.pop("source_path", None)
        return result


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SFTConfigError(f"{location}: expected a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    keys: set[str],
    location: str,
) -> None:
    missing = keys - set(value)
    unexpected = set(value) - keys
    if missing:
        raise SFTConfigError(
            f"{location}: missing keys: {', '.join(sorted(missing))}"
        )
    if unexpected:
        raise SFTConfigError(
            f"{location}: unexpected keys: {', '.join(sorted(unexpected))}"
        )


def _string(value: Any, location: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SFTConfigError(f"{location}: expected a non-empty string")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise SFTConfigError(f"{location}: expected a boolean")
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SFTConfigError(
            f"{location}: expected an integer >= {minimum}"
        )
    return value


def _number(
    value: Any,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SFTConfigError(f"{location}: expected a number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise SFTConfigError(f"{location}: must be >= {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            comparator = "<=" if maximum_inclusive else "<"
            raise SFTConfigError(f"{location}: must be {comparator} {maximum}")
    return result


def _choice(value: Any, location: str, choices: set[str]) -> str:
    result = _string(value, location)
    assert result is not None
    if result not in choices:
        raise SFTConfigError(
            f"{location}: expected one of {', '.join(sorted(choices))}, got {result!r}"
        )
    return result


def _expand_environment(value: Any, environ: Mapping[str, str], location: str) -> Any:
    if isinstance(value, str):
        missing: list[str] = []

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            replacement = environ.get(name)
            if replacement is None:
                missing.append(name)
                return match.group(0)
            return replacement

        result = _ENV_PATTERN.sub(replace, value)
        if missing:
            raise SFTConfigError(
                f"{location}: environment variable is not set: {', '.join(missing)}"
            )
        return result
    if isinstance(value, list):
        return [
            _expand_environment(item, environ, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            key: _expand_environment(item, environ, f"{location}.{key}")
            for key, item in value.items()
        }
    return value


def _resolve_path(value: Any, location: str, project_root: Path) -> Path:
    text = _string(value, location)
    assert text is not None
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _parse_experiment(value: Any, project_root: Path) -> ExperimentConfig:
    item = _mapping(value, "experiment")
    _exact_keys(
        item,
        keys={"name", "seed", "output_root"},
        location="experiment",
    )
    name = _string(item["name"], "experiment.name")
    assert name is not None
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name):
        raise SFTConfigError(
            "experiment.name: use lowercase letters, digits, dot, dash, or underscore"
        )
    return ExperimentConfig(
        name=name,
        seed=_integer(item["seed"], "experiment.seed"),
        output_root=_resolve_path(
            item["output_root"], "experiment.output_root", project_root
        ),
    )


def _parse_model(value: Any, project_root: Path) -> ModelConfig:
    item = _mapping(value, "model")
    _exact_keys(
        item,
        keys={
            "path",
            "dtype",
            "attention_implementation",
            "trust_remote_code",
            "use_cache",
            "gradient_checkpointing",
        },
        location="model",
    )
    return ModelConfig(
        path=_resolve_path(item["path"], "model.path", project_root),
        dtype=_choice(
            item["dtype"],
            "model.dtype",
            {"bfloat16", "float16", "float32"},
        ),  # type: ignore[arg-type]
        attention_implementation=_choice(
            item["attention_implementation"],
            "model.attention_implementation",
            {"flash_attention_2", "sdpa", "eager"},
        ),  # type: ignore[arg-type]
        trust_remote_code=_boolean(
            item["trust_remote_code"], "model.trust_remote_code"
        ),
        use_cache=_boolean(item["use_cache"], "model.use_cache"),
        gradient_checkpointing=_boolean(
            item["gradient_checkpointing"], "model.gradient_checkpointing"
        ),
    )


def _parse_data(value: Any, project_root: Path) -> DataConfig:
    item = _mapping(value, "data")
    _exact_keys(
        item,
        keys={
            "path",
            "max_seq_length",
            "num_workers",
            "pin_memory",
            "shuffle",
            "length_bucket",
            "bucket_size",
        },
        location="data",
    )
    return DataConfig(
        path=_resolve_path(item["path"], "data.path", project_root),
        max_seq_length=_integer(
            item["max_seq_length"], "data.max_seq_length", minimum=1
        ),
        num_workers=_integer(item["num_workers"], "data.num_workers"),
        pin_memory=_boolean(item["pin_memory"], "data.pin_memory"),
        shuffle=_boolean(item["shuffle"], "data.shuffle"),
        length_bucket=_boolean(item["length_bucket"], "data.length_bucket"),
        bucket_size=_integer(item["bucket_size"], "data.bucket_size", minimum=1),
    )


def _parse_lora(value: Any) -> LoraConfig:
    item = _mapping(value, "lora")
    _exact_keys(
        item,
        keys={"enabled", "rank", "alpha", "dropout", "bias", "target_modules"},
        location="lora",
    )
    raw_modules = item["target_modules"]
    if not isinstance(raw_modules, list) or not raw_modules:
        raise SFTConfigError("lora.target_modules: expected a non-empty list")
    modules = tuple(
        _string(module, f"lora.target_modules[{index}]")
        for index, module in enumerate(raw_modules)
    )
    assert all(module is not None for module in modules)
    typed_modules = tuple(module for module in modules if module is not None)
    if len(set(typed_modules)) != len(typed_modules):
        raise SFTConfigError("lora.target_modules: duplicate module names")
    return LoraConfig(
        enabled=_boolean(item["enabled"], "lora.enabled"),
        rank=_integer(item["rank"], "lora.rank", minimum=1),
        alpha=_integer(item["alpha"], "lora.alpha", minimum=1),
        dropout=_number(
            item["dropout"], "lora.dropout", minimum=0, maximum=1
        ),
        bias=_choice(
            item["bias"], "lora.bias", {"none", "all", "lora_only"}
        ),  # type: ignore[arg-type]
        target_modules=typed_modules,
    )


def _parse_optimization(value: Any) -> OptimizationConfig:
    item = _mapping(value, "optimization")
    _exact_keys(
        item,
        keys={
            "epochs",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "max_grad_norm",
            "warmup_ratio",
            "scheduler",
        },
        location="optimization",
    )
    return OptimizationConfig(
        epochs=_integer(item["epochs"], "optimization.epochs", minimum=1),
        micro_batch_size=_integer(
            item["micro_batch_size"],
            "optimization.micro_batch_size",
            minimum=1,
        ),
        gradient_accumulation_steps=_integer(
            item["gradient_accumulation_steps"],
            "optimization.gradient_accumulation_steps",
            minimum=1,
        ),
        learning_rate=_number(
            item["learning_rate"], "optimization.learning_rate", minimum=0
        ),
        weight_decay=_number(
            item["weight_decay"], "optimization.weight_decay", minimum=0
        ),
        adam_beta1=_number(
            item["adam_beta1"],
            "optimization.adam_beta1",
            minimum=0,
            maximum=1,
            maximum_inclusive=False,
        ),
        adam_beta2=_number(
            item["adam_beta2"],
            "optimization.adam_beta2",
            minimum=0,
            maximum=1,
            maximum_inclusive=False,
        ),
        adam_epsilon=_number(
            item["adam_epsilon"], "optimization.adam_epsilon", minimum=0
        ),
        max_grad_norm=_number(
            item["max_grad_norm"], "optimization.max_grad_norm", minimum=0
        ),
        warmup_ratio=_number(
            item["warmup_ratio"],
            "optimization.warmup_ratio",
            minimum=0,
            maximum=1,
            maximum_inclusive=False,
        ),
        scheduler=_choice(
            item["scheduler"],
            "optimization.scheduler",
            {"linear", "cosine", "constant"},
        ),  # type: ignore[arg-type]
    )


def _parse_runtime(value: Any) -> RuntimeConfig:
    item = _mapping(value, "runtime")
    _exact_keys(
        item,
        keys={"device", "tf32", "compile", "deterministic"},
        location="runtime",
    )
    return RuntimeConfig(
        device=_choice(
            item["device"], "runtime.device", {"cuda", "cpu", "mps"}
        ),  # type: ignore[arg-type]
        tf32=_boolean(item["tf32"], "runtime.tf32"),
        compile=_boolean(item["compile"], "runtime.compile"),
        deterministic=_boolean(item["deterministic"], "runtime.deterministic"),
    )


def _parse_checkpoint(value: Any, project_root: Path) -> CheckpointConfig:
    item = _mapping(value, "checkpoint")
    _exact_keys(
        item,
        keys={"save_steps", "keep_last", "save_optimizer", "resume_from"},
        location="checkpoint",
    )
    raw_resume = item["resume_from"]
    resume_from = (
        None
        if raw_resume is None
        else _resolve_path(raw_resume, "checkpoint.resume_from", project_root)
    )
    return CheckpointConfig(
        save_steps=_integer(
            item["save_steps"], "checkpoint.save_steps", minimum=1
        ),
        keep_last=_integer(item["keep_last"], "checkpoint.keep_last", minimum=1),
        save_optimizer=_boolean(
            item["save_optimizer"], "checkpoint.save_optimizer"
        ),
        resume_from=resume_from,
    )


def _parse_tracking(value: Any) -> TrackingConfig:
    item = _mapping(value, "tracking")
    _exact_keys(item, keys={"wandb"}, location="tracking")
    wandb = _mapping(item["wandb"], "tracking.wandb")
    _exact_keys(
        wandb,
        keys={"enabled", "project", "entity", "log_steps", "watch_model"},
        location="tracking.wandb",
    )
    project = _string(wandb["project"], "tracking.wandb.project")
    entity = _string(wandb["entity"], "tracking.wandb.entity", nullable=True)
    assert project is not None
    return TrackingConfig(
        wandb=WandbConfig(
            enabled=_boolean(wandb["enabled"], "tracking.wandb.enabled"),
            project=project,
            entity=entity,
            log_steps=_integer(
                wandb["log_steps"], "tracking.wandb.log_steps", minimum=1
            ),
            watch_model=_boolean(
                wandb["watch_model"], "tracking.wandb.watch_model"
            ),
        )
    )


def _validate_paths_and_contract(config: SFTConfig) -> None:
    if not config.model.path.is_dir():
        raise SFTConfigError(f"model.path does not exist: {config.model.path}")
    if not (config.model.path / "config.json").is_file():
        raise SFTConfigError(
            f"model.path is not a complete model directory: {config.model.path}"
        )
    if not config.data.path.is_dir():
        raise SFTConfigError(f"data.path does not exist: {config.data.path}")
    manifest_path = config.data.path / "manifest.json"
    stats_path = config.data.path / "stats.json"
    if not manifest_path.is_file() or not stats_path.is_file():
        raise SFTConfigError("data.path is missing manifest.json or stats.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    dataset_limit = manifest.get("sequence_contract", {}).get("max_seq_length")
    if dataset_limit != config.data.max_seq_length:
        raise SFTConfigError(
            "data.max_seq_length does not match the tokenized dataset contract: "
            f"{config.data.max_seq_length} != {dataset_limit}"
        )
    actual_max = stats.get("token_lengths", {}).get("max")
    if not isinstance(actual_max, int) or actual_max > config.data.max_seq_length:
        raise SFTConfigError(
            "tokenized dataset contains a sequence above data.max_seq_length"
        )
    if config.runtime.device != "cuda" and config.model.attention_implementation == "flash_attention_2":
        raise SFTConfigError(
            "flash_attention_2 requires runtime.device=cuda"
        )
    if config.model.gradient_checkpointing and config.model.use_cache:
        raise SFTConfigError(
            "gradient checkpointing requires model.use_cache=false"
        )


def load_sft_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    project_root: Path | None = None,
    validate_paths: bool = True,
) -> SFTConfig:
    """Load, expand, parse, and optionally validate an SFT YAML file."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise SFTConfigError(f"config file does not exist: {source_path}")
    root = (project_root or TRAINING_ROOT).expanduser().resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw = _mapping(raw, "config")
    expanded = _expand_environment(raw, environ or os.environ, "config")
    expanded = _mapping(expanded, "config")
    expected_sections = {
        "experiment",
        "model",
        "data",
        "lora",
        "optimization",
        "runtime",
        "checkpoint",
        "tracking",
    }
    _exact_keys(expanded, keys=expected_sections, location="config")
    config = SFTConfig(
        experiment=_parse_experiment(expanded["experiment"], root),
        model=_parse_model(expanded["model"], root),
        data=_parse_data(expanded["data"], root),
        lora=_parse_lora(expanded["lora"]),
        optimization=_parse_optimization(expanded["optimization"]),
        runtime=_parse_runtime(expanded["runtime"]),
        checkpoint=_parse_checkpoint(expanded["checkpoint"], root),
        tracking=_parse_tracking(expanded["tracking"]),
        source_path=source_path,
    )
    if validate_paths:
        _validate_paths_and_contract(config)
    return config


def save_resolved_config(config: SFTConfig, path: str | Path) -> None:
    """Persist the exact resolved values used by a training run."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            config.to_serializable_dict(),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
