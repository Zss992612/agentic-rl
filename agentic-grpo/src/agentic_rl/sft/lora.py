"""LoRA target validation and PEFT adapter injection."""

from __future__ import annotations

from dataclasses import dataclass

from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model
from torch import nn

from agentic_rl.sft.config import LoraConfig


class LoRASetupError(RuntimeError):
    """Raised when LoRA targets or resulting trainable parameters are invalid."""


@dataclass(frozen=True)
class LoRAReport:
    enabled: bool
    matched_modules: dict[str, int]
    total_parameters: int
    trainable_parameters: int
    trainable_fraction: float
    trainable_parameter_names: tuple[str, ...]


def find_target_modules(
    model: nn.Module,
    target_modules: tuple[str, ...],
) -> dict[str, int]:
    counts = {target: 0 for target in target_modules}
    for module_name, _module in model.named_modules():
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name in counts:
            counts[leaf_name] += 1
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise LoRASetupError(
            "LoRA target modules were not found in the model: "
            + ", ".join(missing)
        )
    return counts


def parameter_report(
    model: nn.Module,
    *,
    enabled: bool,
    matched_modules: dict[str, int],
) -> LoRAReport:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if total <= 0:
        raise LoRASetupError("model has no parameters")
    if trainable <= 0:
        raise LoRASetupError("model has no trainable parameters after LoRA setup")
    return LoRAReport(
        enabled=enabled,
        matched_modules=dict(matched_modules),
        total_parameters=total,
        trainable_parameters=trainable,
        trainable_fraction=trainable / total,
        trainable_parameter_names=trainable_names,
    )


def apply_lora(
    model: nn.Module,
    config: LoraConfig,
    *,
    gradient_checkpointing: bool = False,
) -> tuple[nn.Module, LoRAReport]:
    """Inject LoRA adapters and return the wrapped model plus an audit report."""

    if not config.enabled:
        report = parameter_report(model, enabled=False, matched_modules={})
        return model, report

    matched_modules = find_target_modules(model, config.target_modules)
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias=config.bias,
        target_modules=list(config.target_modules),
    )
    wrapped = get_peft_model(model, peft_config)
    if gradient_checkpointing and hasattr(wrapped, "enable_input_require_grads"):
        wrapped.enable_input_require_grads()
    report = parameter_report(
        wrapped,
        enabled=True,
        matched_modules=matched_modules,
    )
    if config.bias == "none" and any(
        "lora_" not in name for name in report.trainable_parameter_names
    ):
        unexpected = [
            name
            for name in report.trainable_parameter_names
            if "lora_" not in name
        ]
        raise LoRASetupError(
            "unexpected non-LoRA trainable parameters: " + ", ".join(unexpected)
        )
    return wrapped, report

