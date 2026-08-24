"""Supervised fine-tuning framework for agentic-rl."""

from agentic_rl.sft.collator import CollationError, DynamicPaddingCollator
from agentic_rl.sft.checkpoint import (
    CheckpointCallback,
    CheckpointError,
    CheckpointManager,
)
from agentic_rl.sft.config import SFTConfig, SFTConfigError, load_sft_config
from agentic_rl.sft.dataloader import SFTDataPipeline, build_train_dataloader
from agentic_rl.sft.dataset import (
    SFTDatasetError,
    SFTSample,
    TokenizedSFTDataset,
    TrajectoryInfo,
)
from agentic_rl.sft.lora import LoRAReport, LoRASetupError, apply_lora
from agentic_rl.sft.modeling import (
    ModelSetupError,
    TokenizerContractReport,
    load_architecture_config,
    load_base_model,
    load_tokenizer,
    validate_tokenizer_contract,
)
from agentic_rl.sft.optimization import (
    OptimizationBundle,
    OptimizationSetupError,
    TrainingSchedule,
    build_optimization,
)
from agentic_rl.sft.runtime import (
    RuntimeSetupError,
    RuntimeState,
    configure_runtime,
    move_batch_to_device,
)
from agentic_rl.sft.sampler import BatchSamplerError, LengthGroupedBatchSampler
from agentic_rl.sft.trainer import (
    OptimizerStepMetrics,
    SFTTrainer,
    SFTTrainerCallback,
    TrainerError,
    TrainerState,
)
from agentic_rl.sft.tracking import (
    ConsoleMetricsCallback,
    TrackingError,
    WandbMetricsCallback,
)

__all__ = [
    "BatchSamplerError",
    "CollationError",
    "CheckpointCallback",
    "CheckpointError",
    "CheckpointManager",
    "ConsoleMetricsCallback",
    "DynamicPaddingCollator",
    "LengthGroupedBatchSampler",
    "LoRAReport",
    "LoRASetupError",
    "ModelSetupError",
    "OptimizationBundle",
    "OptimizationSetupError",
    "OptimizerStepMetrics",
    "RuntimeSetupError",
    "RuntimeState",
    "SFTConfig",
    "SFTDataPipeline",
    "SFTConfigError",
    "SFTDatasetError",
    "SFTSample",
    "SFTTrainer",
    "SFTTrainerCallback",
    "TokenizedSFTDataset",
    "TrajectoryInfo",
    "TrainingSchedule",
    "TrainerError",
    "TrainerState",
    "TrackingError",
    "TokenizerContractReport",
    "apply_lora",
    "build_optimization",
    "build_train_dataloader",
    "configure_runtime",
    "load_architecture_config",
    "load_base_model",
    "load_sft_config",
    "load_tokenizer",
    "move_batch_to_device",
    "validate_tokenizer_contract",
    "WandbMetricsCallback",
]
