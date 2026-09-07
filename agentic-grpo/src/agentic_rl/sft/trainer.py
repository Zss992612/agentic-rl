"""Manual single-device SFT training loop with token-normalized gradients."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from agentic_rl.sft.config import SFTConfig
from agentic_rl.sft.dataloader import SFTDataPipeline
from agentic_rl.sft.optimization import OptimizationBundle
from agentic_rl.sft.runtime import RuntimeState, move_batch_to_device


class TrainerError(RuntimeError):
    """Raised when the training loop encounters an invalid state or value."""


@dataclass
class TrainerState:
    completed_epochs: int = 0
    current_epoch: int = 0
    micro_step: int = 0
    optimizer_step: int = 0
    micro_batches_in_current_epoch: int = 0
    samples_seen: int = 0
    input_tokens_seen: int = 0
    supervised_tokens_seen: int = 0

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, values: Mapping[str, Any]) -> TrainerState:
        expected = set(cls.__dataclass_fields__)
        if set(values) != expected:
            missing = expected - set(values)
            unexpected = set(values) - expected
            details: list[str] = []
            if missing:
                details.append(f"missing={sorted(missing)}")
            if unexpected:
                details.append(f"unexpected={sorted(unexpected)}")
            raise TrainerError("invalid trainer state: " + ", ".join(details))
        try:
            state = cls(**{name: int(values[name]) for name in expected})
        except (TypeError, ValueError) as error:
            raise TrainerError("trainer state contains non-integer values") from error
        if any(value < 0 for value in state.state_dict().values()):
            raise TrainerError("trainer state values must be non-negative")
        return state


@dataclass(frozen=True)
class OptimizerStepMetrics:
    epoch: int
    optimizer_step: int
    micro_steps_in_window: int
    samples_in_window: int
    input_tokens_in_window: int
    supervised_tokens_in_window: int
    token_normalized_loss: float
    gradient_norm_before_clip: float
    learning_rate_used: float
    learning_rate_next: float
    elapsed_seconds: float


class SFTTrainerCallback:
    """No-op lifecycle interface for later checkpoint and W&B callbacks."""

    def on_train_start(self, trainer: SFTTrainer) -> None:
        pass

    def on_optimizer_step(
        self,
        trainer: SFTTrainer,
        metrics: OptimizerStepMetrics,
    ) -> None:
        pass

    def on_epoch_end(self, trainer: SFTTrainer, epoch: int) -> None:
        pass

    def on_train_end(self, trainer: SFTTrainer) -> None:
        pass


def shifted_supervised_token_count(labels: torch.Tensor) -> int:
    """Count labels consumed after the causal LM's internal one-token shift."""

    if labels.ndim != 2 or labels.shape[1] < 2:
        raise TrainerError("labels must have shape [batch, sequence>=2]")
    return int((labels[:, 1:] != -100).sum().item())


def scale_gradients(parameters: Iterable[nn.Parameter], denominator: int) -> None:
    """Convert accumulated token-loss sums into a token-mean gradient."""

    if denominator <= 0:
        raise TrainerError("gradient denominator must be positive")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


class SFTTrainer:
    """Train an assistant-only causal LM on one device without TRL.

    Hugging Face causal LMs return a mean over valid shifted labels.  For each
    micro-batch this trainer multiplies that mean by its supervised-token count,
    accumulates the resulting token-loss sums, and divides gradients by the
    complete accumulation window's token count before clipping and updating.
    This gives every supervised assistant token equal objective weight.
    """

    def __init__(
        self,
        *,
        config: SFTConfig,
        model: nn.Module,
        data_pipeline: SFTDataPipeline,
        optimization: OptimizationBundle,
        runtime: RuntimeState,
        callbacks: Sequence[SFTTrainerCallback] = (),
        state: TrainerState | None = None,
    ) -> None:
        if len(data_pipeline.dataloader) <= 0:
            raise TrainerError("training DataLoader is empty")
        schedule = optimization.schedule
        if schedule.micro_batches_per_epoch != len(data_pipeline.dataloader):
            raise TrainerError(
                "optimization schedule and DataLoader lengths differ: "
                f"{schedule.micro_batches_per_epoch} != {len(data_pipeline.dataloader)}"
            )
        if (
            schedule.gradient_accumulation_steps
            != config.optimization.gradient_accumulation_steps
        ):
            raise TrainerError("gradient accumulation configuration mismatch")

        self.config = config
        self.model = model
        self.data_pipeline = data_pipeline
        self.optimization = optimization
        self.runtime = runtime
        self.callbacks = tuple(callbacks)
        self.state = state or TrainerState()
        self.last_step_metrics: OptimizerStepMetrics | None = None

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.optimization.optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self.optimization.scheduler

    def _emit(self, method_name: str, *args: Any) -> None:
        for callback in self.callbacks:
            method = getattr(callback, method_name)
            method(self, *args)

    def _model_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        with self.runtime.autocast():
            outputs = self.model(**batch)
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, Mapping):
            loss = outputs.get("loss")
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise TrainerError("model output has no scalar loss tensor")
        if not torch.isfinite(loss).item():
            raise TrainerError(f"model produced non-finite loss: {loss}")
        return loss

    def _optimizer_update(
        self,
        *,
        epoch: int,
        micro_steps_in_window: int,
        samples_in_window: int,
        input_tokens_in_window: int,
        supervised_tokens_in_window: int,
        token_loss_sum: float,
        window_start: float,
        completes_epoch: bool,
    ) -> OptimizerStepMetrics:
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise TrainerError("model has no trainable parameters")
        scale_gradients(trainable_parameters, supervised_tokens_in_window)
        gradient_norm = nn.utils.clip_grad_norm_(
            trainable_parameters,
            self.config.optimization.max_grad_norm,
            error_if_nonfinite=True,
        )
        learning_rate_used = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.optimizer_step += 1
        if completes_epoch:
            self.state.completed_epochs = epoch + 1
            self.state.micro_batches_in_current_epoch = 0
        metrics = OptimizerStepMetrics(
            epoch=epoch,
            optimizer_step=self.state.optimizer_step,
            micro_steps_in_window=micro_steps_in_window,
            samples_in_window=samples_in_window,
            input_tokens_in_window=input_tokens_in_window,
            supervised_tokens_in_window=supervised_tokens_in_window,
            token_normalized_loss=token_loss_sum / supervised_tokens_in_window,
            gradient_norm_before_clip=float(gradient_norm.item()),
            learning_rate_used=learning_rate_used,
            learning_rate_next=float(self.optimizer.param_groups[0]["lr"]),
            elapsed_seconds=time.perf_counter() - window_start,
        )
        self.last_step_metrics = metrics
        self._emit("on_optimizer_step", metrics)
        return metrics

    def train(self, *, max_optimizer_steps: int | None = None) -> TrainerState:
        """Run training from the current state and return the updated state.

        ``max_optimizer_steps`` is an absolute global-step ceiling used by CPU
        tests and future bounded runs.  ``None`` runs through all configured
        epochs.  Mid-epoch checkpoint resume is intentionally deferred to the
        checkpoint layer rather than silently approximated here.
        """

        if max_optimizer_steps is not None:
            if max_optimizer_steps <= 0:
                raise TrainerError("max_optimizer_steps must be positive")
            if self.state.optimizer_step >= max_optimizer_steps:
                return self.state
        if self.state.completed_epochs > self.config.optimization.epochs:
            raise TrainerError("trainer state exceeds configured epochs")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._emit("on_train_start")
        stop_requested = False
        accumulation_steps = self.config.optimization.gradient_accumulation_steps

        for epoch in range(
            self.state.completed_epochs,
            self.config.optimization.epochs,
        ):
            self.state.current_epoch = epoch
            self.data_pipeline.set_epoch(epoch)
            num_micro_batches = len(self.data_pipeline.dataloader)
            micro_steps_in_window = 0
            samples_in_window = 0
            input_tokens_in_window = 0
            supervised_tokens_in_window = 0
            token_loss_sum = 0.0
            window_start = time.perf_counter()
            epoch_finished = False
            resume_skip = (
                self.state.micro_batches_in_current_epoch
                if epoch == self.state.completed_epochs
                else 0
            )
            if resume_skip >= num_micro_batches:
                raise TrainerError(
                    "resume state has consumed all micro-batches without "
                    "marking the epoch complete"
                )

            for batch_index, cpu_batch in enumerate(self.data_pipeline.dataloader):
                if batch_index < resume_skip:
                    continue
                batch = move_batch_to_device(
                    cpu_batch,
                    self.runtime.device,
                    non_blocking=self.config.data.pin_memory,
                )
                labels = batch.get("labels")
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")
                if labels is None or input_ids is None or attention_mask is None:
                    raise TrainerError(
                        "training batch must contain input_ids, labels, and attention_mask"
                    )
                supervised_tokens = shifted_supervised_token_count(labels)
                if supervised_tokens <= 0:
                    raise TrainerError(
                        f"epoch {epoch} batch {batch_index} has no supervised tokens"
                    )
                loss = self._model_loss(batch)
                (loss * supervised_tokens).backward()

                batch_size = int(input_ids.shape[0])
                input_tokens = int(attention_mask.sum().item())
                self.state.micro_step += 1
                self.state.micro_batches_in_current_epoch = batch_index + 1
                self.state.samples_seen += batch_size
                self.state.input_tokens_seen += input_tokens
                self.state.supervised_tokens_seen += supervised_tokens
                micro_steps_in_window += 1
                samples_in_window += batch_size
                input_tokens_in_window += input_tokens
                supervised_tokens_in_window += supervised_tokens
                token_loss_sum += float(loss.detach().float().item()) * supervised_tokens

                is_last_batch = batch_index + 1 == num_micro_batches
                window_complete = micro_steps_in_window == accumulation_steps
                if window_complete or is_last_batch:
                    self._optimizer_update(
                        epoch=epoch,
                        micro_steps_in_window=micro_steps_in_window,
                        samples_in_window=samples_in_window,
                        input_tokens_in_window=input_tokens_in_window,
                        supervised_tokens_in_window=supervised_tokens_in_window,
                        token_loss_sum=token_loss_sum,
                        window_start=window_start,
                        completes_epoch=is_last_batch,
                    )
                    micro_steps_in_window = 0
                    samples_in_window = 0
                    input_tokens_in_window = 0
                    supervised_tokens_in_window = 0
                    token_loss_sum = 0.0
                    window_start = time.perf_counter()
                    epoch_finished = is_last_batch

                    if (
                        max_optimizer_steps is not None
                        and self.state.optimizer_step >= max_optimizer_steps
                    ):
                        stop_requested = True
                        break
                if is_last_batch:
                    epoch_finished = True

            if epoch_finished:
                self._emit("on_epoch_end", epoch)
            if stop_requested:
                break

        self._emit("on_train_end")
        return self.state
