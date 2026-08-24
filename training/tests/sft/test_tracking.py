from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from agentic_rl.sft.config import load_sft_config
from agentic_rl.sft.tracking import WandbMetricsCallback
from agentic_rl.sft.trainer import OptimizerStepMetrics, TrainerState

TRAINING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)
MODEL_PATH = Path("/Users/projects/models/Qwen3-4B-Instruct-2507")


class FakeRun:
    def __init__(self) -> None:
        self.defined: list[tuple[tuple, dict]] = []
        self.logged: list[dict] = []
        self.exit_codes: list[int] = []

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def log(self, payload):
        self.logged.append(dict(payload))

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)


class FakeWandb:
    class util:
        @staticmethod
        def generate_id():
            return "fixed-run-id"

    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_kwargs = None
        self.watched = []

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def watch(self, model, **kwargs):
        self.watched.append((model, kwargs))


def test_wandb_callback_initializes_logs_and_finishes(tmp_path: Path) -> None:
    config = load_sft_config(
        CONFIG_PATH,
        environ={"AGENTIC_RL_MODEL_PATH": str(MODEL_PATH)},
    )
    config = replace(
        config,
        experiment=replace(config.experiment, output_root=tmp_path),
        tracking=replace(
            config.tracking,
            wandb=replace(config.tracking.wandb, log_steps=2),
        ),
    )
    fake_wandb = FakeWandb()
    callback = WandbMetricsCallback(
        config=config,
        run_dir=config.experiment.run_dir,
        run_metadata={"dataset": "test"},
        wandb_module=fake_wandb,
    )
    trainer = SimpleNamespace(
        config=config,
        model=object(),
        state=TrainerState(
            micro_step=4,
            optimizer_step=2,
            samples_seen=4,
            input_tokens_seen=400,
            supervised_tokens_seen=80,
        ),
    )
    step_one = OptimizerStepMetrics(
        epoch=0,
        optimizer_step=1,
        micro_steps_in_window=2,
        samples_in_window=2,
        input_tokens_in_window=200,
        supervised_tokens_in_window=40,
        token_normalized_loss=2.0,
        gradient_norm_before_clip=1.5,
        learning_rate_used=1e-4,
        learning_rate_next=2e-4,
        elapsed_seconds=2.0,
    )
    step_two = replace(step_one, optimizer_step=2, token_normalized_loss=1.5)

    callback.on_train_start(trainer)  # type: ignore[arg-type]
    callback.on_optimizer_step(trainer, step_one)  # type: ignore[arg-type]
    callback.on_optimizer_step(trainer, step_two)  # type: ignore[arg-type]
    callback.on_train_end(trainer)  # type: ignore[arg-type]

    assert fake_wandb.init_kwargs["id"] == "fixed-run-id"
    assert fake_wandb.init_kwargs["resume"] == "allow"
    assert fake_wandb.init_kwargs["config"]["run_metadata"] == {"dataset": "test"}
    assert len(fake_wandb.run.logged) == 1
    assert fake_wandb.run.logged[0]["train/optimizer_step"] == 2
    assert fake_wandb.run.logged[0]["train/loss"] == 1.5
    assert fake_wandb.run.logged[0]["train/input_tokens_per_second"] == 100.0
    assert fake_wandb.run.exit_codes == [0]
    assert (config.experiment.run_dir / "wandb_run_id.txt").read_text().strip() == "fixed-run-id"

