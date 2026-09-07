from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from agentic_rl.sft.checkpoint import CheckpointManager
from agentic_rl.sft.config import LoraConfig, load_sft_config
from agentic_rl.sft.lora import apply_lora
from agentic_rl.sft.optimization import build_optimization
from agentic_rl.sft.trainer import TrainerState

TRAINING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)
MODEL_PATH = Path("/Users/projects/models/Qwen3-4B-Instruct-2507")


def tiny_lora_model():
    model = Qwen3ForCausalLM(
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
    lora = LoraConfig(
        enabled=True,
        rank=4,
        alpha=8,
        dropout=0.0,
        bias="none",
        target_modules=("q_proj", "v_proj"),
    )
    wrapped, _report = apply_lora(model, lora)
    return wrapped


def test_checkpoint_round_trip_and_retention(tmp_path: Path) -> None:
    config = load_sft_config(
        CONFIG_PATH,
        environ={"AGENTIC_RL_MODEL_PATH": str(MODEL_PATH)},
    )
    config = replace(
        config,
        experiment=replace(config.experiment, output_root=tmp_path),
        checkpoint=replace(config.checkpoint, save_steps=1, keep_last=2),
    )
    model = tiny_lora_model()
    optimization = build_optimization(
        model,
        replace(config.optimization, epochs=1, gradient_accumulation_steps=1),
        micro_batches_per_epoch=3,
    )
    input_ids = torch.randint(0, 128, (1, 8))
    labels = input_ids.clone()
    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    optimization.optimizer.step()
    optimization.scheduler.step()
    optimization.optimizer.zero_grad(set_to_none=True)
    manager = CheckpointManager(config=config, run_metadata={"test": True})
    probe = next(
        parameter
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    )
    with torch.no_grad():
        probe.fill_(0.125)

    state = TrainerState(optimizer_step=1, micro_step=1)
    step_one = manager.save(
        model=model,
        optimization=optimization,
        trainer_state=state,
    )
    with torch.no_grad():
        probe.zero_()
    restored = manager.load(
        step_one,
        model=model,
        optimization=optimization,
        restore_rng=False,
    )

    assert restored == state
    assert optimization.optimizer.state
    assert torch.allclose(probe, torch.full_like(probe, 0.125))
    assert (step_one / "adapter_model.safetensors").is_file()
    assert (step_one / "trainer_state.json").is_file()
    assert (step_one / "optimizer.pt").is_file()
    assert (step_one / "scheduler.pt").is_file()
    assert (config.experiment.run_dir / "resolved_config.yaml").is_file()

    for step in (2, 3):
        state.optimizer_step = step
        state.micro_step = step
        manager.save(
            model=model,
            optimization=optimization,
            trainer_state=state,
        )

    remaining = sorted(path.name for path in manager.checkpoint_root.iterdir())
    assert remaining == ["step_00000002", "step_00000003"]
    assert manager.latest_checkpoint().name == "step_00000003"
