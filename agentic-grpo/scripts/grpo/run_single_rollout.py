"""Launch one validation-only τ² Retail rollout through VERL and vLLM."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/grpo/smoke_single_rollout.yaml"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, Path, list, dict)):
        return json.dumps(value if isinstance(value, (list, dict)) else str(value))
    return str(value)


def _override(name: str, value: Any) -> str:
    return f"{name}={_hydra_value(value)}"


def build_command(config: dict[str, Any]) -> list[str]:
    paths = config["paths"]
    data = config["data"]
    model = config["model"]
    rollout = config["rollout"]
    trainer = config["trainer"]

    model_path = _required_env(paths["model_env"])
    adapter_path = _required_env(paths["lora_adapter_env"])
    _required_env("TAU2_USER_LLM")
    _required_env("TAU2_USER_API_BASE")

    task_file = (PROJECT_ROOT / paths["task_file"]).resolve()
    template_path = (PROJECT_ROOT / paths["chat_template"]).resolve()
    agent_loop_path = (PROJECT_ROOT / paths["agent_loop_config"]).resolve()
    output_dir = (PROJECT_ROOT / paths["output_dir"]).resolve()
    if not task_file.exists():
        raise FileNotFoundError(
            f"Smoke dataset is missing; run build_smoke_tasks.py first: {task_file}"
        )

    os.environ["AGENTIC_RL_CHAT_TEMPLATE"] = template_path.read_text(encoding="utf-8")
    os.environ["AGENTIC_RL_TRAJECTORY_LOG_DIR"] = str(output_dir / "trajectories")
    overrides = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "data.train_files": task_file,
        "data.val_files": task_file,
        "data.train_batch_size": 1,
        "data.val_batch_size": 1,
        "data.max_prompt_length": data["max_prompt_length"],
        "data.max_response_length": data["max_response_length"],
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "data.shuffle": False,
        "actor_rollout_ref.model.path": model_path,
        "actor_rollout_ref.model.custom_chat_template": (
            "${oc.env:AGENTIC_RL_CHAT_TEMPLATE}"
        ),
        "actor_rollout_ref.model.lora_adapter_path": adapter_path,
        "actor_rollout_ref.model.lora_rank": model["lora_rank"],
        "actor_rollout_ref.model.lora_alpha": model["lora_alpha"],
        "actor_rollout_ref.model.target_modules": model["target_modules"],
        "+actor_rollout_ref.model.override_config.attn_implementation": model[
            "attn_implementation"
        ],
        "actor_rollout_ref.model.use_remove_padding": model["use_remove_padding"],
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.ppo_mini_batch_size": 1,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.load_format": "safetensors",
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.temperature": rollout["temperature"],
        "actor_rollout_ref.rollout.top_p": rollout["top_p"],
        "actor_rollout_ref.rollout.top_k": rollout["top_k"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": rollout[
            "tensor_parallel_size"
        ],
        "actor_rollout_ref.rollout.gpu_memory_utilization": rollout[
            "gpu_memory_utilization"
        ],
        "actor_rollout_ref.rollout.max_model_len": rollout["max_model_len"],
        "actor_rollout_ref.rollout.max_num_batched_tokens": rollout[
            "max_num_batched_tokens"
        ],
        "actor_rollout_ref.rollout.max_num_seqs": rollout["max_num_seqs"],
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": rollout[
            "log_prob_micro_batch_size_per_gpu"
        ],
        "actor_rollout_ref.rollout.enable_chunked_prefill": True,
        "actor_rollout_ref.rollout.enable_prefix_caching": True,
        "actor_rollout_ref.rollout.calculate_log_probs": False,
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "actor_rollout_ref.rollout.multi_turn.format": "hermes",
        "actor_rollout_ref.rollout.agent.default_agent_loop": "tau2_retail",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": agent_loop_path,
        "actor_rollout_ref.rollout.agent.num_workers": rollout["agent_loop_workers"],
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.rollout.val_kwargs.temperature": rollout["temperature"],
        "actor_rollout_ref.rollout.val_kwargs.top_p": rollout["top_p"],
        "actor_rollout_ref.rollout.val_kwargs.top_k": rollout["top_k"],
        "reward.reward_model.enable": False,
        "trainer.project_name": "agentic-rl",
        "trainer.experiment_name": config["experiment"]["name"],
        "trainer.logger": ["console"],
        "trainer.n_gpus_per_node": trainer["n_gpus_per_node"],
        "trainer.nnodes": trainer["nnodes"],
        "trainer.default_local_dir": output_dir,
        "trainer.validation_data_dir": output_dir / "validation",
        "trainer.log_val_generations": 1,
        "trainer.val_before_train": True,
        "trainer.val_only": True,
        "trainer.resume_mode": "disable",
        "trainer.save_freq": -1,
        "trainer.test_freq": -1,
        "trainer.total_epochs": 1,
    }
    return [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        *[_override(name, value) for name, value in overrides.items()],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    command = build_command(config)
    if args.dry_run:
        print("\n".join(command))
        return
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
