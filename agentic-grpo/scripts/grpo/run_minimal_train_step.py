"""Run GRPO training on tau2 Retail trajectories."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/grpo/minimal_train_step.yaml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{os.getpid()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git_state(path: Path) -> dict[str, Any] | None:
    if not (path / ".git").exists():
        return None
    git = ["git", "-c", f"safe.directory={path}", "-C", str(path)]
    try:
        commit = subprocess.check_output(
            [*git, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        status = subprocess.check_output(
            [*git, "status", "--porcelain"],
            text=True,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        return {
            "path": str(path),
            "available": False,
            "error": (error.stderr or str(error)).strip(),
        }
    return {
        "path": str(path),
        "available": True,
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "vllm",
        "verl",
        "tau2",
        "transformers",
        "flash-attn",
        "peft",
        "wandb",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, dict):
        entries = ",".join(
            f"{key}:{_hydra_value(item)}" for key, item in value.items()
        )
        return "{" + entries + "}"
    if isinstance(value, (str, Path, list)):
        return json.dumps(value if isinstance(value, list) else str(value))
    return str(value)


def _override(name: str, value: Any) -> str:
    return f"{name}={_hydra_value(value)}"


def _latest_checkpoint(output_dir: Path) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    tracker_path = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker_path.is_file():
        raise FileNotFoundError(f"Checkpoint tracker is missing: {tracker_path}")

    global_step = int(tracker_path.read_text(encoding="utf-8").strip())
    return checkpoint_dir / f"global_step_{global_step}"


def _configure_tracking(
    config: dict[str, Any],
    output_dir: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    tracking = config.get("tracking", {})
    backends = tracking.get("backends", ["console", "file"])
    if "wandb" not in backends:
        return {"backends": backends}

    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run_id_path = output_dir / "wandb_run_id.txt"
    if resume and run_id_path.is_file():
        wandb_run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        wandb_run_id = uuid4().hex
        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(wandb_run_id + "\n", encoding="utf-8")

    os.environ["WANDB_RUN_ID"] = wandb_run_id
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_MODE"] = tracking.get("mode", "online")
    os.environ["AGENTIC_RL_WANDB_TRAJECTORY_TABLE_FREQ"] = str(
        tracking.get("trajectory_table_freq", 5)
    )
    if group := tracking.get("group"):
        os.environ["WANDB_RUN_GROUP"] = str(group)
    if tags := tracking.get("tags"):
        os.environ["WANDB_TAGS"] = ",".join(str(tag) for tag in tags)

    return {
        "backends": backends,
        "project": tracking.get("project", "agentic-rl"),
        "group": tracking.get("group"),
        "tags": tracking.get("tags", []),
        "mode": os.environ["WANDB_MODE"],
        "wandb_run_id": wandb_run_id,
        "wandb_dir": str(wandb_dir),
    }


def _resume_adapter_path(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    adapter_path = checkpoint_path / "actor" / "lora_adapter"
    adapter_weights = adapter_path / "adapter_model.safetensors"
    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_weights.is_file() or not adapter_config.is_file():
        raise FileNotFoundError(
            f"Checkpoint LoRA adapter is incomplete: {adapter_path}"
        )
    return adapter_path


def build_command(
    config: dict[str, Any],
    *,
    resume: bool = False,
    resume_from: Path | None = None,
) -> tuple[list[str], Path, Path, dict[str, Path]]:
    if resume and resume_from is not None:
        raise ValueError("--resume and --resume-from cannot be used together")

    paths = config["paths"]
    data = config["data"]
    model = config["model"]
    actor = config["actor"]
    rollout = config["rollout"]
    algorithm = config.get("algorithm", {})
    trainer = config["trainer"]
    tracking = config.get("tracking", {})

    if algorithm.get("bypass_mode", False) and not rollout.get("calculate_log_probs", False):
        raise ValueError("bypass_mode requires rollout.calculate_log_probs=true")
    if model.get("use_remove_padding", False) and model["attn_implementation"] not in {
        "flash_attention_2",
        "flash_attention_3",
    }:
        raise ValueError("use_remove_padding requires FlashAttention")

    use_kl_loss = actor.get("use_kl_loss", False)
    use_kl_in_reward = algorithm.get("use_kl_in_reward", False)
    if use_kl_loss and use_kl_in_reward:
        raise ValueError("Enable KL either in actor loss or reward, not both")
    if use_kl_loss and paths.get("lora_adapter_env"):
        raise ValueError(
            "KL training must start from a merged SFT model with a new RL LoRA. "
            "Set paths.lora_adapter_env to null."
        )

    model_path = _required_env(paths["model_env"])
    _required_env("TAU2_USER_LLM")
    _required_env("TAU2_USER_API_BASE")

    task_file = (PROJECT_ROOT / paths["task_file"]).resolve()
    template_path = (PROJECT_ROOT / paths["chat_template"]).resolve()
    agent_loop_path = (PROJECT_ROOT / paths["agent_loop_config"]).resolve()
    output_dir = (PROJECT_ROOT / paths["output_dir"]).resolve()
    trajectory_dir = output_dir / "trajectories"

    resume_checkpoint = None
    if resume:
        resume_checkpoint = _latest_checkpoint(output_dir)
    elif resume_from is not None:
        resume_checkpoint = resume_from.expanduser().resolve()

    if resume_checkpoint is None:
        adapter_env = paths.get("lora_adapter_env")
        adapter_path = Path(_required_env(adapter_env)) if adapter_env else None
        resume_mode = "disable"
    else:
        resume_checkpoint = resume_checkpoint.resolve()
        adapter_path = _resume_adapter_path(resume_checkpoint)
        resume_mode = "resume_path"

    if not task_file.exists():
        raise FileNotFoundError(f"Training dataset is missing: {task_file}")

    os.environ["AGENTIC_RL_CHAT_TEMPLATE"] = template_path.read_text(encoding="utf-8")
    os.environ["AGENTIC_RL_TRAJECTORY_LOG_DIR"] = str(trajectory_dir)

    overrides = {
        "algorithm.adv_estimator": algorithm.get("adv_estimator", "grpo"),
        "algorithm.use_kl_in_reward": use_kl_in_reward,
        "algorithm.norm_adv_by_std_in_grpo": True,
        "data.train_files": task_file,
        "data.val_files": task_file,
        "data.train_batch_size": data.get("train_batch_size", 1),
        "data.val_batch_size": 1,
        "data.dataloader_num_workers": data.get("dataloader_num_workers", 0),
        "data.max_prompt_length": data["max_prompt_length"],
        "data.max_response_length": data["max_response_length"],
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "data.shuffle": data.get("shuffle", False),
        "data.seed": data.get("seed"),
        "actor_rollout_ref.model.path": model_path,
        "actor_rollout_ref.model.custom_chat_template": (
            "${oc.env:AGENTIC_RL_CHAT_TEMPLATE}"
        ),
        "actor_rollout_ref.model.lora_rank": model["lora_rank"],
        "actor_rollout_ref.model.lora_alpha": model["lora_alpha"],
        "actor_rollout_ref.model.target_modules": model["target_modules"],
        "+actor_rollout_ref.model.override_config.attn_implementation": model[
            "attn_implementation"
        ],
        "actor_rollout_ref.model.use_remove_padding": model["use_remove_padding"],
        "actor_rollout_ref.model.use_fused_kernels": model.get(
            "use_fused_kernels", False
        ),
        "actor_rollout_ref.model.fused_kernel_options.impl_backend": model.get(
            "fused_kernel_backend", "torch"
        ),
        "actor_rollout_ref.model.enable_gradient_checkpointing": model.get(
            "enable_gradient_checkpointing", True
        ),
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.ppo_mini_batch_size": actor[
            "ppo_mini_batch_size"
        ],
        "actor_rollout_ref.actor.use_dynamic_bsz": actor.get(
            "use_dynamic_bsz", False
        ),
        "actor_rollout_ref.actor.ppo_epochs": actor["ppo_epochs"],
        "actor_rollout_ref.actor.loss_agg_mode": actor.get(
            "loss_agg_mode", "token-mean"
        ),
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": actor[
            "ppo_max_token_len_per_gpu"
        ],
        "actor_rollout_ref.actor.optim.lr": actor["learning_rate"],
        "actor_rollout_ref.actor.use_kl_loss": use_kl_loss,
        "actor_rollout_ref.actor.kl_loss_coef": actor.get(
            "kl_loss_coef", 0.001
        ),
        "actor_rollout_ref.actor.kl_loss_type": actor.get(
            "kl_loss_type", "low_var_kl"
        ),
        "actor_rollout_ref.actor.use_torch_compile": actor["use_torch_compile"],
        "actor_rollout_ref.actor.checkpoint.save_contents": actor[
            "checkpoint_save_contents"
        ],
        "actor_rollout_ref.actor.checkpoint.load_contents": actor[
            "checkpoint_load_contents"
        ],
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": actor[
            "use_torch_compile"
        ],
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.load_format": "safetensors",
        "actor_rollout_ref.rollout.n": rollout["n"],
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
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": rollout.get(
            "log_prob_use_dynamic_bsz", actor.get("use_dynamic_bsz", False)
        ),
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": rollout.get(
            "log_prob_max_token_len_per_gpu", actor["ppo_max_token_len_per_gpu"]
        ),
        "actor_rollout_ref.rollout.enable_chunked_prefill": True,
        "actor_rollout_ref.rollout.enable_prefix_caching": True,
        "actor_rollout_ref.rollout.calculate_log_probs": rollout.get(
            "calculate_log_probs", False
        ),
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "actor_rollout_ref.rollout.multi_turn.format": "hermes",
        "actor_rollout_ref.rollout.agent.default_agent_loop": "tau2_retail",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": agent_loop_path,
        "actor_rollout_ref.rollout.agent.num_workers": rollout[
            "agent_loop_workers"
        ],
        "reward.reward_model.enable": False,
        "algorithm.rollout_correction.bypass_mode": algorithm.get(
            "bypass_mode", False
        ),
        "trainer.project_name": tracking.get("project", "agentic-rl"),
        "trainer.experiment_name": config["experiment"]["name"],
        "trainer.logger": tracking.get("backends", ["console", "file"]),
        "trainer.n_gpus_per_node": trainer["n_gpus_per_node"],
        "trainer.nnodes": trainer["nnodes"],
        "trainer.default_local_dir": output_dir / "checkpoints",
        "trainer.rollout_data_dir": output_dir / "rollouts",
        "trainer.val_before_train": False,
        "trainer.val_only": False,
        "trainer.resume_mode": resume_mode,
        "trainer.save_freq": trainer["save_freq"],
        "trainer.test_freq": -1,
        "trainer.total_epochs": trainer.get("total_epochs", 1),
        "trainer.total_training_steps": trainer["total_training_steps"],
    }
    if adv_estimator_module := algorithm.get("adv_estimator_module"):
        overrides["+algorithm.adv_estimator_module"] = adv_estimator_module
        overrides["+algorithm.adv_estimator_kwargs"] = algorithm.get(
            "adv_estimator_kwargs", {}
        )
    if not actor.get("use_dynamic_bsz", False):
        overrides["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] = actor[
            "ppo_micro_batch_size_per_gpu"
        ]
    if not rollout.get("log_prob_use_dynamic_bsz", actor.get("use_dynamic_bsz", False)):
        overrides["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] = rollout[
            "log_prob_micro_batch_size_per_gpu"
        ]
    if resume_checkpoint is not None:
        overrides["trainer.resume_from_path"] = resume_checkpoint
    if adapter_path is not None:
        overrides["actor_rollout_ref.model.lora_adapter_path"] = adapter_path

    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        *[_override(name, value) for name, value in overrides.items()],
    ]
    resolved_paths = {
        "model": Path(model_path),
        "chat_template": template_path,
        "agent_loop_config": agent_loop_path,
        "task_file": task_file,
    }
    if adapter_path is not None:
        resolved_paths["adapter"] = adapter_path
    return command, output_dir, trajectory_dir, resolved_paths


def write_group_summaries(
    trajectory_dir: Path,
    run_dir: Path,
    previous_files: set[Path],
) -> list[Path]:
    trajectory_files = sorted(set(trajectory_dir.rglob("*.json")) - previous_files)
    grouped_trajectories: dict[
        int,
        dict[str, list[tuple[Path, dict[str, Any]]]],
    ] = {}
    for path in trajectory_files:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        if "reward" not in trajectory:
            continue
        global_step = int(trajectory["global_step"])
        task_id = str(trajectory["task_id"])
        step_groups = grouped_trajectories.setdefault(global_step, {})
        step_groups.setdefault(task_id, []).append((path, trajectory))

    summary_dir = run_dir / "group_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = []
    for global_step, step_groups in sorted(grouped_trajectories.items()):
        groups = []
        for task_id, group in sorted(step_groups.items(), key=lambda item: int(item[0])):
            rewards = [float(trajectory["reward"]) for _, trajectory in group]
            mean_reward = sum(rewards) / len(rewards)
            if len(rewards) > 1:
                variance = sum((reward - mean_reward) ** 2 for reward in rewards) / (
                    len(rewards) - 1
                )
                reward_std = math.sqrt(variance)
            else:
                reward_std = 0.0

            samples = []
            for (path, trajectory), reward in zip(group, rewards, strict=True):
                samples.append(
                    {
                        "request_id": trajectory["request_id"],
                        "rollout_index": trajectory["rollout_index"],
                        "slot_id": trajectory.get("slot_id"),
                        "trajectory_file": str(path),
                        "reward": reward,
                        "advantage": (reward - mean_reward) / (reward_std + 1.0e-6),
                    }
                )

            groups.append(
                {
                    "task_id": task_id,
                    "group_size": len(samples),
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "mixed_reward_group": len(set(rewards)) > 1,
                    "samples": samples,
                }
            )

        mixed_group_count = sum(group["mixed_reward_group"] for group in groups)
        summary = {
            "global_step": global_step,
            "num_groups": len(groups),
            "total_trajectories": sum(group["group_size"] for group in groups),
            "mixed_group_count": mixed_group_count,
            "mixed_group_ratio": mixed_group_count / len(groups),
            "groups": groups,
        }
        summary_path = summary_dir / f"step_{global_step:06d}.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_paths.append(summary_path)
    return summary_paths


def _run_manifest(
    *,
    run_id: str,
    config_path: Path,
    config: dict[str, Any],
    command: list[str],
    resolved_paths: dict[str, Path],
) -> dict[str, Any]:
    files = [
        config_path,
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/agentic_rl/rollout/tau2_agent_loop.py",
        resolved_paths["chat_template"],
        resolved_paths["agent_loop_config"],
        resolved_paths["task_file"],
        resolved_paths["model"] / "config.json",
        resolved_paths["model"] / "model.safetensors.index.json",
        resolved_paths["model"] / "merge_manifest.json",
        resolved_paths["model"] / "tokenizer_config.json",
        PROJECT_ROOT.parent / "verl/verl/experimental/agent_loop/agent_loop.py",
        PROJECT_ROOT.parent / "verl/verl/models/transformers/dense_common.py",
        PROJECT_ROOT.parent / "verl/verl/trainer/config/algorithm.py",
        PROJECT_ROOT.parent / "verl/verl/trainer/ppo/ray_trainer.py",
        PROJECT_ROOT.parent / "verl/verl/trainer/ppo/metric_utils.py",
        PROJECT_ROOT.parent / "verl/verl/utils/tracking.py",
        PROJECT_ROOT.parent / "verl/verl/utils/experimental/torch_functional.py",
        PROJECT_ROOT.parent / "verl/verl/workers/actor/dp_actor.py",
        PROJECT_ROOT.parent / "verl/verl/workers/fsdp_workers.py",
    ]
    if module_name := config.get("algorithm", {}).get("adv_estimator_module"):
        module_path = PROJECT_ROOT / "src" / Path(*module_name.split("."))
        files.append(module_path.with_suffix(".py"))
        files.append(PROJECT_ROOT / "src/agentic_rl/envs/tau2_retail/progress.py")
    if adapter_path := resolved_paths.get("adapter"):
        files.extend(
            [
                adapter_path / "adapter_config.json",
                adapter_path / "adapter_model.safetensors",
            ]
        )
    return {
        "run_id": run_id,
        "status": "running",
        "started_at": _utc_now(),
        "command": command,
        "config_path": str(config_path),
        "config": config,
        "resolved_paths": {key: str(value) for key, value in resolved_paths.items()},
        "files": [record for path in files if (record := _file_record(path))],
        "packages": _package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "user_simulator": {
            "model": os.environ["TAU2_USER_LLM"],
            "api_base": os.environ["TAU2_USER_API_BASE"],
        },
        "git": {
            "agentic_grpo": _git_state(PROJECT_ROOT),
            "verl": _git_state(PROJECT_ROOT.parent / "verl"),
            "tau2_bench": _git_state(PROJECT_ROOT.parent / "tau2-bench"),
        },
        "tracking": {
            "backends": config.get("tracking", {}).get(
                "backends", ["console", "file"]
            ),
            "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
            "wandb_group": os.environ.get("WANDB_RUN_GROUP"),
            "wandb_tags": os.environ.get("WANDB_TAGS"),
            "wandb_mode": os.environ.get("WANDB_MODE"),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_with_log(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the configured output directory",
    )
    resume_group.add_argument(
        "--resume-from",
        type=Path,
        help="Resume from a specific global_step_N checkpoint directory",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = os.environ.setdefault("AGENTIC_RL_RUN_ID", _new_run_id())
    command, output_dir, trajectory_dir, resolved_paths = build_command(
        config,
        resume=args.resume,
        resume_from=args.resume_from,
    )
    if args.dry_run:
        print("\n".join(command))
        return

    tracking_state = _configure_tracking(
        config,
        output_dir,
        resume=args.resume or args.resume_from is not None,
    )

    run_dir = output_dir / "runs" / run_id
    metrics_path = run_dir / "metrics.jsonl"
    train_log_path = run_dir / "train.log"
    manifest_path = run_dir / "manifest.json"
    os.environ["VERL_FILE_LOGGER_PATH"] = str(metrics_path)
    os.environ["AGENTIC_RL_RUN_MANIFEST_PATH"] = str(manifest_path)

    previous_files = set(trajectory_dir.rglob("*.json"))
    manifest = _run_manifest(
        run_id=run_id,
        config_path=config_path,
        config=config,
        command=command,
        resolved_paths=resolved_paths,
    )
    manifest["tracking"] = tracking_state
    _write_json(manifest_path, manifest)

    started = time.perf_counter()
    summary_paths: list[Path] = []
    try:
        _run_with_log(command, train_log_path)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        manifest["error_message"] = str(error)
        raise
    else:
        manifest["status"] = "completed"
        print(f"GRPO run artifacts: {run_dir}")
    finally:
        summary_paths = write_group_summaries(
            trajectory_dir,
            run_dir,
            previous_files,
        )
        manifest["group_summaries"] = [str(path) for path in summary_paths]
        manifest["finished_at"] = _utc_now()
        manifest["elapsed_seconds"] = time.perf_counter() - started
        _write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
