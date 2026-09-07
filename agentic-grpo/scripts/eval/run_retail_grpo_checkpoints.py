#!/usr/bin/env python3
"""Evaluate the SFT policy and GRPO LoRA checkpoints on tau2 Retail.

One vLLM server hosts the merged SFT model and all requested GRPO adapters.
Each policy writes to its own resumable tau2 result file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYZER = PROJECT_ROOT / "scripts/eval/analyze_passk_group_rewards.py"


@dataclass(frozen=True)
class PolicySpec:
    label: str
    served_name: str
    adapter_path: Path | None

    @property
    def litellm_name(self) -> str:
        return f"openai/{self.served_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SFT and GRPO checkpoints with one multi-LoRA vLLM server."
    )
    parser.add_argument("config", type=Path, help="GRPO evaluation YAML")
    parser.add_argument(
        "--steps",
        nargs="*",
        type=int,
        help="Override models.steps from the YAML, for example: --steps 80 100",
    )
    parser.add_argument(
        "--include-sft",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override models.include_sft from the YAML",
    )
    parser.add_argument(
        "--vllm-bin",
        type=Path,
        help="vLLM executable; defaults to the active virtual environment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the plan without starting vLLM",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        value = yaml.safe_load(config_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def find_vllm_binary(override: Path | None) -> Path:
    candidates = [
        override,
        Path(sys.executable).with_name("vllm"),
        Path(shutil.which("vllm")) if shutil.which("vllm") else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "vLLM executable not found. Activate the server .venv or pass --vllm-bin."
    )


def policy_specs(
    config: dict[str, Any],
    *,
    steps_override: list[int] | None,
    include_sft_override: bool | None,
) -> tuple[Path, list[PolicySpec]]:
    paths = config["paths"]
    models = config["models"]
    checkpoint_root = project_path(paths["checkpoint_root"])
    include_sft = (
        bool(models["include_sft"])
        if include_sft_override is None
        else include_sft_override
    )
    raw_steps = models.get("steps", []) if steps_override is None else steps_override
    steps = sorted(set(int(step) for step in raw_steps))

    specs: list[PolicySpec] = []
    if include_sft:
        specs.append(
            PolicySpec(
                label="sft",
                served_name=str(models["sft_served_name"]),
                adapter_path=None,
            )
        )

    prefix = str(models["grpo_served_name_prefix"])
    for step in steps:
        if step <= 0:
            raise ValueError(f"Checkpoint steps must be positive: {step}")
        adapter_path = (
            checkpoint_root
            / f"global_step_{step}"
            / "actor"
            / "lora_adapter"
        )
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter_path / filename).is_file():
                raise FileNotFoundError(
                    f"Checkpoint step {step} is missing {filename}: {adapter_path}"
                )
        specs.append(
            PolicySpec(
                label=f"step_{step:03d}",
                served_name=f"{prefix}{step:03d}",
                adapter_path=adapter_path,
            )
        )

    if not specs:
        raise ValueError("No policies selected: enable SFT or provide checkpoint steps")
    return checkpoint_root, specs


def task_selection(config: dict[str, Any]) -> tuple[str, list[str]]:
    selection = config["task_selection"]
    split_config = load_mapping(project_path(selection["split_file"]))
    split = split_config["splits"][selection["split_name"]]
    count = int(selection["num_tasks"])
    task_ids = [str(task_id) for task_id in split["task_ids"][:count]]
    if len(task_ids) != count:
        raise ValueError(
            f"Requested {count} tasks from {selection['split_name']}, found {len(task_ids)}"
        )
    return str(split["task_split_name"]), task_ids


def server_endpoint(server: dict[str, Any]) -> tuple[str, int, str]:
    parsed = urlparse(str(server["api_base"]))
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("server.api_base must be an http URL")
    if parsed.path.rstrip("/") != "/v1":
        raise ValueError("server.api_base must end with /v1")
    return parsed.hostname, parsed.port or 8000, str(server["api_key"])


def build_vllm_command(
    *,
    vllm_bin: Path,
    base_model: Path,
    chat_template: Path,
    server: dict[str, Any],
    specs: list[PolicySpec],
) -> list[str]:
    host, port, api_key = server_endpoint(server)
    command = [
        str(vllm_bin),
        "serve",
        str(base_model),
        "--served-model-name",
        str(server["base_served_name"]),
        "--host",
        host,
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--tensor-parallel-size",
        str(server["tensor_parallel_size"]),
        "--max-model-len",
        str(server["max_model_len"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--dtype",
        str(server.get("dtype", "auto")),
        "--generation-config",
        "vllm",
        "--chat-template",
        str(chat_template),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        str(server.get("tool_call_parser", "hermes")),
        "--enable-prefix-caching",
    ]

    adapters = [spec for spec in specs if spec.adapter_path is not None]
    if adapters:
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(server.get("max_lora_rank", 16)),
                "--max-loras",
                "1",
                "--max-cpu-loras",
                str(len(adapters)),
                "--lora-modules",
            ]
        )
        command.extend(
            f"{spec.served_name}={spec.adapter_path}" for spec in adapters
        )
    return command


def no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def wait_for_server(
    process: subprocess.Popen[bytes],
    *,
    api_base: str,
    api_key: str,
    expected_models: set[str],
    timeout_seconds: float,
    log_path: Path,
) -> None:
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    deadline = time.monotonic() + timeout_seconds
    opener = no_proxy_opener()

    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
            raise RuntimeError(f"vLLM exited during startup:\n{tail}")
        try:
            with opener.open(request, timeout=2) as response:
                payload = json.load(response)
            available = {item["id"] for item in payload.get("data", [])}
            if expected_models <= available:
                print("vLLM ready: " + ", ".join(sorted(available)))
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready within {timeout_seconds:.0f}s")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(config))
    if isinstance(clean.get("server"), dict):
        clean["server"].pop("api_key", None)
    return clean


def evaluation_manifest(
    *,
    config_path: Path,
    config: dict[str, Any],
    base_model: Path,
    chat_template: Path,
    task_ids: list[str],
    spec: PolicySpec,
    user_model: str,
    user_api_base: str,
) -> dict[str, Any]:
    artifacts: dict[str, dict[str, str]] = {}
    candidates = [
        base_model / "merge_manifest.json",
        base_model / "config.json",
        chat_template,
    ]
    if spec.adapter_path is not None:
        candidates.extend(
            [
                spec.adapter_path / "adapter_config.json",
                spec.adapter_path / "adapter_model.safetensors",
            ]
        )
    for path in candidates:
        if path.is_file():
            artifacts[str(path)] = {"sha256": sha256_file(path)}

    payload = {
        "format": "agentic_rl_retail_grpo_evaluation",
        "version": 1,
        "policy": {
            "label": spec.label,
            "served_name": spec.served_name,
            "base_model": str(base_model),
            "adapter_path": str(spec.adapter_path) if spec.adapter_path else None,
        },
        "config_path": str(config_path),
        "config": public_config(config),
        "task_ids": task_ids,
        "user": {
            "model": user_model,
            "api_base": user_api_base,
        },
        "versions": {
            "vllm": package_version("vllm"),
            "tau2": package_version("tau2"),
            "transformers": package_version("transformers"),
        },
        "artifacts": artifacts,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def ensure_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != manifest["fingerprint"]:
            raise RuntimeError(
                f"Evaluation settings changed for existing run: {path.parent}. "
                "Use a new output directory instead of mixing results."
            )
        return
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def evaluate_policy(
    *,
    config_path: Path,
    config: dict[str, Any],
    base_model: Path,
    chat_template: Path,
    output_root: Path,
    task_split_name: str,
    task_ids: list[str],
    spec: PolicySpec,
) -> Path:
    from agentic_rl.data.collection import register_model_pricing
    from tau2 import TextRunConfig
    from tau2.data_model.simulation import Results, RewardInfo, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner import get_tasks, run_tasks

    server = config["server"]
    agent = config["agent"]
    user = config["user"]
    sampling = config["sampling"]

    model_dir = output_root / spec.label
    model_dir.mkdir(parents=True, exist_ok=True)
    results_path = model_dir / f"results_{spec.label}.json"

    def classify_context_overflows(results: Results) -> int:
        count = 0
        for simulation in results.simulations:
            if simulation.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR:
                continue

            info = simulation.info or {}
            diagnostic = " ".join(
                str(info.get(field, ""))
                for field in ("error_type", "error", "error_traceback")
            )
            if "ContextWindowExceededError" not in diagnostic:
                continue

            simulation.termination_reason = (
                TerminationReason.CONTEXT_WINDOW_EXCEEDED
            )
            simulation.reward_info = RewardInfo(reward=0.0)
            simulation.info = {
                **info,
                "original_termination_reason": (
                    TerminationReason.INFRASTRUCTURE_ERROR.value
                ),
                "failure_classification": "model_context_window_exceeded",
            }
            count += 1
        return count

    saved_context_overflows = 0
    if results_path.is_file():
        previous_results = Results.load(results_path)
        saved_context_overflows = classify_context_overflows(previous_results)
        if saved_context_overflows:
            previous_results.save(results_path)
            print(
                f"{spec.label}: reclassified {saved_context_overflows} saved context "
                "overflow(s) as zero-reward model failures"
            )

    user_model = required_env(user["model_env"])
    user_api_base = required_env(user["api_base_env"])
    user_api_key = required_env(user["api_key_env"])
    manifest = evaluation_manifest(
        config_path=config_path,
        config=config,
        base_model=base_model,
        chat_template=chat_template,
        task_ids=task_ids,
        spec=spec,
        user_model=user_model,
        user_api_base=user_api_base,
    )
    ensure_manifest(model_dir / "run_manifest.json", manifest)

    register_model_pricing(spec.litellm_name)
    register_model_pricing(user_model)

    agent_args = {
        "api_base": server["api_base"],
        "api_key": server["api_key"],
        **agent,
    }
    user_args = {
        "api_base": user_api_base,
        "api_key": user_api_key,
        **user.get("args", {}),
    }

    run_config = TextRunConfig(
        domain=config["domain"],
        task_split_name=task_split_name,
        task_ids=task_ids,
        llm_agent=spec.litellm_name,
        llm_user=user_model,
        llm_args_agent=agent_args,
        llm_args_user=user_args,
        num_trials=int(sampling["num_trials"]),
        max_concurrency=int(sampling["max_concurrency"]),
        seed=int(sampling["seed"]),
        max_steps=int(sampling["max_steps"]),
        max_errors=int(sampling["max_errors"]),
        max_retries=int(sampling["max_retries"]),
        retry_delay=float(sampling["retry_delay"]),
        auto_resume=True,
        save_to=str(results_path),
    )
    tasks = get_tasks(
        task_set_name=config["domain"],
        task_split_name=task_split_name,
        task_ids=task_ids,
    )

    print(f"\n=== Evaluating {spec.label}: {spec.litellm_name} ===")
    results = run_tasks(
        run_config,
        tasks,
        save_path=results_path,
        save_dir=model_dir,
        evaluation_type=EvaluationType(config["evaluation_type"]),
    )
    new_context_overflows = classify_context_overflows(results)
    if new_context_overflows:
        results.save(results_path)
    context_overflows = saved_context_overflows + new_context_overflows

    infrastructure_errors = [
        simulation
        for simulation in results.simulations
        if simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    ]
    successes = sum(
        simulation.reward_info is not None
        and simulation.reward_info.reward >= 1 - 1e-6
        for simulation in results.simulations
    )
    print(
        f"{spec.label}: successes={successes}/{len(results.simulations)}, "
        f"context_overflows={context_overflows}, "
        f"infrastructure_errors={len(infrastructure_errors)}"
    )
    if infrastructure_errors:
        raise RuntimeError(
            f"{spec.label} has {len(infrastructure_errors)} infrastructure error(s). "
            "Rerun the same command after the API is healthy; completed trials will be skipped."
        )
    return results_path


def run_analysis(config: dict[str, Any], result_paths: list[Path], output_root: Path) -> None:
    trials = int(config["sampling"]["num_trials"])
    command = [
        sys.executable,
        str(ANALYZER),
        *(str(path) for path in result_paths),
        "--output-dir",
        str(output_root / "analysis"),
        "--expected-trials",
        str(trials),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    config_path = args.config.expanduser().resolve()
    config = load_mapping(config_path)
    base_model = Path(required_env(config["paths"]["base_model_env"])).resolve()
    chat_template = project_path(config["paths"]["chat_template"])
    output_root = project_path(config["paths"]["output_dir"])
    if not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"Merged SFT model is incomplete: {base_model}")
    if not chat_template.is_file():
        raise FileNotFoundError(f"Chat template not found: {chat_template}")

    _, specs = policy_specs(
        config,
        steps_override=args.steps,
        include_sft_override=args.include_sft,
    )
    task_split_name, task_ids = task_selection(config)
    vllm_bin = find_vllm_binary(args.vllm_bin)
    command = build_vllm_command(
        vllm_bin=vllm_bin,
        base_model=base_model,
        chat_template=chat_template,
        server=config["server"],
        specs=specs,
    )

    print(f"Config: {config_path}")
    print(f"Tasks: {len(task_ids)} ({', '.join(task_ids)})")
    print("Policies: " + ", ".join(spec.label for spec in specs))
    print(f"Output: {output_root}")
    print("vLLM command:")
    print(shlex.join(command))
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "vllm_server.log"
    environment = os.environ.copy()
    host, _, api_key = server_endpoint(config["server"])
    no_proxy = {"127.0.0.1", "localhost", host}
    current_no_proxy = environment.get("NO_PROXY", environment.get("no_proxy", ""))
    no_proxy.update(item for item in current_no_proxy.split(",") if item)
    environment["NO_PROXY"] = ",".join(sorted(no_proxy))
    environment["no_proxy"] = environment["NO_PROXY"]

    with log_path.open("ab") as server_log:
        server_log.write(f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
        server_log.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            expected_models = {str(config["server"]["base_served_name"])} | {
                spec.served_name for spec in specs if spec.adapter_path is not None
            }
            wait_for_server(
                process,
                api_base=str(config["server"]["api_base"]),
                api_key=api_key,
                expected_models=expected_models,
                timeout_seconds=float(config["server"]["startup_timeout_seconds"]),
                log_path=log_path,
            )
            result_paths = [
                evaluate_policy(
                    config_path=config_path,
                    config=config,
                    base_model=base_model,
                    chat_template=chat_template,
                    output_root=output_root,
                    task_split_name=task_split_name,
                    task_ids=task_ids,
                    spec=spec,
                )
                for spec in specs
            ]
            run_analysis(config, result_paths, output_root)
        finally:
            stop_server(process)


if __name__ == "__main__":
    main()
