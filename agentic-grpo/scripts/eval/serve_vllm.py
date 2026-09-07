#!/usr/bin/env python3
"""Launch a local vLLM server from an evaluation YAML configuration."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path
from urllib.parse import urlparse

import yaml


AGENTIC_RL_DIR = Path(__file__).resolve().parents[3]
DEFAULT_VLLM_BIN = AGENTIC_RL_DIR / ".venv-vllm/bin/vllm"
DEFAULT_HF_HOME = AGENTIC_RL_DIR.parent / "hf-cache"
DEFAULT_MODELSCOPE_CACHE = AGENTIC_RL_DIR.parent / "modelscope-cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch vLLM using the server section of an evaluation YAML file."
    )
    parser.add_argument("config", type=Path, help="Path to an evaluation YAML file")
    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="Value assigned to CUDA_VISIBLE_DEVICES (default: 0)",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help=f"Hugging Face cache directory (default: {DEFAULT_HF_HOME})",
    )
    parser.add_argument(
        "--modelscope-cache",
        type=Path,
        default=DEFAULT_MODELSCOPE_CACHE,
        help=f"ModelScope cache directory (default: {DEFAULT_MODELSCOPE_CACHE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the environment and command without starting vLLM",
    )
    return parser.parse_args()


def load_server_config(config_path: Path) -> dict[str, object]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or not isinstance(config.get("server"), dict):
        raise ValueError(f"Missing server mapping in {config_path}")

    server = config["server"]
    required = (
        "model_path",
        "api_base",
        "api_key",
        "tensor_parallel_size",
        "max_model_len",
        "gpu_memory_utilization",
    )
    missing = [key for key in required if key not in server]
    if missing:
        raise ValueError(f"Missing server keys in {config_path}: {', '.join(missing)}")
    return server


def build_command(server: dict[str, object]) -> list[str]:
    api_base = urlparse(str(server["api_base"]))
    if api_base.scheme != "http" or not api_base.hostname:
        raise ValueError("server.api_base must be an http URL with a hostname")
    if api_base.path.rstrip("/") != "/v1":
        raise ValueError("server.api_base must end with /v1")

    model_path = str(server["model_path"])
    served_model_name = str(server.get("served_model_name", model_path))

    command = [
        str(DEFAULT_VLLM_BIN),
        "serve",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        api_base.hostname,
        "--port",
        str(api_base.port or 8000),
        "--api-key",
        str(server["api_key"]),
        "--tensor-parallel-size",
        str(server["tensor_parallel_size"]),
        "--max-model-len",
        str(server["max_model_len"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--dtype",
        str(server.get("dtype", "auto")),
        "--generation-config",
        str(server.get("generation_config", "vllm")),
    ]

    tool_call_parser = server.get("tool_call_parser")
    if tool_call_parser:
        command.extend(
            [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                str(tool_call_parser),
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")
    if not DEFAULT_VLLM_BIN.is_file():
        raise FileNotFoundError(f"vLLM executable not found: {DEFAULT_VLLM_BIN}")

    server = load_server_config(config_path)
    command = build_command(server)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    environment["HF_HOME"] = str(args.hf_home.resolve())
    environment["VLLM_USE_MODELSCOPE"] = "true"
    environment["MODELSCOPE_CACHE"] = str(args.modelscope_cache.resolve())

    print(f"Config: {config_path}")
    print(f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']}")
    print(f"HF_HOME={environment['HF_HOME']}")
    print(f"VLLM_USE_MODELSCOPE={environment['VLLM_USE_MODELSCOPE']}")
    print(f"MODELSCOPE_CACHE={environment['MODELSCOPE_CACHE']}")
    print(shlex.join(command))

    if args.dry_run:
        return

    args.hf_home.mkdir(parents=True, exist_ok=True)
    args.modelscope_cache.mkdir(parents=True, exist_ok=True)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
