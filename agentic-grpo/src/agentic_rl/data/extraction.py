import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

TIER_ORDER = {"low": 0, "medium": 1, "high": 2}


def load_sources(config_path: Path, simulation_dir: Path) -> tuple:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    simulations = []
    tasks = {}
    source_runs = []

    for tier in config["sampling"]["tiers"]:
        run_name = f"{config['sampling']['run_name_prefix']}_{tier['name']}"
        result_path = simulation_dir / run_name / "results.json"
        results = json.loads(result_path.read_text(encoding="utf-8"))
        source_runs.append(run_name)
        tasks.update({task["id"]: task for task in results["tasks"]})
        for simulation in results["simulations"]:
            simulations.append(
                {
                    "tier": tier["name"],
                    "temperature": tier["temperature"],
                    "source_run": run_name,
                    "simulation": simulation,
                }
            )

    simulations.sort(
        key=lambda item: (
            int(item["simulation"]["task_id"]),
            TIER_ORDER[item["tier"]],
            item["simulation"]["trial"],
        )
    )
    return config, simulations, tasks, source_runs


def simplify_message(message: dict) -> dict:
    tool_calls = message.get("tool_calls") or []
    content = message.get("content")
    if message["role"] == "assistant" and tool_calls:
        content = None

    simplified = {
        "role": message["role"],
        "content": content,
        "turn_idx": message.get("turn_idx"),
    }
    if tool_calls:
        simplified["tool_calls"] = [
            {
                "id": call["id"],
                "name": call["name"],
                "arguments": call["arguments"],
            }
            for call in tool_calls
        ]
    if message["role"] == "tool":
        simplified["tool_call_id"] = message["id"]
        simplified["error"] = message.get("error", False)
    return simplified


def build_stats(messages: list[dict]) -> dict:
    role_counts = Counter(message["role"] for message in messages)
    tool_calls = sum(len(message.get("tool_calls") or []) for message in messages)
    tool_errors = sum(
        message["role"] == "tool" and message.get("error", False)
        for message in messages
    )
    mixed_tool_content = sum(
        message["role"] == "assistant"
        and bool(message.get("tool_calls"))
        and bool((message.get("content") or "").strip())
        for message in messages
    )
    token_totals = {
        "agent_prompt_tokens": 0,
        "agent_completion_tokens": 0,
        "user_prompt_tokens": 0,
        "user_completion_tokens": 0,
    }
    for message in messages:
        usage = message.get("usage") or {}
        prefix = {"assistant": "agent", "user": "user"}.get(message["role"])
        if not prefix or not usage:
            continue
        token_totals[f"{prefix}_prompt_tokens"] += usage.get("prompt_tokens") or 0
        token_totals[f"{prefix}_completion_tokens"] += (
            usage.get("completion_tokens") or 0
        )

    return {
        "message_count": len(messages),
        "assistant_turns": role_counts["assistant"],
        "user_turns": role_counts["user"],
        "tool_call_count": tool_calls,
        "tool_error_count": tool_errors,
        "mixed_tool_content_count": mixed_tool_content,
        **token_totals,
    }


def simplify_task(task: dict) -> tuple[dict, dict]:
    instructions = task["user_scenario"]["instructions"]
    criteria = task["evaluation_criteria"]
    task_spec = {
        "reason_for_call": instructions.get("reason_for_call"),
        "known_info": instructions.get("known_info"),
        "unknown_info": instructions.get("unknown_info"),
        "task_instructions": instructions.get("task_instructions"),
    }
    audit_reference = {
        "expected_actions": criteria.get("actions") or [],
        "reward_basis": criteria.get("reward_basis") or [],
        "env_assertions": criteria.get("env_assertions") or [],
        "nl_assertions": criteria.get("nl_assertions") or [],
        "communicate_info": criteria.get("communicate_info") or [],
    }
    return task_spec, audit_reference


def build_candidate(item: dict, task: dict) -> dict:
    simulation = item["simulation"]
    reward_info = simulation["reward_info"]
    stats = build_stats(simulation["messages"])
    task_spec, audit_reference = simplify_task(task)
    termination = simulation["termination_reason"]
    rejection_reasons = []
    if reward_info["reward"] != 1.0:
        rejection_reasons.append("reward_zero")
    if termination not in {"user_stop", "agent_stop"}:
        rejection_reasons.append(termination)
    if stats["tool_error_count"]:
        rejection_reasons.append("tool_error")

    db_check = reward_info.get("db_check") or {}
    return {
        "trajectory_id": simulation["id"],
        "task_id": simulation["task_id"],
        "tier": item["tier"],
        "temperature": item["temperature"],
        "trial": simulation["trial"],
        "seed": simulation["seed"],
        "source_run": item["source_run"],
        "task_spec": task_spec,
        "audit_reference": audit_reference,
        "outcome": {
            "reward": reward_info["reward"],
            "reward_breakdown": reward_info.get("reward_breakdown") or {},
            "db_match": db_check.get("db_match"),
            "termination_reason": termination,
            "duration_seconds": simulation["duration"],
        },
        "messages": [simplify_message(message) for message in simulation["messages"]],
        "stats": stats,
        "quality": {
            "reward_pass": reward_info["reward"] == 1.0,
            "normal_termination": termination in {"user_stop", "agent_stop"},
            "no_tool_errors": stats["tool_error_count"] == 0,
            "within_length_limit": termination != "max_steps",
            "user_fidelity": None,
            "process_valid": None,
            "diversity_score": None,
            "decision": "pending",
            "rejection_reasons": rejection_reasons,
        },
    }


def build_manifest(
    config: dict,
    candidates: list[dict],
    source_runs: list[str],
    source_config: str,
) -> dict:
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
    from tau2.registry import registry

    environment = registry.get_env_constructor(config["domain"])()
    policy = environment.policy
    tools = [tool.openai_schema for tool in environment.tools.get_tools().values()]
    rewards = Counter(candidate["outcome"]["reward"] for candidate in candidates)
    terminations = Counter(
        candidate["outcome"]["termination_reason"] for candidate in candidates
    )
    return {
        "dataset_name": f"{config['experiment_name']}_candidates",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": config["domain"],
        "task_split": config["task_split"],
        "task_ids": config["task_ids"],
        "source_config": source_config,
        "source_runs": source_runs,
        "provider": config["provider"]["name"],
        "agent_model": config["agent"]["model"],
        "user_model": config["user"]["model"],
        "sampling_tiers": config["sampling"]["tiers"],
        "evaluation_type": config["evaluation_type"],
        "assistant_tool_content_normalization": "set_content_to_null",
        "agent_system_prompt": SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION,
            domain_policy=policy,
        ),
        "retail_policy": policy,
        "tool_schemas": tools,
        "total_trajectories": len(candidates),
        "reward_distribution": {str(key): value for key, value in rewards.items()},
        "termination_distribution": dict(terminations),
    }


def extract_candidates(
    config_path: Path,
    simulation_dir: Path,
    output_dir: Path,
    source_config: str,
) -> int:
    config, simulations, tasks, source_runs = load_sources(
        config_path, simulation_dir
    )
    candidates = [
        build_candidate(item, tasks[item["simulation"]["task_id"]])
        for item in simulations
    ]
    manifest = build_manifest(config, candidates, source_runs, source_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "candidates.jsonl").open("w", encoding="utf-8") as file:
        for candidate in candidates:
            file.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    return len(candidates)
