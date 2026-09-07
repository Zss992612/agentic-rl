#!/usr/bin/env python3
"""Compare SFT, Vanilla GRPO, and credit-assignment Retail evaluations.

The script aligns trajectories by ``(task_id, trial)`` and produces two kinds
of evidence:

1. Outcome transitions: which trials and tasks improved or regressed.
2. Behavioural diagnostics: tool use, observed tool errors, exact retries,
   reference-path differences, protocol issues, length failures, and token cost.

The diagnostics are review candidates, not automatic causal labels. In
particular, a read tool returning an error can be a reasonable negative lookup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFRASTRUCTURE_ERROR = "infrastructure_error"
AMBIGUOUS_EXTERNAL_TERMINATIONS = {
    INFRASTRUCTURE_ERROR,
    "timeout",
    "user_error",
    "unexpected_error",
}
MODEL_LABELS = ("sft", "vanilla", "potential")
COMPARISONS = (
    ("sft", "vanilla"),
    ("sft", "potential"),
    ("vanilla", "potential"),
)
PRIMARY_COMPARISON = ("vanilla", "potential")

READ_PREFIXES = ("find_", "get_", "list_", "search_")
WRITE_PREFIXES = ("cancel_", "modify_", "return_", "exchange_", "place_", "create_")
ABNORMAL_TERMINATIONS = {
    "context_window_exceeded",
    "max_steps",
    "too_many_errors",
    "agent_error",
}

TrajectoryKey = tuple[str, int]


@dataclass
class ResultSet:
    label: str
    path: Path
    model: str
    info: dict[str, Any]
    tasks: dict[str, dict[str, Any]]
    simulations: dict[TrajectoryKey, dict[str, Any]]


@dataclass
class ToolCall:
    name: str
    arguments: Any
    call_id: str | None
    turn_idx: int | None
    message_index: int
    family: str
    error: bool = False
    error_text: str = ""
    response_message_index: int | None = None
    recovery: str = "not_applicable"

    @property
    def signature(self) -> str:
        return f"{self.name}:{canonical_json(self.arguments)}"


@dataclass
class TrajectoryFeatures:
    key: TrajectoryKey
    reward: float
    success: bool
    termination_reason: str
    duration_seconds: float
    assistant_turns: int
    completion_tokens: int
    max_prompt_tokens: int
    calls: list[ToolCall]
    mixed_reply_turns: int
    exact_repeated_calls: int
    consecutive_exact_retries: int
    unobserved_reference_writes: list[str]
    nonreference_writes: list[str]
    diagnostic_flags: list[str]

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    def family_count(self, family: str) -> int:
        return sum(call.family == family for call in self.calls)

    @property
    def tool_errors(self) -> int:
        return sum(call.error for call in self.calls)

    @property
    def read_errors(self) -> int:
        return sum(call.error and call.family == "read" for call in self.calls)

    @property
    def write_errors(self) -> int:
        return sum(call.error and call.family == "write" for call in self.calls)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align three tau2 Retail evaluations and identify regressions, "
            "improvements, and tool-use error patterns."
        )
    )
    parser.add_argument("--sft", type=Path, required=True, help="SFT results JSON")
    parser.add_argument(
        "--vanilla", type=Path, required=True, help="Selected Vanilla GRPO results JSON"
    )
    parser.add_argument(
        "--potential",
        type=Path,
        required=True,
        help="Selected credit-assignment GRPO results JSON",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-trials", type=int, default=4)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def task_sort_key(task_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(task_id))
    except ValueError:
        return (1, task_id)


def key_sort_key(key: TrajectoryKey) -> tuple[tuple[int, int | str], int]:
    return task_sort_key(key[0]), key[1]


def is_successful(reward: float) -> bool:
    return math.isclose(reward, 1.0, abs_tol=1e-6)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_result_set(
    label: str, path: Path, expected_trials: int
) -> tuple[ResultSet, dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    data = read_json(path)
    info = data.get("info")
    raw_tasks = data.get("tasks")
    raw_simulations = data.get("simulations")
    if not isinstance(info, dict):
        raise ValueError(f"{path}: info must be an object")
    if not isinstance(raw_tasks, list):
        raise ValueError(f"{path}: tasks must be an array")
    if not isinstance(raw_simulations, list):
        raise ValueError(f"{path}: simulations must be an array")
    if info.get("num_trials") != expected_trials:
        raise ValueError(
            f"{path}: info.num_trials={info.get('num_trials')!r}, "
            f"expected {expected_trials}"
        )

    tasks: dict[str, dict[str, Any]] = {}
    for task in raw_tasks:
        if not isinstance(task, dict) or task.get("id") is None:
            raise ValueError(f"{path}: every task must have an id")
        task_id = str(task["id"])
        if task_id in tasks:
            raise ValueError(f"{path}: duplicate task id {task_id}")
        tasks[task_id] = task

    simulations: dict[TrajectoryKey, dict[str, Any]] = {}
    invalid_termination_keys: list[dict[str, Any]] = []
    duplicate_keys: list[TrajectoryKey] = []
    for simulation in raw_simulations:
        if not isinstance(simulation, dict):
            raise ValueError(f"{path}: every simulation must be an object")
        task_id = str(simulation.get("task_id"))
        trial = simulation.get("trial")
        if task_id == "None" or not isinstance(trial, int):
            raise ValueError(f"{path}: simulation is missing task_id or integer trial")
        key = task_id, trial
        if key in simulations:
            duplicate_keys.append(key)
        simulations[key] = simulation
        termination_reason = str(simulation.get("termination_reason"))
        if termination_reason in AMBIGUOUS_EXTERNAL_TERMINATIONS:
            invalid_termination_keys.append(
                {"task_id": task_id, "trial": trial, "reason": termination_reason}
            )

    expected_trial_ids = set(range(expected_trials))
    trials_by_task: dict[str, set[int]] = defaultdict(set)
    for task_id, trial in simulations:
        trials_by_task[task_id].add(trial)
    incomplete_tasks = {
        task_id: sorted(trials)
        for task_id, trials in trials_by_task.items()
        if trials != expected_trial_ids
    }
    tasks_without_simulations = sorted(
        set(tasks) - set(trials_by_task), key=task_sort_key
    )

    agent_info = info.get("agent_info") or {}
    model = str(agent_info.get("llm", label)) if isinstance(agent_info, dict) else label
    audit = {
        "label": label,
        "path": str(path),
        "model": model,
        "num_tasks": len(tasks),
        "num_simulations": len(raw_simulations),
        "num_unique_keys": len(simulations),
        "duplicate_keys": [list(key) for key in sorted(duplicate_keys, key=key_sort_key)],
        "invalid_termination_keys": invalid_termination_keys,
        "incomplete_tasks": incomplete_tasks,
        "tasks_without_simulations": tasks_without_simulations,
    }
    problems = (
        duplicate_keys
        or invalid_termination_keys
        or incomplete_tasks
        or tasks_without_simulations
        or set(tasks) != set(trials_by_task)
    )
    if problems:
        raise ValueError(
            f"{path}: evaluation is incomplete or invalid "
            f"(duplicates={len(duplicate_keys)}, "
            f"external_or_ambiguous_terminations={len(invalid_termination_keys)}, "
            f"incomplete_tasks={len(incomplete_tasks)}, "
            f"tasks_without_simulations={len(tasks_without_simulations)})"
        )

    return ResultSet(label, path, model, info, tasks, simulations), audit


def comparable_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_scenario": task.get("user_scenario"),
        "initial_state": task.get("initial_state"),
        "evaluation_criteria": task.get("evaluation_criteria"),
    }


def nested_value(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def validate_alignment(
    result_sets: dict[str, ResultSet], audits: list[dict[str, Any]]
) -> dict[str, Any]:
    reference = result_sets["sft"]
    reference_keys = set(reference.simulations)
    reference_tasks = set(reference.tasks)
    key_mismatches: dict[str, Any] = {}
    seed_mismatches: dict[str, list[dict[str, Any]]] = {}
    task_definition_mismatches: dict[str, list[str]] = {}

    for label in MODEL_LABELS[1:]:
        current = result_sets[label]
        current_keys = set(current.simulations)
        if current_keys != reference_keys:
            key_mismatches[label] = {
                "missing": [
                    list(key)
                    for key in sorted(reference_keys - current_keys, key=key_sort_key)
                ],
                "extra": [
                    list(key)
                    for key in sorted(current_keys - reference_keys, key=key_sort_key)
                ],
            }

        mismatched_seeds = []
        for key in sorted(reference_keys & current_keys, key=key_sort_key):
            old_seed = reference.simulations[key].get("seed")
            new_seed = current.simulations[key].get("seed")
            if old_seed != new_seed:
                mismatched_seeds.append(
                    {"task_id": key[0], "trial": key[1], "sft": old_seed, label: new_seed}
                )
        if mismatched_seeds:
            seed_mismatches[label] = mismatched_seeds

        shared_tasks = reference_tasks & set(current.tasks)
        mismatched_tasks = [
            task_id
            for task_id in sorted(shared_tasks, key=task_sort_key)
            if canonical_json(comparable_task(reference.tasks[task_id]))
            != canonical_json(comparable_task(current.tasks[task_id]))
        ]
        missing_or_extra_tasks = reference_tasks ^ set(current.tasks)
        mismatched_tasks.extend(sorted(missing_or_extra_tasks, key=task_sort_key))
        if mismatched_tasks:
            task_definition_mismatches[label] = mismatched_tasks

    setting_paths = (
        ("num_trials",),
        ("max_steps",),
        ("max_errors",),
        ("seed",),
        ("user_info", "llm"),
        ("environment_info", "domain_name"),
    )
    setting_comparison = []
    for path in setting_paths:
        values = {
            label: nested_value(result_sets[label].info, path) for label in MODEL_LABELS
        }
        setting_comparison.append(
            {
                "field": ".".join(path),
                "values": values,
                "all_equal": len({canonical_json(value) for value in values.values()}) == 1,
            }
        )

    settings_match = all(item["all_equal"] for item in setting_comparison)
    report = {
        "valid": not (
            key_mismatches
            or seed_mismatches
            or task_definition_mismatches
            or not settings_match
        ),
        "datasets": audits,
        "key_mismatches": key_mismatches,
        "seed_mismatches": seed_mismatches,
        "task_definition_mismatches": task_definition_mismatches,
        "evaluation_settings": setting_comparison,
        "pairing_note": (
            "Matching task_id, trial, and seed makes these seed-aligned pairs. "
            "They are not exact counterfactual trajectories because policy divergence "
            "can change later user-simulator messages."
        ),
    }
    return report


def extract_reward(simulation: dict[str, Any]) -> float:
    reward_info = simulation.get("reward_info")
    reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
    if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
        raise ValueError(
            f"Invalid reward for task={simulation.get('task_id')}, "
            f"trial={simulation.get('trial')}: {reward!r}"
        )
    return float(reward)


def tool_family(name: str) -> str:
    if name.startswith(READ_PREFIXES):
        return "read"
    if name.startswith(WRITE_PREFIXES):
        return "write"
    if name == "calculate":
        return "utility"
    if "transfer" in name or "human" in name:
        return "transfer"
    return "other"


def normalize_error(text: str) -> str:
    normalized = re.sub(r"^\s*error:\s*", "", text, flags=re.IGNORECASE)
    normalized = re.sub(r"#W\d+", "<order_id>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b\d{5,}\b", "<id>", normalized)
    normalized = " ".join(normalized.split()).strip().lower()
    return normalized[:240] or "unspecified tool error"


def reference_actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    criteria = task.get("evaluation_criteria")
    if not isinstance(criteria, dict):
        return []
    actions = criteria.get("actions")
    if not isinstance(actions, list):
        return []
    return [action for action in actions if isinstance(action, dict)]


def reason_for_call(task: dict[str, Any]) -> str:
    scenario = task.get("user_scenario")
    instructions = scenario.get("instructions") if isinstance(scenario, dict) else None
    reason = instructions.get("reason_for_call") if isinstance(instructions, dict) else None
    return str(reason or "")


def reference_write_names(task: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(action.get("name"))
            for action in reference_actions(task)
            if isinstance(action.get("name"), str)
            and tool_family(str(action["name"])) == "write"
        }
    )


def reference_action_family(task: dict[str, Any]) -> str:
    names = reference_write_names(task)
    if not names:
        return "read_or_communication"
    families = sorted({name.split("_", 1)[0] for name in names})
    return "+".join(families)


def parse_tool_calls(messages: list[dict[str, Any]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    calls_by_id: dict[str, ToolCall] = {}
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            continue
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict) or not isinstance(raw_call.get("name"), str):
                continue
            call_id = raw_call.get("id")
            call = ToolCall(
                name=raw_call["name"],
                arguments=raw_call.get("arguments", {}),
                call_id=str(call_id) if call_id is not None else None,
                turn_idx=(
                    message.get("turn_idx")
                    if isinstance(message.get("turn_idx"), int)
                    else None
                ),
                message_index=message_index,
                family=tool_family(raw_call["name"]),
            )
            calls.append(call)
            if call.call_id:
                calls_by_id[call.call_id] = call

    for message_index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        call_id = message.get("id")
        call = calls_by_id.get(str(call_id)) if call_id is not None else None
        if call is None:
            continue
        call.error = bool(message.get("error"))
        call.error_text = normalize_error(str(message.get("content", ""))) if call.error else ""
        call.response_message_index = message_index

    for call in calls:
        if call.error:
            call.recovery = classify_recovery(call, calls, messages)
    return calls


def classify_recovery(
    failed_call: ToolCall,
    calls: list[ToolCall],
    messages: list[dict[str, Any]],
) -> str:
    if failed_call.response_message_index is None:
        return "missing_response"

    later_calls = [
        call for call in calls if call.message_index > failed_call.response_message_index
    ]
    next_call = min(later_calls, key=lambda call: call.message_index, default=None)
    next_user_index = next(
        (
            index
            for index, message in enumerate(messages)
            if index > failed_call.response_message_index and message.get("role") == "user"
        ),
        None,
    )
    if next_user_index is not None and (
        next_call is None or next_user_index < next_call.message_index
    ):
        return "ask_user"
    if next_call is None:
        return "abandon"
    if next_call.name != failed_call.name:
        return "alternate_tool"
    if next_call.signature == failed_call.signature:
        return "exact_retry"
    return "corrected_retry"


def assistant_token_metrics(messages: list[dict[str, Any]]) -> tuple[int, int, int]:
    turns = 0
    completion_tokens = 0
    prompt_tokens: list[int] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        # tau2 inserts a fixed greeting at turn 0. It has no usage record and is
        # not a policy generation, so exclude it from the assistant-turn count.
        if isinstance(usage, dict) or message.get("tool_calls"):
            turns += 1
        if not isinstance(usage, dict):
            continue
        if isinstance(usage.get("completion_tokens"), int):
            completion_tokens += usage["completion_tokens"]
        if isinstance(usage.get("prompt_tokens"), int):
            prompt_tokens.append(usage["prompt_tokens"])
    return turns, completion_tokens, max(prompt_tokens, default=0)


def extract_features(
    key: TrajectoryKey,
    simulation: dict[str, Any],
    task: dict[str, Any],
) -> TrajectoryFeatures:
    raw_messages = simulation.get("messages") or []
    messages = [message for message in raw_messages if isinstance(message, dict)]
    calls = parse_tool_calls(messages)
    signatures = [call.signature for call in calls]
    signature_counts = Counter(signatures)
    exact_repeated_calls = sum(count - 1 for count in signature_counts.values() if count > 1)
    consecutive_exact_retries = sum(
        left == right for left, right in zip(signatures, signatures[1:])
    )
    mixed_reply_turns = sum(
        message.get("role") == "assistant"
        and bool((message.get("content") or "").strip())
        and bool(message.get("tool_calls"))
        for message in messages
    )
    turns, completion_tokens, max_prompt_tokens = assistant_token_metrics(messages)

    reference_writes = reference_write_names(task)
    actual_write_names = {call.name for call in calls if call.family == "write"}
    unobserved_reference_writes = sorted(set(reference_writes) - actual_write_names)
    nonreference_writes = sorted(actual_write_names - set(reference_writes))

    termination_reason = str(simulation.get("termination_reason"))
    flags = []
    if termination_reason in ABNORMAL_TERMINATIONS:
        flags.append(f"abnormal_termination:{termination_reason}")
    if any(call.error and call.family == "write" for call in calls):
        flags.append("write_tool_error")
    if exact_repeated_calls:
        flags.append("exact_repeated_call")
    if mixed_reply_turns:
        flags.append("mixed_text_and_tool_call")
    if any(call.family == "transfer" for call in calls):
        flags.append("transfer_to_human")
    if any(call.error and call.family == "read" for call in calls):
        flags.append("read_tool_error_needs_review")

    reward = extract_reward(simulation)
    duration = simulation.get("duration")
    return TrajectoryFeatures(
        key=key,
        reward=reward,
        success=is_successful(reward),
        termination_reason=termination_reason,
        duration_seconds=float(duration) if isinstance(duration, (int, float)) else 0.0,
        assistant_turns=turns,
        completion_tokens=completion_tokens,
        max_prompt_tokens=max_prompt_tokens,
        calls=calls,
        mixed_reply_turns=mixed_reply_turns,
        exact_repeated_calls=exact_repeated_calls,
        consecutive_exact_retries=consecutive_exact_retries,
        unobserved_reference_writes=unobserved_reference_writes,
        nonreference_writes=nonreference_writes,
        diagnostic_flags=flags,
    )


def transition(old_success: bool, new_success: bool) -> str:
    if old_success and not new_success:
        return "regressed_1_to_0"
    if not old_success and new_success:
        return "improved_0_to_1"
    return "stable_success" if old_success else "stable_failure"


def group_type(success_count: int, expected_trials: int) -> str:
    if success_count == 0:
        return "all_failure"
    if success_count == expected_trials:
        return "all_success"
    return "mixed"


def feature_columns(prefix: str, feature: TrajectoryFeatures, model: str) -> dict[str, Any]:
    errors = Counter(call.error_text for call in feature.calls if call.error)
    recoveries = Counter(call.recovery for call in feature.calls if call.error)
    return {
        f"{prefix}_model": model,
        f"{prefix}_reward": feature.reward,
        f"{prefix}_success": feature.success,
        f"{prefix}_termination": feature.termination_reason,
        f"{prefix}_duration_seconds": feature.duration_seconds,
        f"{prefix}_assistant_turns": feature.assistant_turns,
        f"{prefix}_completion_tokens": feature.completion_tokens,
        f"{prefix}_max_prompt_tokens": feature.max_prompt_tokens,
        f"{prefix}_tool_calls": feature.total_calls,
        f"{prefix}_read_calls": feature.family_count("read"),
        f"{prefix}_write_calls": feature.family_count("write"),
        f"{prefix}_utility_calls": feature.family_count("utility"),
        f"{prefix}_transfer_calls": feature.family_count("transfer"),
        f"{prefix}_tool_errors": feature.tool_errors,
        f"{prefix}_read_errors": feature.read_errors,
        f"{prefix}_write_errors": feature.write_errors,
        f"{prefix}_exact_repeated_calls": feature.exact_repeated_calls,
        f"{prefix}_consecutive_exact_retries": feature.consecutive_exact_retries,
        f"{prefix}_mixed_reply_turns": feature.mixed_reply_turns,
        f"{prefix}_unobserved_reference_writes": " | ".join(
            feature.unobserved_reference_writes
        ),
        f"{prefix}_nonreference_writes": " | ".join(feature.nonreference_writes),
        f"{prefix}_tool_sequence": " -> ".join(call.name for call in feature.calls),
        f"{prefix}_error_messages": canonical_json(errors),
        f"{prefix}_error_recoveries": canonical_json(recoveries),
        f"{prefix}_diagnostic_flags": " | ".join(feature.diagnostic_flags),
    }


def build_paired_trials(
    result_sets: dict[str, ResultSet],
    features: dict[str, dict[TrajectoryKey, TrajectoryFeatures]],
) -> list[dict[str, Any]]:
    rows = []
    base = result_sets["sft"]
    for key in sorted(base.simulations, key=key_sort_key):
        task_id, trial = key
        task = base.tasks[task_id]
        reference = reference_actions(task)
        row: dict[str, Any] = {
            "task_id": task_id,
            "trial": trial,
            "seed": base.simulations[key].get("seed"),
            "reason_for_call": reason_for_call(task),
            "reference_action_family": reference_action_family(task),
            "reference_action_names": " -> ".join(
                str(action.get("name", "")) for action in reference
            ),
            "reference_write_names": " | ".join(reference_write_names(task)),
            "reference_actions": canonical_json(reference),
        }
        for label in MODEL_LABELS:
            row.update(feature_columns(label, features[label][key], result_sets[label].model))
        for old_label, new_label in COMPARISONS:
            name = f"{old_label}_to_{new_label}"
            row[name] = transition(
                features[old_label][key].success, features[new_label][key].success
            )
        rows.append(row)
    return rows


def task_review_priority(
    old_count: int,
    new_count: int,
    expected_trials: int,
    num_trial_flips: int,
) -> str:
    delta = new_count - old_count
    if (old_count == expected_trials and new_count < expected_trials) or (
        old_count == 0 and new_count > 0
    ):
        return "P0"
    if abs(delta) >= 2:
        return "P0"
    if delta or num_trial_flips:
        return "P1"
    return "P2"


def build_task_transitions(
    result_sets: dict[str, ResultSet],
    features: dict[str, dict[TrajectoryKey, TrajectoryFeatures]],
    expected_trials: int,
) -> list[dict[str, Any]]:
    rows = []
    tasks = result_sets["sft"].tasks
    for old_label, new_label in COMPARISONS:
        for task_id in sorted(tasks, key=task_sort_key):
            keys = [(task_id, trial) for trial in range(expected_trials)]
            old_count = sum(features[old_label][key].success for key in keys)
            new_count = sum(features[new_label][key].success for key in keys)
            delta = new_count - old_count
            regressed_trials = [
                trial
                for task, trial in keys
                if features[old_label][(task, trial)].success
                and not features[new_label][(task, trial)].success
            ]
            improved_trials = [
                trial
                for task, trial in keys
                if not features[old_label][(task, trial)].success
                and features[new_label][(task, trial)].success
            ]
            num_trial_flips = len(regressed_trials) + len(improved_trials)
            rows.append(
                {
                    "comparison": f"{old_label}_to_{new_label}",
                    "task_id": task_id,
                    "reason_for_call": reason_for_call(tasks[task_id]),
                    "reference_action_family": reference_action_family(tasks[task_id]),
                    "reference_write_names": " | ".join(
                        reference_write_names(tasks[task_id])
                    ),
                    "old_success_count": old_count,
                    "new_success_count": new_count,
                    "delta_successes": delta,
                    "old_reward_mean": old_count / expected_trials,
                    "new_reward_mean": new_count / expected_trials,
                    "delta_reward_mean": delta / expected_trials,
                    "old_group_type": group_type(old_count, expected_trials),
                    "new_group_type": group_type(new_count, expected_trials),
                    "regressed_trials": canonical_json(regressed_trials),
                    "improved_trials": canonical_json(improved_trials),
                    "change": (
                        "improved"
                        if delta > 0
                        else "regressed"
                        if delta < 0
                        else "unchanged_net_with_trial_flips"
                        if num_trial_flips
                        else "unchanged"
                    ),
                    "review_priority": task_review_priority(
                        old_count,
                        new_count,
                        expected_trials,
                        num_trial_flips,
                    ),
                }
            )
    return rows


def cohort_memberships(
    label: str,
    key: TrajectoryKey,
    features: dict[str, dict[TrajectoryKey, TrajectoryFeatures]],
) -> list[str]:
    feature = features[label][key]
    cohorts = ["overall", "success" if feature.success else "failure"]
    if label in PRIMARY_COMPARISON:
        old_label, new_label = PRIMARY_COMPARISON
        cohorts.append(
            "vanilla_to_potential:"
            + transition(features[old_label][key].success, features[new_label][key].success)
        )
    elif label == "sft":
        cohorts.append(
            "sft_to_potential:"
            + transition(feature.success, features["potential"][key].success)
        )
    return cohorts


def aggregate_tool_rows(
    label: str,
    model: str,
    cohort: str,
    cohort_features: list[TrajectoryFeatures],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[ToolCall]] = defaultdict(list)
    owner_by_call_id: dict[int, TrajectoryKey] = {}
    for feature in cohort_features:
        for call in feature.calls:
            owner_by_call_id[id(call)] = feature.key
            buckets[("all", "all_tools")].append(call)
            buckets[("family", call.family)].append(call)
            buckets[("tool", call.name)].append(call)

    rows = []
    for (level, name), calls in sorted(buckets.items()):
        error_calls = [call for call in calls if call.error]
        error_messages = Counter(call.error_text for call in error_calls)
        recoveries = Counter(call.recovery for call in error_calls)
        affected_keys = {owner_by_call_id[id(call)] for call in error_calls}
        rows.append(
            {
                "model_label": label,
                "model": model,
                "cohort": cohort,
                "level": level,
                "tool_or_family": name,
                "num_trajectories": len(cohort_features),
                "total_calls": len(calls),
                "error_calls": len(error_calls),
                "errors_per_100_calls": 100 * len(error_calls) / len(calls),
                "affected_trajectories": len(affected_keys),
                "affected_tasks": len({key[0] for key in affected_keys}),
                "exact_retry_after_error": recoveries["exact_retry"],
                "corrected_retry_after_error": recoveries["corrected_retry"],
                "alternate_tool_after_error": recoveries["alternate_tool"],
                "ask_user_after_error": recoveries["ask_user"],
                "abandon_after_error": recoveries["abandon"],
                "error_messages": canonical_json(error_messages),
            }
        )
    return rows


def build_tool_error_summary(
    result_sets: dict[str, ResultSet],
    features: dict[str, dict[TrajectoryKey, TrajectoryFeatures]],
) -> list[dict[str, Any]]:
    rows = []
    for label in MODEL_LABELS:
        by_cohort: dict[str, list[TrajectoryFeatures]] = defaultdict(list)
        for key in sorted(features[label], key=key_sort_key):
            for cohort in cohort_memberships(label, key, features):
                by_cohort[cohort].append(features[label][key])
        for cohort, cohort_features in sorted(by_cohort.items()):
            rows.extend(
                aggregate_tool_rows(
                    label, result_sets[label].model, cohort, cohort_features
                )
            )
    return rows


def review_trace(simulation: dict[str, Any]) -> list[dict[str, Any]]:
    tool_names: dict[str, str] = {}
    for message in simulation.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id") is not None:
                tool_names[str(call["id"])] = str(call.get("name", "unknown"))

    trace = []
    for message in simulation.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            item: dict[str, Any] = {
                "turn_idx": message.get("turn_idx"),
                "role": "assistant",
            }
            if message.get("content"):
                item["content"] = str(message["content"])
            if calls:
                item["tool_calls"] = [
                    {
                        "name": call.get("name"),
                        "arguments": call.get("arguments"),
                    }
                    for call in calls
                    if isinstance(call, dict)
                ]
            trace.append(item)
        elif role == "user":
            trace.append(
                {
                    "turn_idx": message.get("turn_idx"),
                    "role": "user",
                    "content": str(message.get("content") or ""),
                }
            )
        elif role == "tool":
            call_id = message.get("id")
            trace.append(
                {
                    "turn_idx": message.get("turn_idx"),
                    "role": "tool",
                    "tool_name": tool_names.get(str(call_id), "unknown"),
                    "error": bool(message.get("error")),
                    "content": str(message.get("content") or ""),
                }
            )
    return trace


def review_model_payload(
    result_set: ResultSet,
    feature: TrajectoryFeatures,
    simulation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": result_set.model,
        "reward": feature.reward,
        "success": feature.success,
        "termination_reason": feature.termination_reason,
        "duration_seconds": feature.duration_seconds,
        "diagnostic_flags": feature.diagnostic_flags,
        "unobserved_reference_writes": feature.unobserved_reference_writes,
        "nonreference_writes": feature.nonreference_writes,
        "trace": review_trace(simulation),
    }


def build_review_cases(
    result_sets: dict[str, ResultSet],
    features: dict[str, dict[TrajectoryKey, TrajectoryFeatures]],
) -> list[dict[str, Any]]:
    cases = []
    tasks = result_sets["sft"].tasks
    for key in sorted(result_sets["sft"].simulations, key=key_sort_key):
        transitions = {
            f"{old}_to_{new}": transition(
                features[old][key].success, features[new][key].success
            )
            for old, new in COMPARISONS
        }
        if not any(
            value.startswith("regressed") or value.startswith("improved")
            for value in transitions.values()
        ):
            continue
        task = tasks[key[0]]
        cases.append(
            {
                "task_id": key[0],
                "trial": key[1],
                "seed": result_sets["sft"].simulations[key].get("seed"),
                "transitions": transitions,
                "task": {
                    "reason_for_call": reason_for_call(task),
                    "reference_action_family": reference_action_family(task),
                    "reference_actions": reference_actions(task),
                },
                "models": {
                    label: review_model_payload(
                        result_sets[label],
                        features[label][key],
                        result_sets[label].simulations[key],
                    )
                    for label in MODEL_LABELS
                },
                "manual_attribution": {
                    "primary_cause": None,
                    "secondary_flags": [],
                    "first_decisive_turn": None,
                    "tool_name": None,
                    "evidence": None,
                    "confidence": None,
                },
            }
        )
    return cases


def build_manual_attribution_template(
    review_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case in review_cases:
        primary_transition = case["transitions"]["vanilla_to_potential"]
        if primary_transition in {"stable_success", "stable_failure"}:
            continue
        rows.append(
            {
                "task_id": case["task_id"],
                "trial": case["trial"],
                "seed": case["seed"],
                "transition": primary_transition,
                "reference_action_family": case["task"]["reference_action_family"],
                "primary_cause": "",
                "secondary_flags": "",
                "first_decisive_turn": "",
                "tool_name": "",
                "evidence": "",
                "confidence": "",
                "reviewer_notes": "",
            }
        )
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def model_overview(
    result_set: ResultSet, model_features: dict[TrajectoryKey, TrajectoryFeatures]
) -> dict[str, Any]:
    values = list(model_features.values())
    total_calls = sum(feature.total_calls for feature in values)
    total_errors = sum(feature.tool_errors for feature in values)
    return {
        "label": result_set.label,
        "model": result_set.model,
        "successes": sum(feature.success for feature in values),
        "total": len(values),
        "average_reward": mean(feature.reward for feature in values),
        "average_tool_calls": mean(feature.total_calls for feature in values),
        "tool_errors_per_100_calls": 100 * total_errors / total_calls if total_calls else 0.0,
        "average_assistant_turns": mean(feature.assistant_turns for feature in values),
        "average_completion_tokens": mean(feature.completion_tokens for feature in values),
        "abnormal_terminations": sum(
            feature.termination_reason in ABNORMAL_TERMINATIONS for feature in values
        ),
    }


def comparison_summary(
    old_label: str,
    new_label: str,
    features: dict[str, dict[TrajectoryKey, TrajectoryFeatures]],
    expected_trials: int,
) -> dict[str, Any]:
    counts = Counter(
        transition(features[old_label][key].success, features[new_label][key].success)
        for key in features[old_label]
    )
    matrix = {
        str(old_count): {str(new_count): 0 for new_count in range(expected_trials + 1)}
        for old_count in range(expected_trials + 1)
    }
    task_deltas = []
    task_ids = sorted({key[0] for key in features[old_label]}, key=task_sort_key)
    for task_id in task_ids:
        keys = [(task_id, trial) for trial in range(expected_trials)]
        old_count = sum(features[old_label][key].success for key in keys)
        new_count = sum(features[new_label][key].success for key in keys)
        matrix[str(old_count)][str(new_count)] += 1
        regressed_trials = [
            key[1]
            for key in keys
            if features[old_label][key].success and not features[new_label][key].success
        ]
        improved_trials = [
            key[1]
            for key in keys
            if not features[old_label][key].success and features[new_label][key].success
        ]
        if old_count != new_count or regressed_trials or improved_trials:
            task_deltas.append(
                {
                    "task_id": task_id,
                    "old_successes": old_count,
                    "new_successes": new_count,
                    "delta": new_count - old_count,
                    "regressed_trials": regressed_trials,
                    "improved_trials": improved_trials,
                }
            )
    return {
        "comparison": f"{old_label}_to_{new_label}",
        "trial_transitions": dict(counts),
        "task_success_count_matrix": matrix,
        "changed_tasks": task_deltas,
    }


def cohort_diagnostics(
    transition_name: str,
    features: dict[str, dict[TrajectoryKey, TrajectoryFeatures]],
) -> dict[str, Any]:
    selected = [
        features["potential"][key]
        for key in features["potential"]
        if transition(features["vanilla"][key].success, features["potential"][key].success)
        == transition_name
    ]
    return {
        "trajectories": len(selected),
        "mean_tool_calls": mean(item.total_calls for item in selected),
        "mean_read_calls": mean(item.family_count("read") for item in selected),
        "mean_write_calls": mean(item.family_count("write") for item in selected),
        "mean_tool_errors": mean(item.tool_errors for item in selected),
        "mean_exact_repeated_calls": mean(item.exact_repeated_calls for item in selected),
        "write_error_trajectories": sum(item.write_errors > 0 for item in selected),
        "mixed_reply_trajectories": sum(item.mixed_reply_turns > 0 for item in selected),
        "abnormal_terminations": sum(
            item.termination_reason in ABNORMAL_TERMINATIONS for item in selected
        ),
        "mean_assistant_turns": mean(item.assistant_turns for item in selected),
        "mean_completion_tokens": mean(item.completion_tokens for item in selected),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows:
        if not fieldnames:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as output:
            csv.DictWriter(output, fieldnames=fieldnames).writeheader()
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def format_task_changes(changes: list[dict[str, Any]], direction: str) -> str:
    selected = [
        item
        for item in changes
        if (item["delta"] > 0 if direction == "improved" else item["delta"] < 0)
    ]
    selected.sort(key=lambda item: (-abs(item["delta"]), task_sort_key(item["task_id"])))
    if not selected:
        return "None"
    return ", ".join(
        f"{item['task_id']} ({item['old_successes']}→{item['new_successes']})"
        for item in selected
    )


def format_net_zero_flips(changes: list[dict[str, Any]]) -> str:
    selected = [item for item in changes if item["delta"] == 0]
    if not selected:
        return "None"
    return ", ".join(
        (
            f"{item['task_id']} "
            f"(1→0 trials={item['regressed_trials']}, "
            f"0→1 trials={item['improved_trials']})"
        )
        for item in selected
    )


def write_summary(
    path: Path,
    alignment: dict[str, Any],
    overviews: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Retail paired regression analysis",
        "",
        "## Data alignment",
        "",
        f"Alignment valid: **{alignment['valid']}**.",
        "",
        alignment["pairing_note"],
        "",
        "## Model overview",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "Label",
                "Success",
                "Reward",
                "Tool calls / traj",
                "Errors / 100 calls",
                "Assistant turns",
                "Completion tokens",
            ],
            [
                [
                    item["label"],
                    f"{item['successes']}/{item['total']}",
                    f"{item['average_reward']:.4f}",
                    f"{item['average_tool_calls']:.2f}",
                    f"{item['tool_errors_per_100_calls']:.2f}",
                    f"{item['average_assistant_turns']:.2f}",
                    f"{item['average_completion_tokens']:.1f}",
                ]
                for item in overviews
            ],
        )
    )
    lines.extend(["", "## Paired outcome transitions", ""])
    for item in comparisons:
        counts = item["trial_transitions"]
        lines.extend(
            [
                f"### {item['comparison']}",
                "",
                (
                    f"Trial level: regressions={counts.get('regressed_1_to_0', 0)}, "
                    f"improvements={counts.get('improved_0_to_1', 0)}, "
                    f"stable successes={counts.get('stable_success', 0)}, "
                    f"stable failures={counts.get('stable_failure', 0)}."
                ),
                "",
                "Regressed tasks: "
                + format_task_changes(item["changed_tasks"], "regressed")
                + ".",
                "",
                "Improved tasks: "
                + format_task_changes(item["changed_tasks"], "improved")
                + ".",
                "",
                "Net-zero tasks with paired flips: "
                + format_net_zero_flips(item["changed_tasks"])
                + ".",
                "",
            ]
        )

    lines.extend(["## Potential-model diagnostic cohorts", ""])
    diagnostic_rows = []
    for name in ("regressed_1_to_0", "improved_0_to_1"):
        item = diagnostics[name]
        diagnostic_rows.append(
            [
                name,
                item["trajectories"],
                f"{item['mean_tool_calls']:.2f}",
                f"{item['mean_tool_errors']:.2f}",
                f"{item['mean_exact_repeated_calls']:.2f}",
                item["write_error_trajectories"],
                item["mixed_reply_trajectories"],
                item["abnormal_terminations"],
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Cohort",
                "N",
                "Tool calls",
                "Tool errors",
                "Exact repeats",
                "Write-error trajectories",
                "Mixed reply",
                "Abnormal stop",
            ],
            diagnostic_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "- Tool `error=true` is an observed environment response, not "
                "automatically a model-attributable error."
            ),
            (
                "- Use `manual_attribution.csv` and `review_cases.jsonl` to assign "
                "one primary cause to each Vanilla→Potential flip."
            ),
            (
                "- Reference actions describe one replay path to the target DB; "
                "differences from that path are descriptive and are not automatic errors."
            ),
            (
                "- The selected Vanilla and Potential checkpoints occur at different "
                "training steps; this is a selected-model comparison, not a clean "
                "causal estimate of the reward method alone."
            ),
            (
                "- If these test cases influence the next method design, report the "
                "test set as analyzed rather than untouched."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.expected_trials < 1:
        raise ValueError("--expected-trials must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "sft": args.sft,
        "vanilla": args.vanilla,
        "potential": args.potential,
    }
    result_sets: dict[str, ResultSet] = {}
    audits = []
    for label in MODEL_LABELS:
        result_set, audit = load_result_set(
            label, input_paths[label], args.expected_trials
        )
        result_sets[label] = result_set
        audits.append(audit)

    alignment = validate_alignment(result_sets, audits)
    write_json(output_dir / "alignment.json", alignment)
    if not alignment["valid"]:
        raise ValueError(
            f"Evaluation files are not aligned; inspect {output_dir / 'alignment.json'}"
        )

    features = {
        label: {
            key: extract_features(
                key,
                simulation,
                result_sets[label].tasks[key[0]],
            )
            for key, simulation in result_sets[label].simulations.items()
        }
        for label in MODEL_LABELS
    }
    paired_trials = build_paired_trials(result_sets, features)
    task_transitions = build_task_transitions(
        result_sets, features, args.expected_trials
    )
    tool_summary = build_tool_error_summary(result_sets, features)
    review_cases = build_review_cases(result_sets, features)
    manual_template = build_manual_attribution_template(review_cases)

    overviews = [
        model_overview(result_sets[label], features[label]) for label in MODEL_LABELS
    ]
    comparisons = [
        comparison_summary(old, new, features, args.expected_trials)
        for old, new in COMPARISONS
    ]
    diagnostics = {
        name: cohort_diagnostics(name, features)
        for name in ("regressed_1_to_0", "improved_0_to_1")
    }

    write_csv(output_dir / "paired_trials.csv", paired_trials)
    write_csv(output_dir / "per_task_transitions.csv", task_transitions)
    write_csv(output_dir / "tool_error_summary.csv", tool_summary)
    write_jsonl(output_dir / "review_cases.jsonl", review_cases)
    write_csv(
        output_dir / "manual_attribution.csv",
        manual_template,
        fieldnames=[
            "task_id",
            "trial",
            "seed",
            "transition",
            "reference_action_family",
            "primary_cause",
            "secondary_flags",
            "first_decisive_turn",
            "tool_name",
            "evidence",
            "confidence",
            "reviewer_notes",
        ],
    )
    write_json(
        output_dir / "paired_summary.json",
        {
            "alignment": alignment,
            "model_overview": overviews,
            "comparisons": comparisons,
            "potential_diagnostic_cohorts": diagnostics,
        },
    )
    write_summary(
        output_dir / "summary.md",
        alignment,
        overviews,
        comparisons,
        diagnostics,
    )

    print(f"Reports written to: {output_dir}")
    for item in overviews:
        print(
            f"{item['label']}: {item['successes']}/{item['total']} successes, "
            f"{item['tool_errors_per_100_calls']:.2f} observed tool errors/100 calls"
        )
    primary = next(
        item for item in comparisons if item["comparison"] == "vanilla_to_potential"
    )
    counts = primary["trial_transitions"]
    print(
        "Vanilla -> Potential: "
        f"regressions={counts.get('regressed_1_to_0', 0)}, "
        f"improvements={counts.get('improved_0_to_1', 0)}"
    )
    print(
        "Review order: manual_attribution.csv first, then review_cases.jsonl for evidence."
    )


if __name__ == "__main__":
    main()
