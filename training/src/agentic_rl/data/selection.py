import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher


def quality_score(candidate: dict) -> float:
    stats = candidate["stats"]
    flags = set(candidate["screening"]["review_flags"])
    score = float(candidate["semantic_screening"]["confidence"])
    score -= 0.02 * min(stats["tool_error_count"], 3)
    score -= 0.04 * ("long_trajectory" in flags)
    score -= 0.04 * ("many_tool_calls" in flags)
    score -= 0.002 * max(stats["assistant_turns"] - 10, 0)
    return max(score, 0.0)


def trajectory_features(candidate: dict) -> tuple[set[str], tuple[str, ...]]:
    text_parts = []
    tool_trace = []
    for message in candidate["messages"]:
        if message["role"] in {"user", "assistant"} and message.get("content"):
            text_parts.append(message["content"].lower())
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                arguments = json.dumps(
                    call["arguments"], ensure_ascii=False, sort_keys=True
                )
                tool_trace.append(f"{call['name']}:{arguments}")

    words = re.findall(r"[a-z0-9]+", " ".join(text_parts))
    if len(words) < 3:
        text_ngrams = set(words)
    else:
        text_ngrams = {
            " ".join(words[index : index + 3])
            for index in range(len(words) - 2)
        }
    return text_ngrams, tuple(tool_trace)


def trajectory_similarity(
    left: tuple[set[str], tuple[str, ...]],
    right: tuple[set[str], tuple[str, ...]],
) -> float:
    left_text, left_tools = left
    right_text, right_tools = right
    union = left_text | right_text
    text_similarity = len(left_text & right_text) / len(union) if union else 1.0
    tool_similarity = SequenceMatcher(None, left_tools, right_tools).ratio()
    return 0.75 * text_similarity + 0.25 * tool_similarity


def select_candidates(
    candidates: list[dict],
    target_size: int = 256,
    min_per_task: int = 2,
) -> list[dict]:
    eligible = [
        row
        for row in candidates
        if row["semantic_screening"]["decision"] == "accepted"
    ]
    if target_size > len(eligible):
        raise ValueError("target_size exceeds the number of accepted candidates")

    by_task = defaultdict(list)
    for row in eligible:
        by_task[row["task_id"]].append(row)
    if any(len(rows) < min_per_task for rows in by_task.values()):
        raise ValueError("at least one task has fewer accepted rows than min_per_task")

    features = {
        row["trajectory_id"]: trajectory_features(row) for row in eligible
    }
    selected = []
    selected_by_task = defaultdict(list)
    remaining = {row["trajectory_id"]: row for row in eligible}
    tier_counts = Counter()
    tier_targets = {
        "low": target_size // 4,
        "medium": target_size // 2,
        "high": target_size - target_size // 4 - target_size // 2,
    }

    def score(row: dict) -> tuple:
        task_rows = selected_by_task[row["task_id"]]
        max_similarity = max(
            (
                trajectory_similarity(
                    features[row["trajectory_id"]],
                    features[chosen["trajectory_id"]],
                )
                for chosen in task_rows
            ),
            default=0.0,
        )
        novelty = 1.0 - max_similarity
        tier = row["tier"]
        tier_need = max(tier_targets[tier] - tier_counts[tier], 0) / max(
            tier_targets[tier], 1
        )
        task_balance = 1.0 / (len(task_rows) + 1)
        combined = (
            0.70 * quality_score(row)
            + 0.20 * novelty
            + 0.06 * tier_need
            + 0.04 * task_balance
        )
        return combined, -row["screening"]["task_rank"], row["trajectory_id"]

    def add_best(pool: list[dict], phase: str) -> None:
        chosen = max(pool, key=score)
        chosen_score = score(chosen)[0]
        chosen["selection"] = {
            "phase": phase,
            "score": round(chosen_score, 6),
            "quality_score": round(quality_score(chosen), 6),
        }
        selected.append(chosen)
        selected_by_task[chosen["task_id"]].append(chosen)
        tier_counts[chosen["tier"]] += 1
        remaining.pop(chosen["trajectory_id"])

    for task_id in sorted(by_task, key=int):
        for _ in range(min_per_task):
            pool = [
                row
                for row in by_task[task_id]
                if row["trajectory_id"] in remaining
            ]
            add_best(pool, "task_coverage")

    while len(selected) < target_size:
        add_best(list(remaining.values()), "global_diversity")

    for index, row in enumerate(selected, start=1):
        row["selection"]["selected_index"] = index
    return selected


def build_selection_summary(
    candidates: list[dict], selected: list[dict]
) -> dict:
    accepted = [
        row
        for row in candidates
        if row["semantic_screening"]["decision"] == "accepted"
    ]
    task_counts = Counter(row["task_id"] for row in selected)
    return {
        "accepted_candidates": len(accepted),
        "selected_candidates": len(selected),
        "unselected_accepted": len(accepted) - len(selected),
        "task_coverage": len(task_counts),
        "task_count_distribution": dict(sorted(Counter(task_counts.values()).items())),
        "tier_distribution": dict(Counter(row["tier"] for row in selected)),
        "selection_phase_distribution": dict(
            Counter(row["selection"]["phase"] for row in selected)
        ),
        "review_flag_distribution": dict(
            Counter(
                flag
                for row in selected
                for flag in row["screening"]["review_flags"]
            )
        ),
        "selected_with_tool_errors": sum(
            row["stats"]["tool_error_count"] > 0 for row in selected
        ),
        "task_counts": dict(
            sorted(task_counts.items(), key=lambda item: int(item[0]))
        ),
    }
