from collections import Counter

TIER_ORDER = {"low": 0, "medium": 1, "high": 2}
LONG_TRAJECTORY_TURNS = 20
MANY_TOOL_CALLS = 15


def inspect_candidate(candidate: dict) -> dict:
    messages = candidate["messages"]
    outcome = candidate["outcome"]
    stats = candidate["stats"]
    rejection_reasons = []
    review_flags = []

    if outcome["reward"] != 1.0:
        rejection_reasons.append("reward_zero")
    if outcome["db_match"] is False:
        rejection_reasons.append("db_mismatch")
    if outcome["termination_reason"] not in {"user_stop", "agent_stop"}:
        rejection_reasons.append("abnormal_termination")
    if not messages:
        rejection_reasons.append("missing_messages")
    if stats["assistant_turns"] == 0:
        rejection_reasons.append("missing_assistant_turn")

    for message in messages:
        if message["role"] != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        has_text = bool((message.get("content") or "").strip())
        if len(tool_calls) > 1:
            review_flags.append("parallel_tool_calls")
        if tool_calls and has_text:
            review_flags.append("text_with_tool_call")
        for tool_call in tool_calls:
            if not tool_call.get("name") or not isinstance(
                tool_call.get("arguments"), dict
            ):
                rejection_reasons.append("malformed_tool_call")

    if stats["tool_error_count"]:
        review_flags.append("tool_error_needs_review")
    if stats["assistant_turns"] > LONG_TRAJECTORY_TURNS:
        review_flags.append("long_trajectory")
    if stats["tool_call_count"] > MANY_TOOL_CALLS:
        review_flags.append("many_tool_calls")

    return {
        "hard_pass": not rejection_reasons,
        "hard_rejection_reasons": sorted(set(rejection_reasons)),
        "review_flags": sorted(set(review_flags)),
        "task_rank": None,
    }


def rank_key(candidate: dict) -> tuple:
    screening = candidate["screening"]
    return (
        not screening["hard_pass"],
        len(screening["review_flags"]),
        candidate["stats"]["tool_error_count"],
        candidate["stats"]["assistant_turns"],
        candidate["stats"]["tool_call_count"],
        TIER_ORDER[candidate["tier"]],
        candidate["trial"],
    )


def rank_by_task(candidates: list[dict]) -> list[dict]:
    ranked = []
    task_ids = sorted({candidate["task_id"] for candidate in candidates}, key=int)
    for task_id in task_ids:
        task_rows = [row for row in candidates if row["task_id"] == task_id]
        task_rows.sort(key=rank_key)
        for rank, row in enumerate(task_rows, start=1):
            row["screening"]["task_rank"] = rank
        ranked.extend(task_rows)
    return ranked


def build_summary(candidates: list[dict]) -> dict:
    hard_reasons = Counter(
        reason
        for candidate in candidates
        for reason in candidate["screening"]["hard_rejection_reasons"]
    )
    review_flags = Counter(
        flag
        for candidate in candidates
        for flag in candidate["screening"]["review_flags"]
    )
    task_counts = []
    for task_id in sorted({row["task_id"] for row in candidates}, key=int):
        rows = [row for row in candidates if row["task_id"] == task_id]
        task_counts.append(
            {
                "task_id": task_id,
                "total": len(rows),
                "hard_pass": sum(row["screening"]["hard_pass"] for row in rows),
                "needs_review": sum(
                    bool(row["screening"]["review_flags"])
                    and row["screening"]["hard_pass"]
                    for row in rows
                ),
            }
        )

    return {
        "total": len(candidates),
        "hard_pass": sum(row["screening"]["hard_pass"] for row in candidates),
        "hard_rejected": sum(
            not row["screening"]["hard_pass"] for row in candidates
        ),
        "hard_rejection_counts": dict(hard_reasons),
        "review_flag_counts": dict(review_flags),
        "task_counts": task_counts,
    }


def screen_and_rank(candidates: list[dict]) -> tuple[list[dict], dict]:
    for candidate in candidates:
        candidate["screening"] = inspect_candidate(candidate)
    ranked = rank_by_task(candidates)
    return ranked, build_summary(ranked)
