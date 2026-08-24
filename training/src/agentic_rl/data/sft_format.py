from collections import Counter

from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE


def convert_message(message: dict) -> dict:
    role = message["role"]
    converted = {"role": role, "content": message.get("content")}
    tool_calls = message.get("tool_calls") or []

    if tool_calls:
        if role != "assistant":
            raise ValueError("only assistant tool calls are supported in SFT data")
        converted["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
            for call in tool_calls
        ]
    if role == "tool":
        converted["tool_call_id"] = message["tool_call_id"]
    return converted


def build_sft_messages(candidate: dict, system_prompt: str) -> list[dict]:
    source_messages = list(candidate["messages"])
    if (
        source_messages
        and source_messages[0]["role"] == "assistant"
        and source_messages[0].get("content")
        == DEFAULT_FIRST_AGENT_MESSAGE.content
        and not source_messages[0].get("tool_calls")
    ):
        source_messages = source_messages[1:]

    # The simulator often acknowledges the final answer and then ends the run.
    # It has no following assistant target, so it is not part of the SFT example.
    while source_messages and source_messages[-1]["role"] == "user":
        source_messages = source_messages[:-1]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(convert_message(message) for message in source_messages)
    validate_sft_messages(messages)
    return messages


def validate_sft_messages(messages: list[dict]) -> None:
    if not messages or messages[0]["role"] != "system":
        raise ValueError("SFT conversation must start with a system message")
    if len(messages) < 3 or messages[1]["role"] != "user":
        raise ValueError("SFT conversation must contain an initial user message")
    if messages[-1]["role"] != "assistant":
        raise ValueError("SFT conversation must end with an assistant message")

    calls = Counter()
    results = Counter()
    assistant_targets = 0
    for message in messages:
        role = message["role"]
        content = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role}")
        if role == "assistant":
            if bool(content) == bool(tool_calls):
                raise ValueError("assistant message must contain text or tool calls")
            assistant_targets += 1
            calls.update(call["id"] for call in tool_calls)
        if role == "tool":
            results[message["tool_call_id"]] += 1

    if calls != results:
        raise ValueError("assistant tool calls and tool results are not paired")
    if assistant_targets == 0:
        raise ValueError("SFT conversation has no assistant target")


def build_sft_row(candidate: dict, system_prompt: str, tools: list[dict]) -> dict:
    return {
        "trajectory_id": candidate["trajectory_id"],
        "task_id": candidate["task_id"],
        "messages": build_sft_messages(candidate, system_prompt),
        "tools": tools,
        "metadata": {
            "tier": candidate["tier"],
            "temperature": candidate["temperature"],
            "trial": candidate["trial"],
            "seed": candidate["seed"],
            "source_run": candidate["source_run"],
            "selection": candidate["selection"],
        },
    }


def build_sft_summary(rows: list[dict], source_path: str) -> dict:
    message_counts = Counter(
        message["role"] for row in rows for message in row["messages"]
    )
    return {
        "format": "openai_messages_with_tools",
        "source_path": source_path,
        "examples": len(rows),
        "task_coverage": len({row["task_id"] for row in rows}),
        "tool_schema_count": len(rows[0]["tools"]) if rows else 0,
        "message_role_counts": dict(message_counts),
        "intended_loss_mask": "assistant_only",
        "tokenized": False,
        "target_model": None,
        "removed_tau2_greeting": True,
        "removed_user_stop": True,
    }
