"""Canonical, model-independent schema for offline SFT preprocessing.

This module is deliberately unaware of Qwen, chat templates, token IDs, labels,
padding, and PyTorch. Its only job is to parse one JSONL row into a typed object
and reject structurally unsafe trajectories before model-specific rendering.
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

SUPPORTED_ROLES = {"system", "user", "assistant", "tool"}
TOP_LEVEL_KEYS = {"trajectory_id", "task_id", "messages", "tools", "metadata"}


class SchemaValidationError(ValueError):
    """A validation error with a precise source and field location."""

    def __init__(self, location: str, message: str) -> None:
        self.location = location
        self.message = message
        super().__init__(f"{location}: {message}")


@dataclass(frozen=True)
class FunctionCall:
    """The model-generated function name and JSON arguments."""

    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": deepcopy(self.arguments)}


@dataclass(frozen=True)
class ToolCall:
    """One OpenAI-compatible assistant tool call."""

    id: str
    type: str
    function: FunctionCall

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_dict(),
        }


@dataclass(frozen=True)
class Message:
    """One canonical conversation message."""

    role: str
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def to_chat_dict(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message


@dataclass(frozen=True)
class ToolDefinition:
    """One function schema supplied to the target model."""

    type: str
    name: str
    description: str | None
    parameters: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)

    def to_chat_dict(self) -> dict[str, Any]:
        return deepcopy(self.raw)


@dataclass(frozen=True)
class SFTExample:
    """One validated, model-independent SFT trajectory."""

    trajectory_id: str
    task_id: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    metadata: dict[str, Any]
    source: str | None = None

    def chat_messages(self) -> list[dict[str, Any]]:
        return [message.to_chat_dict() for message in self.messages]

    def chat_tools(self) -> list[dict[str, Any]]:
        return [tool.to_chat_dict() for tool in self.tools]


def _error(location: str, message: str) -> SchemaValidationError:
    return SchemaValidationError(location, message)


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(location, "expected a JSON object")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(location, "expected a JSON array")
    return value


def _require_non_empty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(location, "expected a non-empty string")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    location: str,
) -> None:
    keys = set(value)
    missing = required - keys
    unexpected = keys - required - optional
    if missing:
        raise _error(location, f"missing keys: {', '.join(sorted(missing))}")
    if unexpected:
        raise _error(location, f"unexpected keys: {', '.join(sorted(unexpected))}")


def _validate_json_value(value: Any, location: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise _error(location, "value is not strict JSON-serializable") from error


def _parse_function_call(value: Any, location: str) -> FunctionCall:
    function = _require_mapping(value, location)
    _require_exact_keys(
        function,
        required={"name", "arguments"},
        optional=set(),
        location=location,
    )
    name = _require_non_empty_string(function["name"], f"{location}.name")
    arguments = _require_mapping(function["arguments"], f"{location}.arguments")
    if any(not isinstance(key, str) for key in arguments):
        raise _error(f"{location}.arguments", "all argument keys must be strings")
    arguments_dict = deepcopy(dict(arguments))
    _validate_json_value(arguments_dict, f"{location}.arguments")
    return FunctionCall(name=name, arguments=arguments_dict)


def _parse_tool_call(value: Any, location: str) -> ToolCall:
    call = _require_mapping(value, location)
    _require_exact_keys(
        call,
        required={"id", "type", "function"},
        optional=set(),
        location=location,
    )
    call_id = _require_non_empty_string(call["id"], f"{location}.id")
    call_type = _require_non_empty_string(call["type"], f"{location}.type")
    if call_type != "function":
        raise _error(f"{location}.type", "only function tool calls are supported")
    return ToolCall(
        id=call_id,
        type=call_type,
        function=_parse_function_call(call["function"], f"{location}.function"),
    )


def _parse_message(value: Any, index: int, source: str) -> Message:
    location = f"{source}.messages[{index}]"
    message = _require_mapping(value, location)
    role = _require_non_empty_string(message.get("role"), f"{location}.role")
    if role not in SUPPORTED_ROLES:
        raise _error(f"{location}.role", f"unsupported role {role!r}")

    if role in {"system", "user"}:
        _require_exact_keys(
            message,
            required={"role", "content"},
            optional=set(),
            location=location,
        )
        content = _require_non_empty_string(message["content"], f"{location}.content")
        return Message(role=role, content=content)

    if role == "assistant":
        _require_exact_keys(
            message,
            required={"role", "content"},
            optional={"tool_calls"},
            location=location,
        )
        content_value = message["content"]
        if content_value is not None and not isinstance(content_value, str):
            raise _error(f"{location}.content", "expected a string or null")
        has_content = isinstance(content_value, str) and bool(content_value.strip())
        raw_calls = message.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        calls = tuple(
            _parse_tool_call(call, f"{location}.tool_calls[{call_index}]")
            for call_index, call in enumerate(
                _require_list(raw_calls, f"{location}.tool_calls")
            )
        )
        if has_content == bool(calls):
            raise _error(
                location,
                "assistant must contain exactly one of non-empty content or tool_calls",
            )
        if len(calls) > 1:
            raise _error(location, "strict serial SFT permits at most one tool call")
        return Message(role=role, content=content_value, tool_calls=calls)

    _require_exact_keys(
        message,
        required={"role", "content", "tool_call_id"},
        optional=set(),
        location=location,
    )
    content_value = message["content"]
    if not isinstance(content_value, str):
        raise _error(f"{location}.content", "tool content must be a string")
    return Message(
        role=role,
        content=content_value,
        tool_call_id=_require_non_empty_string(
            message["tool_call_id"], f"{location}.tool_call_id"
        ),
    )


def _parse_tool_definition(value: Any, index: int, source: str) -> ToolDefinition:
    location = f"{source}.tools[{index}]"
    tool = _require_mapping(value, location)
    _require_exact_keys(
        tool,
        required={"type", "function"},
        optional=set(),
        location=location,
    )
    tool_type = _require_non_empty_string(tool["type"], f"{location}.type")
    if tool_type != "function":
        raise _error(f"{location}.type", "only function tools are supported")

    function_location = f"{location}.function"
    function = _require_mapping(tool["function"], function_location)
    _require_exact_keys(
        function,
        required={"name", "parameters"},
        optional={"description", "strict"},
        location=function_location,
    )
    name = _require_non_empty_string(function["name"], f"{function_location}.name")
    description = function.get("description")
    if description is not None and not isinstance(description, str):
        raise _error(f"{function_location}.description", "expected a string or null")
    parameters = _require_mapping(
        function["parameters"], f"{function_location}.parameters"
    )
    parameters_dict = deepcopy(dict(parameters))
    _validate_json_value(parameters_dict, f"{function_location}.parameters")
    raw = deepcopy(dict(tool))
    _validate_json_value(raw, location)
    return ToolDefinition(
        type=tool_type,
        name=name,
        description=description,
        parameters=parameters_dict,
        raw=raw,
    )


def _validate_conversation(
    messages: tuple[Message, ...],
    tools: tuple[ToolDefinition, ...],
    source: str,
) -> None:
    if len(messages) < 3:
        raise _error(f"{source}.messages", "conversation must contain at least 3 messages")
    if messages[0].role != "system":
        raise _error(f"{source}.messages[0]", "conversation must start with system")
    if messages[1].role != "user":
        raise _error(f"{source}.messages[1]", "system must be followed by user")
    if messages[-1].role != "assistant" or not messages[-1].content:
        raise _error(
            f"{source}.messages[{len(messages) - 1}]",
            "conversation must end with an assistant text response",
        )
    if any(message.role == "system" for message in messages[1:]):
        raise _error(f"{source}.messages", "system is only allowed as the first message")

    tool_names = [tool.name for tool in tools]
    duplicate_names = [
        name for name, count in Counter(tool_names).items() if count > 1
    ]
    if duplicate_names:
        raise _error(
            f"{source}.tools",
            f"duplicate tool names: {', '.join(sorted(duplicate_names))}",
        )
    available_tools = set(tool_names)
    call_ids: list[str] = []
    result_ids: list[str] = []

    for index, message in enumerate(messages):
        if message.role == "assistant" and message.tool_calls:
            call = message.tool_calls[0]
            call_ids.append(call.id)
            if call.function.name not in available_tools:
                raise _error(
                    f"{source}.messages[{index}].tool_calls[0].function.name",
                    f"unknown tool {call.function.name!r}",
                )
            if index + 1 >= len(messages) or messages[index + 1].role != "tool":
                raise _error(
                    f"{source}.messages[{index}]",
                    "assistant tool call must be followed immediately by tool result",
                )
            if messages[index + 1].tool_call_id != call.id:
                raise _error(
                    f"{source}.messages[{index + 1}].tool_call_id",
                    f"expected {call.id!r} from preceding assistant tool call",
                )
        elif message.role == "tool":
            result_ids.append(message.tool_call_id or "")
            if index == 0 or not messages[index - 1].tool_calls:
                raise _error(
                    f"{source}.messages[{index}]",
                    "tool result must follow an assistant tool call",
                )

    duplicate_call_ids = [
        call_id for call_id, count in Counter(call_ids).items() if count > 1
    ]
    if duplicate_call_ids:
        raise _error(
            f"{source}.messages",
            f"duplicate tool call IDs: {', '.join(sorted(duplicate_call_ids))}",
        )
    if Counter(call_ids) != Counter(result_ids):
        raise _error(f"{source}.messages", "tool calls and tool results are not paired")

    allowed_transitions = {
        "system": {"user"},
        "user": {"assistant"},
        "tool": {"assistant"},
    }
    for index, (left, right) in enumerate(zip(messages, messages[1:])):
        if left.role == "assistant":
            expected = {"tool"} if left.tool_calls else {"user"}
        else:
            expected = allowed_transitions[left.role]
        if right.role not in expected:
            raise _error(
                f"{source}.messages[{index + 1}]",
                f"invalid role transition {left.role!r} -> {right.role!r}",
            )


def parse_sft_example(value: Any, *, source: str = "<memory>") -> SFTExample:
    """Parse and validate one JSON-compatible SFT row."""

    row = _require_mapping(value, source)
    _require_exact_keys(
        row,
        required=TOP_LEVEL_KEYS,
        optional=set(),
        location=source,
    )
    trajectory_id = _require_non_empty_string(
        row["trajectory_id"], f"{source}.trajectory_id"
    )
    task_id = _require_non_empty_string(row["task_id"], f"{source}.task_id")
    raw_messages = _require_list(row["messages"], f"{source}.messages")
    raw_tools = _require_list(row["tools"], f"{source}.tools")
    if not raw_tools:
        raise _error(f"{source}.tools", "at least one tool definition is required")
    metadata = _require_mapping(row["metadata"], f"{source}.metadata")
    metadata_dict = deepcopy(dict(metadata))
    _validate_json_value(metadata_dict, f"{source}.metadata")

    messages = tuple(
        _parse_message(message, index, source)
        for index, message in enumerate(raw_messages)
    )
    tools = tuple(
        _parse_tool_definition(tool, index, source)
        for index, tool in enumerate(raw_tools)
    )
    _validate_conversation(messages, tools, source)
    return SFTExample(
        trajectory_id=trajectory_id,
        task_id=task_id,
        messages=messages,
        tools=tools,
        metadata=metadata_dict,
        source=source,
    )


def iter_sft_examples(path: str | Path) -> Iterator[SFTExample]:
    """Stream validated examples from JSONL without loading the whole file."""

    input_path = Path(path)
    with input_path.open(encoding="utf-8") as input_file:
        found_example = False
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            found_example = True
            source = f"{input_path}:{line_number}"
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise _error(source, f"invalid JSON: {error.msg}") from error
            yield parse_sft_example(value, source=source)
        if not found_example:
            raise _error(str(input_path), "JSONL file contains no examples")
