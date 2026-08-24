"""Render validated SFT examples with the official Qwen chat template.

Checkpoint two intentionally stops at human-readable text. It does not request
token IDs, construct labels, truncate sequences, import PyTorch, or write the
final training arrays.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from agentic_rl.preprocessing.schema import SFTExample

QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>"
QWEN_TOOL_CALL_START = "<tool_call>"
QWEN_TOOL_CALL_END = "</tool_call>"
QWEN_TOOL_RESPONSE_START = "<tool_response>"
QWEN_TOOL_RESPONSE_END = "</tool_response>"


class RenderingError(ValueError):
    """Raised when Qwen rendering violates the expected training contract."""


@dataclass(frozen=True)
class RenderedTrajectory:
    """Human-readable Qwen input plus structural audit counts."""

    trajectory_id: str
    task_id: str
    text: str
    chat_template_sha256: str
    message_count: int
    tool_definition_count: int
    assistant_message_count: int
    assistant_tool_call_count: int
    tool_result_count: int

    @property
    def character_count(self) -> int:
        return len(self.text)


def chat_template_sha256(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        raise RenderingError("target tokenizer does not define a chat template")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _count_assistant_tool_calls(example: SFTExample) -> int:
    return sum(len(message.tool_calls) for message in example.messages)


def _validate_rendered_text(
    example: SFTExample,
    text: str,
    *,
    expected_tool_calls: int,
    expected_tool_results: int,
) -> None:
    if not text.startswith(f"{QWEN_IM_START}system\n"):
        raise RenderingError("Qwen rendering must start with the system role")
    if not text.endswith(f"{QWEN_IM_END}\n"):
        raise RenderingError("training rendering must end with a complete message")
    if example.messages[0].content not in text:
        raise RenderingError("system prompt is missing from rendered text")
    if "<tools>" not in text or "</tools>" not in text:
        raise RenderingError("tool definitions are missing from the Qwen system block")

    for tool in example.tools:
        if tool.name not in text:
            raise RenderingError(
                f"tool definition {tool.name!r} is missing from rendered text"
            )

    # The official Qwen system block contains a literal <tool_call> example.
    # Only tags after the first completed system message represent trajectory
    # events and should be compared with the source messages.
    system_end = text.find(f"{QWEN_IM_END}\n")
    if system_end < 0:
        raise RenderingError("rendered Qwen system message has no end marker")
    conversation_text = text[system_end + len(f"{QWEN_IM_END}\n") :]
    tag_counts = {
        QWEN_TOOL_CALL_START: conversation_text.count(QWEN_TOOL_CALL_START),
        QWEN_TOOL_CALL_END: conversation_text.count(QWEN_TOOL_CALL_END),
        QWEN_TOOL_RESPONSE_START: conversation_text.count(QWEN_TOOL_RESPONSE_START),
        QWEN_TOOL_RESPONSE_END: conversation_text.count(QWEN_TOOL_RESPONSE_END),
    }
    if tag_counts[QWEN_TOOL_CALL_START] != expected_tool_calls:
        raise RenderingError(
            "rendered tool-call count does not match assistant tool calls: "
            f"{tag_counts[QWEN_TOOL_CALL_START]} != {expected_tool_calls}"
        )
    if tag_counts[QWEN_TOOL_CALL_END] != expected_tool_calls:
        raise RenderingError("rendered tool-call tags are unbalanced")
    if tag_counts[QWEN_TOOL_RESPONSE_START] != expected_tool_results:
        raise RenderingError(
            "rendered tool-response count does not match tool messages: "
            f"{tag_counts[QWEN_TOOL_RESPONSE_START]} != {expected_tool_results}"
        )
    if tag_counts[QWEN_TOOL_RESPONSE_END] != expected_tool_results:
        raise RenderingError("rendered tool-response tags are unbalanced")

    call_ids = {
        call.id
        for message in example.messages
        for call in message.tool_calls
    }
    result_ids = {
        message.tool_call_id
        for message in example.messages
        if message.tool_call_id is not None
    }
    rendered_ids = sorted(
        call_id for call_id in call_ids | result_ids if call_id in text
    )
    if rendered_ids:
        raise RenderingError(
            "Qwen rendering unexpectedly contains transport-layer tool call IDs: "
            + ", ".join(rendered_ids)
        )


def render_sft_example(example: SFTExample, tokenizer: Any) -> RenderedTrajectory:
    """Render one validated trajectory with Qwen's official template.

    ``add_generation_prompt`` is fixed to ``False`` because the final assistant
    message already exists and is a supervised training target.
    """

    template_hash = chat_template_sha256(tokenizer)
    try:
        text = tokenizer.apply_chat_template(
            example.chat_messages(),
            tools=example.chat_tools(),
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as error:
        raise RenderingError(
            f"failed to render trajectory {example.trajectory_id}: {error}"
        ) from error
    if not isinstance(text, str) or not text:
        raise RenderingError("Qwen chat template did not return non-empty text")

    assistant_messages = sum(
        message.role == "assistant" for message in example.messages
    )
    assistant_tool_calls = _count_assistant_tool_calls(example)
    tool_results = sum(message.role == "tool" for message in example.messages)
    _validate_rendered_text(
        example,
        text,
        expected_tool_calls=assistant_tool_calls,
        expected_tool_results=tool_results,
    )
    return RenderedTrajectory(
        trajectory_id=example.trajectory_id,
        task_id=example.task_id,
        text=text,
        chat_template_sha256=template_hash,
        message_count=len(example.messages),
        tool_definition_count=len(example.tools),
        assistant_message_count=assistant_messages,
        assistant_tool_call_count=assistant_tool_calls,
        tool_result_count=tool_results,
    )
