"""Construct assistant-only causal-LM labels for Qwen trajectories.

Checkpoint four operates on one already audited token sequence.  It keeps
assistant payloads (including tool calls and ``<|im_end|>``) as targets while
masking all headers and external context with ``IGNORE_INDEX``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from agentic_rl.preprocessing.rendering import QWEN_IM_END, QWEN_IM_START
from agentic_rl.preprocessing.schema import SFTExample
from agentic_rl.preprocessing.tokenization import TokenizedTrajectory

IGNORE_INDEX = -100
AssistantKind = Literal["text", "tool_call"]


class MaskingError(ValueError):
    """Raised when assistant supervision boundaries cannot be proven correct."""


@dataclass(frozen=True)
class AssistantSpan:
    """Token boundaries for one assistant message.

    ``header_start:target_start`` is masked. ``target_start:target_end`` is
    supervised, and includes the final ``<|im_end|>`` token.
    """

    ordinal: int
    kind: AssistantKind
    header_start: int
    target_start: int
    target_end: int
    masked_leading_character_count: int
    masked_leading_text: str

    @property
    def supervised_token_count(self) -> int:
        return self.target_end - self.target_start


@dataclass(frozen=True)
class MaskedTrajectory:
    """One checkpoint-four trajectory with assistant-only labels."""

    tokenized: TokenizedTrajectory
    labels: tuple[int, ...]
    assistant_spans: tuple[AssistantSpan, ...]
    ignore_index: int = IGNORE_INDEX

    @property
    def supervised_token_count(self) -> int:
        return sum(label != self.ignore_index for label in self.labels)

    @property
    def masked_token_count(self) -> int:
        return len(self.labels) - self.supervised_token_count

    @property
    def supervised_fraction(self) -> float:
        return self.supervised_token_count / len(self.labels)


def _assistant_character_spans(text: str) -> list[tuple[int, int, int]]:
    """Return ``(header_start, content_start, target_end)`` character spans."""

    header = f"{QWEN_IM_START}assistant\n"
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    while True:
        header_start = text.find(header, cursor)
        if header_start < 0:
            return spans
        content_start = header_start + len(header)
        im_end_start = text.find(QWEN_IM_END, content_start)
        if im_end_start < 0:
            raise MaskingError(
                f"assistant message at character {header_start} has no <|im_end|>"
            )
        target_end = im_end_start + len(QWEN_IM_END)
        spans.append((header_start, content_start, target_end))
        cursor = target_end


def _first_token_starting_at(
    offsets: list[tuple[int, int]],
    character_index: int,
) -> int:
    for token_index, (start, _end) in enumerate(offsets):
        if start == character_index:
            return token_index
    raise MaskingError(f"no token starts at rendered character {character_index}")


def build_assistant_labels(
    example: SFTExample,
    tokenized: TokenizedTrajectory,
    tokenizer: Any,
) -> MaskedTrajectory:
    """Build labels without truncation, padding, or manual causal shifting."""

    input_ids = tokenized.input_ids
    im_end_id = tokenizer.convert_tokens_to_ids(QWEN_IM_END)
    if not isinstance(im_end_id, int):
        raise MaskingError("target tokenizer does not expose Qwen role delimiters")

    encoded_with_offsets = tokenizer(
        tokenized.rendered.text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    offset_input_ids = tuple(encoded_with_offsets["input_ids"])
    offsets = [tuple(offset) for offset in encoded_with_offsets["offset_mapping"]]
    if offset_input_ids != input_ids:
        raise MaskingError("offset-mapped tokenization differs from checkpoint three")
    if len(offsets) != len(input_ids):
        raise MaskingError("token offset count does not match input IDs")

    character_spans = _assistant_character_spans(tokenized.rendered.text)
    assistant_messages = [
        message for message in example.messages if message.role == "assistant"
    ]
    if len(character_spans) != len(assistant_messages):
        raise MaskingError(
            "assistant header count does not match validated source messages: "
            f"{len(character_spans)} != {len(assistant_messages)}"
        )

    labels = [IGNORE_INDEX] * len(input_ids)
    spans: list[AssistantSpan] = []
    previous_target_end = 0
    for ordinal, (character_span, message) in enumerate(
        zip(character_spans, assistant_messages, strict=True)
    ):
        header_char_start, content_char_start, target_char_end = character_span
        header_start = _first_token_starting_at(offsets, header_char_start)
        target_indices = [
            token_index
            for token_index, (start, end) in enumerate(offsets)
            if end > start
            and start >= content_char_start
            and end <= target_char_end
        ]
        if not target_indices:
            raise MaskingError(f"assistant message {ordinal} has no target tokens")
        if target_indices != list(range(target_indices[0], target_indices[-1] + 1)):
            raise MaskingError(f"assistant message {ordinal} target is not contiguous")
        target_start = target_indices[0]
        target_end = target_indices[-1] + 1
        im_end_index = target_end - 1
        if header_start < previous_target_end:
            raise MaskingError("assistant supervision spans overlap")
        if input_ids[im_end_index] != im_end_id:
            raise MaskingError(f"assistant message {ordinal} target does not end correctly")

        overlapping_prefix_tokens = [
            (start, end)
            for start, end in offsets[header_start:target_start]
            if start < content_char_start < end
        ]
        masked_leading_end = max(
            (end for _start, end in overlapping_prefix_tokens),
            default=content_char_start,
        )
        masked_leading_text = tokenized.rendered.text[
            content_char_start:masked_leading_end
        ]
        masked_leading_characters = len(masked_leading_text)

        labels[target_start:target_end] = input_ids[target_start:target_end]
        kind: AssistantKind = "tool_call" if message.tool_calls else "text"
        decoded_target = tokenizer.decode(
            input_ids[target_start:im_end_index],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        is_rendered_tool_call = decoded_target.startswith("<tool_call>\n")
        if (kind == "tool_call") != is_rendered_tool_call:
            raise MaskingError(
                f"assistant message {ordinal} source/rendered kind mismatch"
            )
        spans.append(
            AssistantSpan(
                ordinal=ordinal,
                kind=kind,
                header_start=header_start,
                target_start=target_start,
                target_end=target_end,
                masked_leading_character_count=masked_leading_characters,
                masked_leading_text=masked_leading_text,
            )
        )
        previous_target_end = target_end

    for index, (input_id, label) in enumerate(zip(input_ids, labels, strict=True)):
        if label not in (IGNORE_INDEX, input_id):
            raise MaskingError(f"invalid label at token {index}: {label}")
    if not any(label != IGNORE_INDEX for label in labels):
        raise MaskingError("trajectory contains no supervised assistant tokens")

    return MaskedTrajectory(
        tokenized=tokenized,
        labels=tuple(labels),
        assistant_spans=tuple(spans),
    )
