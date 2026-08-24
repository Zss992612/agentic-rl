"""Tokenize and round-trip audit rendered Qwen trajectories.

Checkpoint three deliberately stops before label construction.  Its contract is
that the exact text accepted at checkpoint two maps deterministically to token
IDs and decodes back without any textual change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from agentic_rl.preprocessing.rendering import RenderedTrajectory, render_sft_example
from agentic_rl.preprocessing.schema import SFTExample


class TokenizationError(ValueError):
    """Raised when the Qwen encode/decode round trip is not lossless."""


@dataclass(frozen=True)
class TokenizedTrajectory:
    """Checkpoint-three result; labels are intentionally not represented."""

    rendered: RenderedTrajectory
    input_ids: tuple[int, ...]
    decoded_text: str
    direct_chat_template_match: bool

    @property
    def token_count(self) -> int:
        return len(self.input_ids)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.rendered.text.encode("utf-8")).hexdigest()

    @property
    def decoded_sha256(self) -> str:
        return hashlib.sha256(self.decoded_text.encode("utf-8")).hexdigest()

    @property
    def round_trip_exact(self) -> bool:
        return self.decoded_text == self.rendered.text


def _first_text_difference(left: str, right: str) -> int | None:
    for index, (left_char, right_char) in enumerate(zip(left, right, strict=False)):
        if left_char != right_char:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _difference_context(left: str, right: str, index: int, radius: int = 80) -> str:
    start = max(0, index - radius)
    end = index + radius
    return (
        f"first difference at character {index}; "
        f"original={left[start:end]!r}; decoded={right[start:end]!r}"
    )


def tokenize_rendered_trajectory(
    rendered: RenderedTrajectory,
    tokenizer: Any,
) -> TokenizedTrajectory:
    """Encode a rendered trajectory without padding, truncation, or new tokens."""

    input_ids = tokenizer.encode(rendered.text, add_special_tokens=False)
    if not isinstance(input_ids, list) or not input_ids:
        raise TokenizationError("tokenizer returned no token IDs")
    if not all(isinstance(token_id, int) for token_id in input_ids):
        raise TokenizationError("tokenizer returned non-integer token IDs")

    decoded_text = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    difference = _first_text_difference(rendered.text, decoded_text)
    if difference is not None:
        raise TokenizationError(
            "Qwen encode/decode round trip changed the rendered trajectory: "
            + _difference_context(rendered.text, decoded_text, difference)
        )

    reencoded_ids = tokenizer.encode(decoded_text, add_special_tokens=False)
    if reencoded_ids != input_ids:
        raise TokenizationError("decoded text does not re-encode to the same token IDs")

    return TokenizedTrajectory(
        rendered=rendered,
        input_ids=tuple(input_ids),
        decoded_text=decoded_text,
        direct_chat_template_match=False,
    )


def tokenize_sft_example(example: SFTExample, tokenizer: Any) -> TokenizedTrajectory:
    """Render then tokenize one example and cross-check the direct template path."""

    rendered = render_sft_example(example, tokenizer)
    tokenized = tokenize_rendered_trajectory(rendered, tokenizer)
    direct_ids = tokenizer.apply_chat_template(
        example.chat_messages(),
        tools=example.chat_tools(),
        tokenize=True,
        add_generation_prompt=False,
    )
    if list(tokenized.input_ids) != list(direct_ids):
        raise TokenizationError(
            "tokenizing rendered text differs from apply_chat_template(tokenize=True)"
        )
    return TokenizedTrajectory(
        rendered=tokenized.rendered,
        input_ids=tokenized.input_ids,
        decoded_text=tokenized.decoded_text,
        direct_chat_template_match=True,
    )
