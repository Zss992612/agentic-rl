"""Human-readable audits for tokenized and masked trajectories."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from agentic_rl.preprocessing.masking import IGNORE_INDEX, MaskedTrajectory


@dataclass(frozen=True)
class TokenAuditRow:
    index: int
    input_id: int
    label: int
    loss: bool
    region: str
    token: str


def token_regions(masked: MaskedTrajectory) -> list[str]:
    """Assign an explanatory region to every token for visual inspection."""

    regions = ["external_context"] * len(masked.labels)
    for span in masked.assistant_spans:
        for index in range(span.header_start, span.target_start):
            regions[index] = "assistant_header"
        for index in range(span.target_start, span.target_end - 1):
            regions[index] = f"assistant_{span.kind}"
        regions[span.target_end - 1] = "assistant_end"
    return regions


def iter_token_audit_rows(
    masked: MaskedTrajectory,
    tokenizer: Any,
) -> Iterator[TokenAuditRow]:
    regions = token_regions(masked)
    tokens = tokenizer.convert_ids_to_tokens(list(masked.tokenized.input_ids))
    for index, (input_id, label, region, token) in enumerate(
        zip(
            masked.tokenized.input_ids,
            masked.labels,
            regions,
            tokens,
            strict=True,
        )
    ):
        yield TokenAuditRow(
            index=index,
            input_id=input_id,
            label=label,
            loss=label != IGNORE_INDEX,
            region=region,
            token=token,
        )


def render_span_audit(masked: MaskedTrajectory, tokenizer: Any) -> str:
    """Render complete assistant targets with their exact token boundaries."""

    lines: list[str] = []
    input_ids = masked.tokenized.input_ids
    for span in masked.assistant_spans:
        target = tokenizer.decode(
            input_ids[span.target_start : span.target_end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        lines.extend(
            [
                (
                    f"--- assistant {span.ordinal:02d} | {span.kind} | "
                    f"tokens [{span.target_start}:{span.target_end}) | "
                    f"count={span.supervised_token_count} | "
                    f"masked_leading_chars={span.masked_leading_character_count} ---"
                ),
                target,
                "",
            ]
        )
    return "\n".join(lines)
