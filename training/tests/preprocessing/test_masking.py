from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rl.preprocessing.masking import IGNORE_INDEX, build_assistant_labels
from agentic_rl.preprocessing.schema import iter_sft_examples
from agentic_rl.preprocessing.tokenization import tokenize_sft_example

TRAINING_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = TRAINING_DIR / "data/sft/retail_serial_v3/sft_train_256.jsonl"
TOKENIZER_PATH = TRAINING_DIR / "artifacts/tokenizers/Qwen3-4B-Instruct-2507"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    if not TOKENIZER_PATH.is_dir():
        pytest.skip("local Qwen tokenizer artifact is unavailable")
    return transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        use_fast=True,
        local_files_only=True,
    )


@pytest.fixture(scope="module")
def first_masked(tokenizer):
    example = next(iter_sft_examples(DATA_PATH))
    tokenized = tokenize_sft_example(example, tokenizer)
    return build_assistant_labels(example, tokenized, tokenizer)


def test_label_shape_and_values(first_masked) -> None:
    input_ids = first_masked.tokenized.input_ids
    labels = first_masked.labels

    assert len(labels) == len(input_ids)
    assert all(label in (IGNORE_INDEX, input_id) for input_id, label in zip(input_ids, labels, strict=True))
    assert first_masked.supervised_token_count > 0
    assert first_masked.masked_token_count > 0


def test_assistant_spans_match_source_messages(first_masked) -> None:
    assert len(first_masked.assistant_spans) == 12
    assert sum(span.kind == "tool_call" for span in first_masked.assistant_spans) == 7
    assert sum(span.kind == "text" for span in first_masked.assistant_spans) == 5
    assert sum(
        span.masked_leading_character_count > 0
        for span in first_masked.assistant_spans
    ) == 1
    assert sum(
        span.masked_leading_character_count
        for span in first_masked.assistant_spans
    ) == 2
    assert all(
        not span.masked_leading_text or span.masked_leading_text.isspace()
        for span in first_masked.assistant_spans
    )


def test_headers_are_masked_and_targets_are_supervised(first_masked) -> None:
    input_ids = first_masked.tokenized.input_ids
    labels = first_masked.labels

    for span in first_masked.assistant_spans:
        assert all(label == IGNORE_INDEX for label in labels[span.header_start : span.target_start])
        assert labels[span.target_start : span.target_end] == input_ids[span.target_start : span.target_end]
        assert labels[span.target_end] == IGNORE_INDEX


def test_each_assistant_end_token_is_supervised(first_masked, tokenizer) -> None:
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for span in first_masked.assistant_spans:
        assert first_masked.tokenized.input_ids[span.target_end - 1] == im_end_id
        assert first_masked.labels[span.target_end - 1] == im_end_id


def test_actual_tool_responses_are_fully_masked(first_masked, tokenizer) -> None:
    text = first_masked.tokenized.rendered.text
    labels = first_masked.labels
    offsets = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )["offset_mapping"]
    start_tag = "<tool_response>"
    end_tag = "</tool_response>"
    starts: list[int] = []
    cursor = 0
    while True:
        start = text.find(start_tag, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + len(start_tag)
    assert len(starts) == 7
    for start in starts:
        end = text.index(end_tag, start) + len(end_tag)
        response_token_indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        assert all(
            labels[index] == IGNORE_INDEX for index in response_token_indices
        )
