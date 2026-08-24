from __future__ import annotations

from pathlib import Path

import pytest

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


def test_first_trajectory_round_trips_exactly(tokenizer) -> None:
    example = next(iter_sft_examples(DATA_PATH))
    result = tokenize_sft_example(example, tokenizer)

    assert result.input_ids
    assert result.round_trip_exact
    assert result.direct_chat_template_match
    assert result.text_sha256 == result.decoded_sha256
    assert result.decoded_text == result.rendered.text


def test_checkpoint_three_does_not_construct_labels(tokenizer) -> None:
    example = next(iter_sft_examples(DATA_PATH))
    result = tokenize_sft_example(example, tokenizer)

    assert not hasattr(result, "labels")
