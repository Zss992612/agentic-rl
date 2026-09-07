from pathlib import Path

import pytest

from agentic_rl.preprocessing.rendering import render_sft_example
from agentic_rl.preprocessing.schema import iter_sft_examples

TRAINING_DIR = Path(__file__).resolve().parents[2]
TRAIN_DATA = TRAINING_DIR / "data/sft/retail_serial_v3/sft_train_256.jsonl"
TOKENIZER_DIR = TRAINING_DIR / "artifacts/tokenizers/Qwen3-4B-Instruct-2507"


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_DIR.is_dir():
        pytest.skip("local Qwen tokenizer artifact has not been downloaded")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        use_fast=True,
        local_files_only=True,
    )


def test_real_qwen_template_renders_first_training_trajectory(tokenizer):
    example = next(iter_sft_examples(TRAIN_DATA))
    rendered = render_sft_example(example, tokenizer)

    assert rendered.text.startswith("<|im_start|>system\n")
    assert rendered.text.endswith("<|im_end|>\n")
    assert "<tools>" in rendered.text
    assert "</tools>" in rendered.text
    conversation_text = rendered.text.split("<|im_end|>\n", maxsplit=1)[1]
    assert conversation_text.count("<tool_call>") == rendered.assistant_tool_call_count
    assert conversation_text.count("<tool_response>") == rendered.tool_result_count
    assert rendered.tool_definition_count == 16
    assert rendered.message_count == len(example.messages)


def test_real_qwen_template_does_not_render_transport_ids(tokenizer):
    example = next(iter_sft_examples(TRAIN_DATA))
    rendered = render_sft_example(example, tokenizer)
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

    assert call_ids
    assert call_ids == result_ids
    assert all(call_id not in rendered.text for call_id in call_ids)


def test_real_qwen_template_contains_system_tools_and_observations(tokenizer):
    example = next(iter_sft_examples(TRAIN_DATA))
    rendered = render_sft_example(example, tokenizer)

    assert example.messages[0].content in rendered.text
    assert all(tool.name in rendered.text for tool in example.tools)
    tool_messages = [message for message in example.messages if message.role == "tool"]
    assert tool_messages
    assert all(message.content in rendered.text for message in tool_messages)
