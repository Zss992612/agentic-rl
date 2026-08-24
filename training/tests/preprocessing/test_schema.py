import json
from pathlib import Path

import pytest

from agentic_rl.preprocessing.schema import (
    SchemaValidationError,
    iter_sft_examples,
    parse_sft_example,
)

TRAINING_DIR = Path(__file__).resolve().parents[2]
TRAIN_DATA = TRAINING_DIR / "data/sft/retail_serial_v3/sft_train_256.jsonl"


def valid_row():
    return {
        "trajectory_id": "trajectory-1",
        "task_id": "24",
        "messages": [
            {"role": "system", "content": "Retail policy"},
            {"role": "user", "content": "Check order #W123."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_order_details",
                            "arguments": {"order_id": "#W123"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"status":"delivered"}',
                "tool_call_id": "call-1",
            },
            {"role": "assistant", "content": "The order was delivered."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_order_details",
                    "description": "Get order details.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            }
        ],
        "metadata": {"tier": "medium"},
    }


def test_parse_round_trips_canonical_chat_fields():
    row = valid_row()
    example = parse_sft_example(row)

    assert example.trajectory_id == "trajectory-1"
    assert example.task_id == "24"
    assert example.chat_messages() == row["messages"]
    assert example.chat_tools() == row["tools"]
    assert example.messages[2].tool_calls[0].function.arguments == {
        "order_id": "#W123"
    }


def test_rejects_parallel_tool_calls():
    row = valid_row()
    row["messages"][2]["tool_calls"].append(
        {
            "id": "call-2",
            "type": "function",
            "function": {
                "name": "get_order_details",
                "arguments": {"order_id": "#W456"},
            },
        }
    )

    with pytest.raises(SchemaValidationError, match="at most one tool call"):
        parse_sft_example(row)


def test_rejects_mismatched_tool_result_id():
    row = valid_row()
    row["messages"][3]["tool_call_id"] = "wrong-id"

    with pytest.raises(SchemaValidationError, match="expected 'call-1'"):
        parse_sft_example(row)


def test_rejects_unknown_tool_name():
    row = valid_row()
    row["messages"][2]["tool_calls"][0]["function"]["name"] = "missing_tool"

    with pytest.raises(SchemaValidationError, match="unknown tool 'missing_tool'"):
        parse_sft_example(row)


def test_jsonl_error_reports_line_number(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(valid_row()) + "\nnot-json\n", encoding="utf-8")
    iterator = iter_sft_examples(path)

    assert next(iterator).trajectory_id == "trajectory-1"
    with pytest.raises(SchemaValidationError, match=r"data\.jsonl:2"):
        next(iterator)


def test_all_256_training_rows_satisfy_the_schema():
    examples = list(iter_sft_examples(TRAIN_DATA))

    assert len(examples) == 256
    assert len({example.trajectory_id for example in examples}) == 256
    assert len({example.task_id for example in examples}) == 64
    assert all(len(example.tools) == 16 for example in examples)
    assert all(example.messages[0].role == "system" for example in examples)
    assert all(example.messages[-1].role == "assistant" for example in examples)
