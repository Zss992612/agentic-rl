from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from agentic_rl.sft.collator import CollationError, DynamicPaddingCollator
from agentic_rl.sft.dataset import SFTSample, TokenizedSFTDataset

TRAINING_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    TRAINING_ROOT
    / "data/tokenized/retail_serial_v3_qwen3_4b_instruct_2507"
)


def sample(index: int, input_ids: list[int], labels: list[int]) -> SFTSample:
    input_array = np.asarray(input_ids, dtype=np.int32)
    label_array = np.asarray(labels, dtype=np.int32)
    return SFTSample(
        dataset_index=index,
        trajectory_id=f"trajectory-{index}",
        task_id=str(index),
        input_ids=input_array,
        labels=label_array,
        token_count=len(input_ids),
        supervised_token_count=sum(label != -100 for label in labels),
    )


def test_dynamic_right_padding_contract() -> None:
    collator = DynamicPaddingCollator(pad_token_id=151_643)

    batch = collator(
        [
            sample(0, [10, 11, 12], [-100, 11, 12]),
            sample(1, [20, 21, 22, 23, 24], [-100, -100, 22, 23, 24]),
        ]
    )

    assert set(batch) == {"input_ids", "labels", "attention_mask"}
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["input_ids"].tolist() == [
        [10, 11, 12, 151_643, 151_643],
        [20, 21, 22, 23, 24],
    ]
    assert batch["labels"].tolist() == [
        [-100, 11, 12, -100, -100],
        [-100, -100, 22, 23, 24],
    ]
    assert batch["attention_mask"].tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]


def test_padding_can_round_to_multiple_without_crossing_maximum() -> None:
    collator = DynamicPaddingCollator(
        pad_token_id=0,
        max_seq_length=10,
        pad_to_multiple_of=8,
    )

    rounded = collator([sample(0, [1, 2, 3, 4, 5], [-100] * 5)])
    capped = collator([sample(1, list(range(9)), [-100] * 9)])

    assert rounded["input_ids"].shape == (1, 8)
    assert capped["input_ids"].shape == (1, 10)


def test_real_samples_are_padded_only_to_batch_maximum() -> None:
    dataset = TokenizedSFTDataset(DATA_PATH, verify_hashes=False)
    samples = [dataset[0], dataset[1]]
    collator = DynamicPaddingCollator(
        pad_token_id=151_643,
        max_seq_length=17_408,
    )

    batch = collator(samples)
    expected_width = max(sample.token_count for sample in samples)

    assert batch["input_ids"].shape == (2, expected_width)
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["attention_mask"].sum().item() == sum(
        sample.token_count for sample in samples
    )
    assert (batch["labels"] != -100).sum().item() == sum(
        sample.supervised_token_count for sample in samples
    )


def test_collator_refuses_to_truncate_or_accept_invalid_batches() -> None:
    collator = DynamicPaddingCollator(pad_token_id=0, max_seq_length=2)

    with pytest.raises(CollationError, match="empty batch"):
        collator([])
    with pytest.raises(CollationError, match="does not truncate"):
        collator([sample(0, [1, 2, 3], [-100, 2, 3])])

