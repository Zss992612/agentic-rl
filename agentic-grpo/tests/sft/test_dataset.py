from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from agentic_rl.sft.dataset import SFTDatasetError, TokenizedSFTDataset

TRAINING_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    TRAINING_ROOT
    / "data/tokenized/retail_serial_v3_qwen3_4b_instruct_2507"
)


@pytest.fixture(scope="module")
def dataset() -> TokenizedSFTDataset:
    return TokenizedSFTDataset(
        DATA_PATH,
        expected_max_seq_length=17_408,
        verify_hashes=True,
    )


def test_dataset_contract(dataset: TokenizedSFTDataset) -> None:
    assert len(dataset) == 256
    assert dataset.total_tokens == 2_637_872
    assert dataset.total_supervised_tokens == 414_294
    assert dataset.max_seq_length == 17_408
    assert dataset.lengths.dtype == np.int64
    assert int(dataset.lengths.min()) == 6_145
    assert int(dataset.lengths.max()) == 16_904
    assert not dataset.lengths.flags.writeable


def test_first_sample_is_unpadded_numpy_view(dataset: TokenizedSFTDataset) -> None:
    sample = dataset[0]

    assert sample.dataset_index == 0
    assert sample.trajectory_id == "7befd7d4-2a48-4dda-891f-c072fb008517"
    assert sample.task_id == "0"
    assert sample.token_count == 10_185
    assert sample.supervised_token_count == 1_290
    assert sample.input_ids.shape == (10_185,)
    assert sample.labels.shape == (10_185,)
    assert sample.input_ids.dtype == np.int32
    assert sample.labels.dtype == np.int32
    assert not sample.input_ids.flags.writeable
    assert not sample.labels.flags.writeable
    assert np.all((sample.labels == -100) | (sample.labels == sample.input_ids))


def test_negative_index_returns_last_trajectory(dataset: TokenizedSFTDataset) -> None:
    sample = dataset[-1]
    info = dataset.trajectory_info(255)

    assert sample.dataset_index == 255
    assert sample.trajectory_id == info.trajectory_id
    assert sample.token_count == int(dataset.lengths[-1])


def test_dataset_pickling_drops_and_reopens_memmaps(
    dataset: TokenizedSFTDataset,
) -> None:
    dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))

    assert restored._input_ids is None
    assert restored._labels is None
    assert restored[0].input_ids.shape == (10_185,)
    assert restored._input_ids is not None
    assert restored._labels is not None


def test_rejects_wrong_max_sequence_length() -> None:
    with pytest.raises(SFTDatasetError, match="does not match dataset"):
        TokenizedSFTDataset(
            DATA_PATH,
            expected_max_seq_length=16_384,
            verify_hashes=False,
        )


def test_rejects_invalid_indices(dataset: TokenizedSFTDataset) -> None:
    with pytest.raises(IndexError):
        _ = dataset[256]
    with pytest.raises(IndexError):
        _ = dataset[-257]
    with pytest.raises(TypeError, match="bool"):
        _ = dataset[True]
