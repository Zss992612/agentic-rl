from __future__ import annotations

import pytest

from agentic_rl.sft.sampler import (
    BatchSamplerError,
    LengthGroupedBatchSampler,
)


def test_length_grouping_covers_every_index_once() -> None:
    lengths = [103, 10, 101, 12, 100, 11, 102, 13]
    sampler = LengthGroupedBatchSampler(
        lengths,
        batch_size=2,
        bucket_size=4,
        shuffle=True,
        seed=42,
    )

    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert len(sampler) == 4
    assert sorted(flattened) == list(range(len(lengths)))
    assert all(len(batch) == 2 for batch in batches)
    assert all(
        max(lengths[index] for index in batch)
        - min(lengths[index] for index in batch)
        <= 3
        for batch in batches
    )


def test_shuffle_is_reproducible_per_epoch_and_changes_across_epochs() -> None:
    sampler = LengthGroupedBatchSampler(
        list(range(1, 33)),
        batch_size=4,
        bucket_size=8,
        shuffle=True,
        seed=7,
    )

    epoch_zero_first = list(sampler)
    epoch_zero_second = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert epoch_zero_first == epoch_zero_second
    assert epoch_one != epoch_zero_first
    assert sorted(sum(epoch_one, [])) == list(range(32))


def test_no_shuffle_returns_length_order() -> None:
    lengths = [30, 10, 20, 40]
    sampler = LengthGroupedBatchSampler(
        lengths,
        batch_size=2,
        bucket_size=4,
        shuffle=False,
    )

    assert list(sampler) == [[1, 2], [0, 3]]


def test_drop_last_only_drops_global_incomplete_batch() -> None:
    sampler = LengthGroupedBatchSampler(
        [1, 2, 3, 4, 5],
        batch_size=2,
        bucket_size=3,
        shuffle=False,
        drop_last=True,
    )

    assert len(sampler) == 2
    assert list(sampler) == [[0, 1], [2, 3]]


def test_invalid_sampler_contract_is_rejected() -> None:
    with pytest.raises(BatchSamplerError, match="bucket_size"):
        LengthGroupedBatchSampler([1, 2], batch_size=2, bucket_size=1)
    with pytest.raises(BatchSamplerError, match=r"lengths\[1\]"):
        LengthGroupedBatchSampler([1, 0], batch_size=1, bucket_size=1)
    with pytest.raises(BatchSamplerError, match="batch_size"):
        LengthGroupedBatchSampler([1, 2], batch_size="2", bucket_size=2)  # type: ignore[arg-type]
    with pytest.raises(BatchSamplerError, match="shuffle"):
        LengthGroupedBatchSampler(
            [1, 2], batch_size=1, bucket_size=2, shuffle=1  # type: ignore[arg-type]
        )
