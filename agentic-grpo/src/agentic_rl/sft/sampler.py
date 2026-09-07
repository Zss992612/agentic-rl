"""Length-aware batch construction for SFT trajectories.

The sampler only decides which dataset indices share a batch.  It never reads
or changes token data; dynamic padding remains the collator's responsibility.
"""

from __future__ import annotations

import math
import operator
import random
from collections.abc import Iterator, Sequence


class BatchSamplerError(ValueError):
    """Raised when a batch-sampling contract is invalid."""


class LengthGroupedBatchSampler:
    """Yield batches whose examples have similar sequence lengths.

    Indices are first sorted by length and divided into local buckets.  On
    every epoch, samples are shuffled inside each bucket and all resulting
    batches are shuffled globally.  This keeps batches length-homogeneous
    without fixing training order from shortest to longest.

    The class deliberately has no PyTorch dependency.  Its ``__iter__`` and
    ``__len__`` methods satisfy the batch-sampler protocol expected by a
    PyTorch DataLoader.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        bucket_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        normalized_batch_size = self._positive_integer(batch_size, "batch_size")
        normalized_bucket_size = self._positive_integer(bucket_size, "bucket_size")
        if normalized_bucket_size < normalized_batch_size:
            raise BatchSamplerError("bucket_size must be >= batch_size")
        if not isinstance(shuffle, bool):
            raise BatchSamplerError("shuffle must be a boolean")
        if not isinstance(drop_last, bool):
            raise BatchSamplerError("drop_last must be a boolean")
        normalized_seed = self._integer(seed, "seed")

        normalized_lengths: list[int] = []
        for index, raw_length in enumerate(lengths):
            try:
                length = operator.index(raw_length)
            except TypeError as error:
                raise BatchSamplerError(
                    f"lengths[{index}] must be an integer"
                ) from error
            if length <= 0:
                raise BatchSamplerError(
                    f"lengths[{index}] must be a positive integer"
                )
            normalized_lengths.append(length)

        self.lengths = tuple(normalized_lengths)
        self.batch_size = normalized_batch_size
        self.bucket_size = normalized_bucket_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = normalized_seed
        self.epoch = 0

    @staticmethod
    def _integer(value: int, name: str) -> int:
        if isinstance(value, bool):
            raise BatchSamplerError(f"{name} must be an integer, not bool")
        try:
            return operator.index(value)
        except TypeError as error:
            raise BatchSamplerError(f"{name} must be an integer") from error

    @classmethod
    def _positive_integer(cls, value: int, name: str) -> int:
        normalized = cls._integer(value, name)
        if normalized <= 0:
            raise BatchSamplerError(f"{name} must be a positive integer")
        return normalized

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle stream for one training epoch."""

        normalized = self._integer(epoch, "epoch")
        if normalized < 0:
            raise BatchSamplerError("epoch must be a non-negative integer")
        self.epoch = normalized

    def _ordered_indices(self, rng: random.Random) -> list[int]:
        if not self.lengths:
            return []
        sorted_indices = sorted(
            range(len(self.lengths)),
            key=self.lengths.__getitem__,
        )
        buckets = [
            sorted_indices[start : start + self.bucket_size]
            for start in range(0, len(sorted_indices), self.bucket_size)
        ]
        if self.shuffle:
            for bucket in buckets:
                rng.shuffle(bucket)
        return [index for bucket in buckets for index in bucket]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        ordered = self._ordered_indices(rng)
        batches = [
            ordered[start : start + self.batch_size]
            for start in range(0, len(ordered), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)
