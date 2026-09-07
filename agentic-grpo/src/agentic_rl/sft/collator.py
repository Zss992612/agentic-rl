"""Dynamic right-padding for tokenized assistant-only SFT samples."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import torch

from agentic_rl.sft.dataset import SFTSample


class CollationError(ValueError):
    """Raised when samples cannot form a valid training batch."""


class DynamicPaddingCollator:
    """Convert unpadded NumPy samples into padded PyTorch tensors.

    ``input_ids`` are padded with the tokenizer's pad token, ``labels`` with
    ``-100`` so cross entropy ignores them, and ``attention_mask`` with zero.
    The target width is the longest trajectory in this batch, optionally
    rounded up to a hardware-friendly multiple without crossing
    ``max_seq_length``.
    """

    def __init__(
        self,
        *,
        pad_token_id: int,
        label_pad_token_id: int = -100,
        max_seq_length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.pad_token_id = self._integer(pad_token_id, "pad_token_id")
        if self.pad_token_id < 0:
            raise CollationError("pad_token_id must be non-negative")
        self.label_pad_token_id = self._integer(
            label_pad_token_id, "label_pad_token_id"
        )
        self.max_seq_length = (
            None
            if max_seq_length is None
            else self._positive_integer(max_seq_length, "max_seq_length")
        )
        self.pad_to_multiple_of = (
            None
            if pad_to_multiple_of is None
            else self._positive_integer(
                pad_to_multiple_of, "pad_to_multiple_of"
            )
        )

    @staticmethod
    def _integer(value: int, name: str) -> int:
        if isinstance(value, bool):
            raise CollationError(f"{name} must be an integer, not bool")
        try:
            return operator.index(value)
        except TypeError as error:
            raise CollationError(f"{name} must be an integer") from error

    @classmethod
    def _positive_integer(cls, value: int, name: str) -> int:
        normalized = cls._integer(value, name)
        if normalized <= 0:
            raise CollationError(f"{name} must be a positive integer")
        return normalized

    def _target_length(self, longest: int) -> int:
        if self.max_seq_length is not None and longest > self.max_seq_length:
            raise CollationError(
                f"sample length {longest} exceeds max_seq_length "
                f"{self.max_seq_length}; collator does not truncate trajectories"
            )
        target = longest
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            target = ((longest + multiple - 1) // multiple) * multiple
            if self.max_seq_length is not None:
                target = min(target, self.max_seq_length)
        return target

    def __call__(self, samples: Sequence[SFTSample]) -> dict[str, torch.Tensor]:
        if not samples:
            raise CollationError("cannot collate an empty batch")

        lengths: list[int] = []
        for row, sample in enumerate(samples):
            if not isinstance(sample, SFTSample):
                raise CollationError(
                    f"samples[{row}] must be an SFTSample, got "
                    f"{type(sample).__name__}"
                )
            if sample.input_ids.ndim != 1 or sample.labels.ndim != 1:
                raise CollationError(f"samples[{row}] arrays must be one-dimensional")
            if sample.input_ids.shape != sample.labels.shape:
                raise CollationError(f"samples[{row}] input and label shapes differ")
            if len(sample.input_ids) != sample.token_count:
                raise CollationError(f"samples[{row}] token_count is inconsistent")
            if sample.token_count <= 0:
                raise CollationError(f"samples[{row}] is empty")
            lengths.append(sample.token_count)

        target_length = self._target_length(max(lengths))
        batch_size = len(samples)
        input_ids = torch.full(
            (batch_size, target_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, target_length),
            self.label_pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, target_length),
            dtype=torch.bool,
        )

        for row, (sample, length) in enumerate(zip(samples, lengths, strict=True)):
            # torch.tensor intentionally copies the read-only mmap slices and
            # converts persisted int32 values to the int64 embedding dtype.
            input_ids[row, :length] = torch.tensor(
                sample.input_ids,
                dtype=torch.long,
            )
            labels[row, :length] = torch.tensor(
                sample.labels,
                dtype=torch.long,
            )
            attention_mask[row, :length] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
