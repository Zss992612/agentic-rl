"""Memory-mapped access to the finalized tokenized SFT dataset.

The dataset layer deliberately returns unpadded NumPy views.  PyTorch tensor
conversion, dynamic padding, and attention-mask construction belong to the
collator so this storage contract can be tested without PyTorch or a GPU.
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agentic_rl.preprocessing.storage import (
    FORMAT_NAME,
    FORMAT_VERSION,
    sha256_file,
    validate_dataset_directory,
)


class SFTDatasetError(ValueError):
    """Raised when the persisted training data cannot be safely consumed."""


@dataclass(frozen=True)
class TrajectoryInfo:
    dataset_index: int
    trajectory_id: str
    task_id: str
    token_count: int
    supervised_token_count: int
    masked_token_count: int


@dataclass(frozen=True)
class SFTSample:
    """One unpadded trajectory returned as read-only NumPy views."""

    dataset_index: int
    trajectory_id: str
    task_id: str
    input_ids: NDArray[np.int32]
    labels: NDArray[np.int32]
    token_count: int
    supervised_token_count: int


class TokenizedSFTDataset:
    """Map-style dataset backed by flat `.npy` memory maps.

    The class intentionally does not inherit from ``torch.utils.data.Dataset``:
    PyTorch DataLoader only requires ``__len__`` and ``__getitem__``, while
    avoiding that inheritance keeps local data auditing independent of torch.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        expected_max_seq_length: int | None = None,
        verify_hashes: bool = True,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.is_dir():
            raise SFTDatasetError(
                f"tokenized dataset directory does not exist: {self.directory}"
            )
        try:
            validated = validate_dataset_directory(
                self.directory,
                verify_hashes=verify_hashes,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SFTDatasetError(f"invalid tokenized dataset: {error}") from error

        self.manifest_path = self.directory / "manifest.json"
        self.stats_path = self.directory / "stats.json"
        self.metadata_path = self.directory / "metadata.jsonl"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
        if self.manifest.get("format_name") != FORMAT_NAME:
            raise SFTDatasetError("unexpected tokenized dataset format")
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise SFTDatasetError("unsupported tokenized dataset format version")

        self.max_seq_length = self.manifest.get("sequence_contract", {}).get(
            "max_seq_length"
        )
        if not isinstance(self.max_seq_length, int) or self.max_seq_length <= 0:
            raise SFTDatasetError("manifest has no valid max_seq_length")
        if (
            expected_max_seq_length is not None
            and expected_max_seq_length != self.max_seq_length
        ):
            raise SFTDatasetError(
                "configured max sequence length does not match dataset: "
                f"{expected_max_seq_length} != {self.max_seq_length}"
            )

        self._offsets = np.load(
            self.directory / "offsets.npy",
            allow_pickle=False,
        )
        self._offsets.setflags(write=False)
        self._lengths = np.diff(self._offsets)
        self._lengths.setflags(write=False)
        if int(self._lengths.max()) > self.max_seq_length:
            raise SFTDatasetError(
                "dataset contains a trajectory above its max_seq_length contract"
            )

        self._trajectory_info = self._load_trajectory_info()
        if len(self._trajectory_info) != validated["trajectories"]:
            raise SFTDatasetError("metadata and validated trajectory counts differ")
        if int(self._offsets[-1]) != validated["tokens"]:
            raise SFTDatasetError("offsets and validated token counts differ")
        self.total_tokens = validated["tokens"]
        self.total_supervised_tokens = validated["supervised_tokens"]
        self.manifest_sha256 = sha256_file(self.manifest_path)

        # Open lazily so DataLoader worker pickling never serializes live mmap
        # handles from the parent process.
        self._input_ids: NDArray[np.int32] | None = None
        self._labels: NDArray[np.int32] | None = None

    def _load_trajectory_info(self) -> tuple[TrajectoryInfo, ...]:
        records: list[TrajectoryInfo] = []
        with self.metadata_path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                try:
                    raw: dict[str, Any] = json.loads(line)
                    info = TrajectoryInfo(
                        dataset_index=int(raw["dataset_index"]),
                        trajectory_id=str(raw["trajectory_id"]),
                        task_id=str(raw["task_id"]),
                        token_count=int(raw["token_count"]),
                        supervised_token_count=int(raw["supervised_token_count"]),
                        masked_token_count=int(raw["masked_token_count"]),
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise SFTDatasetError(
                        f"invalid metadata record on line {line_number}"
                    ) from error
                expected_index = line_number - 1
                if info.dataset_index != expected_index:
                    raise SFTDatasetError(
                        f"metadata index mismatch on line {line_number}"
                    )
                expected_length = int(
                    self._offsets[expected_index + 1]
                    - self._offsets[expected_index]
                )
                if info.token_count != expected_length:
                    raise SFTDatasetError(
                        f"metadata token count mismatch on line {line_number}"
                    )
                if (
                    info.supervised_token_count + info.masked_token_count
                    != info.token_count
                ):
                    raise SFTDatasetError(
                        f"metadata supervision counts mismatch on line {line_number}"
                    )
                records.append(info)
        return tuple(records)

    def _ensure_arrays_open(self) -> None:
        if self._input_ids is None:
            self._input_ids = np.load(
                self.directory / "input_ids.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
        if self._labels is None:
            self._labels = np.load(
                self.directory / "labels.npy",
                mmap_mode="r",
                allow_pickle=False,
            )

    def _normalize_index(self, index: int) -> int:
        if isinstance(index, bool):
            raise TypeError("dataset index must be an integer, not bool")
        try:
            normalized = operator.index(index)
        except TypeError as error:
            raise TypeError("dataset index must be an integer") from error
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(f"dataset index out of range: {index}")
        return normalized

    def __len__(self) -> int:
        return len(self._trajectory_info)

    def __getitem__(self, index: int) -> SFTSample:
        normalized = self._normalize_index(index)
        self._ensure_arrays_open()
        assert self._input_ids is not None and self._labels is not None
        start = int(self._offsets[normalized])
        end = int(self._offsets[normalized + 1])
        info = self._trajectory_info[normalized]
        input_ids = self._input_ids[start:end]
        labels = self._labels[start:end]
        if input_ids.shape != labels.shape or len(input_ids) != info.token_count:
            raise SFTDatasetError(f"sample shape mismatch at index {normalized}")
        return SFTSample(
            dataset_index=normalized,
            trajectory_id=info.trajectory_id,
            task_id=info.task_id,
            input_ids=input_ids,
            labels=labels,
            token_count=info.token_count,
            supervised_token_count=info.supervised_token_count,
        )

    def trajectory_info(self, index: int) -> TrajectoryInfo:
        return self._trajectory_info[self._normalize_index(index)]

    @property
    def lengths(self) -> NDArray[np.int64]:
        """Read-only per-trajectory lengths used later by bucket sampling."""

        return self._lengths

    def close(self) -> None:
        """Drop local mmap references; they reopen lazily on the next access."""

        self._input_ids = None
        self._labels = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_input_ids"] = None
        state["_labels"] = None
        return state

