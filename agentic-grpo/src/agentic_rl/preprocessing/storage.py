"""Storage contract and integrity checks for the final flat SFT dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agentic_rl.preprocessing.masking import IGNORE_INDEX

FORMAT_NAME = "agentic_rl_flat_sft"
FORMAT_VERSION = 1
INPUT_DTYPE = np.dtype("int32")
LABEL_DTYPE = np.dtype("int32")
OFFSET_DTYPE = np.dtype("int64")


class StorageValidationError(ValueError):
    """Raised when persisted arrays violate the training-data contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_flat_arrays(
    input_ids: NDArray[np.integer[Any]],
    labels: NDArray[np.integer[Any]],
    offsets: NDArray[np.integer[Any]],
) -> None:
    """Validate shape, dtype, offsets, and assistant-only label values."""

    if input_ids.ndim != 1 or labels.ndim != 1 or offsets.ndim != 1:
        raise StorageValidationError("input_ids, labels, and offsets must be 1-D")
    if input_ids.dtype != INPUT_DTYPE:
        raise StorageValidationError(
            f"input_ids dtype must be {INPUT_DTYPE}, got {input_ids.dtype}"
        )
    if labels.dtype != LABEL_DTYPE:
        raise StorageValidationError(
            f"labels dtype must be {LABEL_DTYPE}, got {labels.dtype}"
        )
    if offsets.dtype != OFFSET_DTYPE:
        raise StorageValidationError(
            f"offsets dtype must be {OFFSET_DTYPE}, got {offsets.dtype}"
        )
    if input_ids.shape != labels.shape:
        raise StorageValidationError("input_ids and labels must have equal shape")
    if len(offsets) < 2:
        raise StorageValidationError("offsets must contain at least one trajectory")
    if int(offsets[0]) != 0:
        raise StorageValidationError("offsets must start at zero")
    if int(offsets[-1]) != len(input_ids):
        raise StorageValidationError(
            "final offset must equal the flattened token count"
        )
    if np.any(np.diff(offsets) <= 0):
        raise StorageValidationError("offsets must be strictly increasing")
    invalid_labels = (labels != IGNORE_INDEX) & (labels != input_ids)
    if np.any(invalid_labels):
        first = int(np.flatnonzero(invalid_labels)[0])
        raise StorageValidationError(
            f"label at flat token {first} is neither {IGNORE_INDEX} nor input_id"
        )
    if not np.any(labels != IGNORE_INDEX):
        raise StorageValidationError("dataset contains no supervised tokens")


def validate_dataset_directory(
    directory: Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, int]:
    """Load a finished dataset read-only and verify all cross-file invariants."""

    required = {
        "input_ids.npy",
        "labels.npy",
        "offsets.npy",
        "metadata.jsonl",
        "stats.json",
        "manifest.json",
    }
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        raise StorageValidationError(
            f"dataset directory is missing files: {', '.join(missing)}"
        )

    input_ids = np.load(directory / "input_ids.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(directory / "labels.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(directory / "offsets.npy", mmap_mode="r", allow_pickle=False)
    validate_flat_arrays(input_ids, labels, offsets)

    metadata: list[dict[str, Any]] = []
    with (directory / "metadata.jsonl").open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise StorageValidationError(
                    f"invalid metadata JSON on line {line_number}"
                ) from error
            metadata.append(record)
    trajectory_count = len(offsets) - 1
    if len(metadata) != trajectory_count:
        raise StorageValidationError(
            "metadata row count does not match offsets: "
            f"{len(metadata)} != {trajectory_count}"
        )
    for index, record in enumerate(metadata):
        expected_start = int(offsets[index])
        expected_end = int(offsets[index + 1])
        if record.get("dataset_index") != index:
            raise StorageValidationError(f"metadata index mismatch at row {index}")
        if record.get("offset_start") != expected_start:
            raise StorageValidationError(f"metadata start mismatch at row {index}")
        if record.get("offset_end") != expected_end:
            raise StorageValidationError(f"metadata end mismatch at row {index}")
        if record.get("token_count") != expected_end - expected_start:
            raise StorageValidationError(f"metadata length mismatch at row {index}")

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_name") != FORMAT_NAME:
        raise StorageValidationError("unexpected dataset format name")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise StorageValidationError("unsupported dataset format version")
    if manifest.get("counts", {}).get("trajectories") != trajectory_count:
        raise StorageValidationError("manifest trajectory count mismatch")
    if manifest.get("counts", {}).get("tokens") != len(input_ids):
        raise StorageValidationError("manifest token count mismatch")

    if verify_hashes:
        for name, file_info in manifest.get("files", {}).items():
            path = directory / name
            if not path.is_file():
                raise StorageValidationError(f"manifest file is missing: {name}")
            actual = sha256_file(path)
            if actual != file_info.get("sha256"):
                raise StorageValidationError(f"SHA-256 mismatch for {name}")

    return {
        "trajectories": trajectory_count,
        "tokens": len(input_ids),
        "supervised_tokens": int(np.count_nonzero(labels != IGNORE_INDEX)),
    }
