from __future__ import annotations

import numpy as np
import pytest

from agentic_rl.preprocessing.storage import (
    StorageValidationError,
    validate_flat_arrays,
)


def test_valid_flat_arrays() -> None:
    input_ids = np.asarray([10, 11, 12, 20, 21], dtype=np.int32)
    labels = np.asarray([-100, 11, 12, -100, 21], dtype=np.int32)
    offsets = np.asarray([0, 3, 5], dtype=np.int64)

    validate_flat_arrays(input_ids, labels, offsets)


def test_rejects_non_monotonic_offsets() -> None:
    input_ids = np.asarray([10, 11], dtype=np.int32)
    labels = np.asarray([-100, 11], dtype=np.int32)
    offsets = np.asarray([0, 0, 2], dtype=np.int64)

    with pytest.raises(StorageValidationError, match="strictly increasing"):
        validate_flat_arrays(input_ids, labels, offsets)


def test_rejects_label_that_is_not_input_id() -> None:
    input_ids = np.asarray([10, 11], dtype=np.int32)
    labels = np.asarray([-100, 99], dtype=np.int32)
    offsets = np.asarray([0, 2], dtype=np.int64)

    with pytest.raises(StorageValidationError, match="neither -100 nor input_id"):
        validate_flat_arrays(input_ids, labels, offsets)
