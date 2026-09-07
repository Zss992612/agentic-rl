from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/preprocessing/scan_sft_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("scan_sft_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_numeric_summary_and_percentiles() -> None:
    summary = MODULE.numeric_summary([1, 2, 3, 4, 5])

    assert summary["min"] == 1
    assert summary["max"] == 5
    assert summary["median"] == 3
    assert summary["p90"] == 4.6


def test_length_bins() -> None:
    assert MODULE.length_bin(4_096, 16_384) == "00000-04096"
    assert MODULE.length_bin(4_097, 16_384) == "04097-08192"
    assert MODULE.length_bin(8_193, 16_384) == "08193-12288"
    assert MODULE.length_bin(16_384, 16_384) == "12289-16384"
    assert MODULE.length_bin(16_385, 16_384) == ">16384"
