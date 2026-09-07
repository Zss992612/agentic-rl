import json

import pytest

from agentic_rl.data import semantic_screening


def test_normalize_low_confidence_to_human_review() -> None:
    judgment = {
        "user_fidelity": True,
        "process_valid": True,
        "tool_error_resolved": None,
        "final_response_consistent": True,
        "reasons": [],
        "evidence_turns": [2],
        "confidence": 0.7,
    }
    result = semantic_screening.normalize_judgment(
        judgment,
        has_tool_errors=False,
        confidence_threshold=0.8,
    )
    assert result["decision"] == "needs_human_review"


def test_batch_resumes_by_trajectory_id(tmp_path, monkeypatch) -> None:
    candidate = {
        "trajectory_id": "trajectory-1",
        "task_id": "0",
        "screening": {"hard_pass": True, "task_rank": 1},
    }
    judgment = {
        "user_fidelity": True,
        "process_valid": True,
        "tool_error_resolved": None,
        "final_response_consistent": True,
        "decision": "accepted",
        "reasons": [],
        "evidence_turns": [],
        "confidence": 0.9,
    }
    monkeypatch.setattr(
        semantic_screening,
        "judge_candidate",
        lambda candidate, retail_policy, config: judgment,
    )
    output_path = tmp_path / "screened.jsonl"
    config = {"max_candidates": 20, "max_concurrency": 1}

    semantic_screening.run_batch([candidate], "policy", config, output_path)
    semantic_screening.run_batch([candidate], "policy", config, output_path)

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["semantic_screening"]["decision"] == "accepted"


def test_batch_only_catches_expected_judge_errors(tmp_path, monkeypatch) -> None:
    candidate = {
        "trajectory_id": "trajectory-1",
        "task_id": "0",
        "screening": {"hard_pass": True, "task_rank": 1},
    }
    config = {"max_candidates": 20, "max_concurrency": 1}
    output_path = tmp_path / "screened.jsonl"

    def raise_judge_error(*args) -> None:
        raise semantic_screening.JudgeError("expected failure")

    monkeypatch.setattr(semantic_screening, "judge_candidate", raise_judge_error)
    errors = semantic_screening.run_batch(
        [candidate], "policy", config, output_path
    )
    assert errors[0]["error"] == "expected failure"

    def raise_programming_error(*args) -> None:
        raise KeyError("unexpected bug")

    monkeypatch.setattr(
        semantic_screening, "judge_candidate", raise_programming_error
    )
    with pytest.raises(KeyError, match="unexpected bug"):
        semantic_screening.run_batch([candidate], "policy", config, output_path)
