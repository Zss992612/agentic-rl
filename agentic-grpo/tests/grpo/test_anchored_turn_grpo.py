from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from agentic_rl.algorithms.anchored_turn_grpo import (
    compute_tau2_anchored_turn_grpo_advantage,
)
from verl.trainer.ppo.core_algos import (
    compute_grpo_outcome_advantage,
    get_adv_estimator_fn,
)


def _credits(*sample_spans: list[dict[str, object]]) -> dict[str, np.ndarray]:
    values = np.empty(len(sample_spans), dtype=object)
    values[:] = sample_spans
    return {"turn_credit_spans": values}


def _span(
    start: int,
    end: int,
    *,
    write: float = 0.0,
    fact: float = 0.0,
    error: bool = False,
) -> dict[str, object]:
    return {
        "start": start,
        "end": end,
        "new_progress": write,
        "new_fact_progress": fact,
        "model_error": error,
    }


def _mixed_group_inputs() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    return rewards, response_mask, np.array(["task-1", "task-1"], dtype=object)


def test_estimator_is_registered() -> None:
    assert (
        get_adv_estimator_fn("tau2_anchored_turn_grpo")
        is compute_tau2_anchored_turn_grpo_advantage
    )


def test_failed_trajectory_is_not_locally_shaped() -> None:
    rewards, response_mask, index = _mixed_group_inputs()
    spans = _credits(
        [_span(0, 2), _span(3, 4), _span(4, 7)],
        [
            _span(0, 2, write=1.0),
            _span(3, 4, fact=1.0, error=True),
            _span(4, 7, write=1.0, fact=1.0),
        ],
    )

    expected, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=False,
    )
    actual, _ = compute_tau2_anchored_turn_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        non_tensor_batch=spans,
        norm_adv_by_std_in_grpo=False,
    )

    torch.testing.assert_close(actual[1], expected[1])


def test_success_credit_preserves_sign_and_token_mean() -> None:
    rewards, response_mask, index = _mixed_group_inputs()
    advantages, _ = compute_tau2_anchored_turn_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        non_tensor_batch=_credits(
            [
                _span(0, 2, write=0.5),
                _span(3, 4, write=1.0, error=True),
                _span(4, 7),
            ],
            [_span(0, 2), _span(3, 4), _span(4, 7)],
        ),
        norm_adv_by_std_in_grpo=False,
    )

    active = response_mask[0].bool()
    outcome = 0.5
    assert torch.all(advantages[0, active] > 0)
    assert advantages[0, active].min() >= 0.6 * outcome
    assert advantages[0, active].max() <= 1.4 * outcome
    torch.testing.assert_close(advantages[0, active].mean(), torch.tensor(outcome))
    assert advantages[0, 0] > advantages[0, 3]


def test_length_scale_is_capped_at_four() -> None:
    rewards = torch.tensor(
        [
            [0.0] * 19 + [1.0],
            [0.0] * 20,
        ]
    )
    response_mask = torch.ones_like(rewards)
    long_turn_end = 20
    advantages, _ = compute_tau2_anchored_turn_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["task-1", "task-1"], dtype=object),
        non_tensor_batch=_credits(
            [_span(0, 1, write=0.1), _span(1, long_turn_end)],
            [_span(0, long_turn_end)],
        ),
        norm_adv_by_std_in_grpo=False,
    )

    q = 0.8 * 0.1 * math.sqrt(16.0)
    q_mean = q / 20
    expected_first_token = 0.5 * (1.0 + 0.4 * (q - q_mean))
    torch.testing.assert_close(advantages[0, 0], torch.tensor(expected_first_token))


def test_zero_variance_success_group_stays_zero() -> None:
    rewards = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    response_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    advantages, _ = compute_tau2_anchored_turn_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["task-1", "task-1"], dtype=object),
        non_tensor_batch=_credits(
            [_span(0, 1, write=1.0)],
            [_span(0, 1, fact=1.0)],
        ),
    )

    torch.testing.assert_close(advantages, torch.zeros_like(advantages))


def test_turn_spans_must_partition_assistant_tokens() -> None:
    with pytest.raises(ValueError, match="must partition"):
        compute_tau2_anchored_turn_grpo_advantage(
            token_level_rewards=torch.zeros((1, 4)),
            response_mask=torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
            index=np.array(["task-1"], dtype=object),
            non_tensor_batch=_credits([_span(0, 2)]),
        )
