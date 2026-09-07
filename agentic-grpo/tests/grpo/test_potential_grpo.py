from __future__ import annotations

import numpy as np
import torch

from agentic_rl.algorithms.potential_grpo import (
    compute_tau2_potential_grpo_advantage,
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
    new_progress: float = 0.0,
    new_fact_progress: float = 0.0,
    model_error: bool = False,
) -> dict[str, object]:
    return {
        "start": start,
        "end": end,
        "new_progress": new_progress,
        "new_fact_progress": new_fact_progress,
        "model_error": model_error,
    }


def test_estimator_is_registered_by_string_name() -> None:
    assert (
        get_adv_estimator_fn("tau2_potential_grpo")
        is compute_tau2_potential_grpo_advantage
    )


def test_zero_coefficients_match_vanilla_grpo() -> None:
    rewards = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    index = np.array(["task-1", "task-1"], dtype=object)

    expected = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=False,
    )
    actual = compute_tau2_potential_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        non_tensor_batch={},
        progress_coef=0.0,
        fact_progress_coef=0.0,
        error_coef=0.0,
        norm_adv_by_std_in_grpo=False,
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_local_credit_only_affects_its_assistant_span() -> None:
    rewards = torch.zeros((1, 8))
    response_mask = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
    )
    spans = _credits(
        [
            _span(0, 2, new_progress=0.5, new_fact_progress=0.5),
            _span(4, 6, model_error=True),
        ]
    )

    advantages, returns = compute_tau2_potential_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["task-1"], dtype=object),
        non_tensor_batch=spans,
        progress_coef=0.4,
        fact_progress_coef=0.1,
        error_coef=0.3,
    )

    expected = torch.tensor([[0.25, 0.25, 0.0, 0.0, -0.3, -0.3, 0.0, 0.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_successful_trajectory_error_turn_is_fixed_negative() -> None:
    rewards = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    advantages, returns = compute_tau2_potential_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["task-1", "task-1"], dtype=object),
        non_tensor_batch=_credits(
            [
                _span(
                    0,
                    1,
                    new_progress=1.0,
                    new_fact_progress=1.0,
                    model_error=True,
                )
            ],
            [],
        ),
        progress_coef=0.25,
        fact_progress_coef=0.1,
        error_coef=0.25,
        norm_adv_by_std_in_grpo=False,
    )

    expected = torch.tensor(
        [
            [-0.25, 0.5, 0.0],
            [-0.5, -0.5, 0.0],
        ]
    )
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_zero_error_coefficient_disables_error_override() -> None:
    rewards = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    advantages, _ = compute_tau2_potential_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["task-1", "task-1"], dtype=object),
        non_tensor_batch=_credits(
            [_span(0, 1, model_error=True)],
            [],
        ),
        progress_coef=0.25,
        error_coef=0.0,
        norm_adv_by_std_in_grpo=False,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor(
            [
                [0.5, 0.5, 0.0],
                [-0.5, -0.5, 0.0],
            ]
        ),
    )


def test_failed_trajectory_positive_progress_gets_positive_local_credit() -> None:
    rewards = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    advantages, returns = compute_tau2_potential_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["task-1", "task-1"], dtype=object),
        non_tensor_batch=_credits(
            [],
            [_span(0, 1, new_progress=0.5, new_fact_progress=0.5)],
        ),
        progress_coef=0.25,
        fact_progress_coef=0.1,
        norm_adv_by_std_in_grpo=False,
    )

    expected = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [0.175, -0.5, 0.0],
        ]
    )
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_fact_progress_is_independent_from_write_progress() -> None:
    advantages, returns = compute_tau2_potential_grpo_advantage(
        token_level_rewards=torch.zeros((1, 6)),
        response_mask=torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0]]),
        index=np.array(["task-1"], dtype=object),
        non_tensor_batch=_credits(
            [
                _span(0, 2, new_fact_progress=0.5),
                _span(3, 5, new_progress=0.5),
            ]
        ),
        progress_coef=0.25,
        fact_progress_coef=0.1,
    )

    expected = torch.tensor([[0.05, 0.05, 0.0, 0.125, 0.125, 0.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_legacy_span_without_fact_progress_remains_supported() -> None:
    legacy_span = {
        "start": 0,
        "end": 2,
        "new_progress": 0.5,
        "model_error": False,
    }
    advantages, _ = compute_tau2_potential_grpo_advantage(
        token_level_rewards=torch.zeros((1, 2)),
        response_mask=torch.ones((1, 2)),
        index=np.array(["task-1"], dtype=object),
        non_tensor_batch=_credits([legacy_span]),
        progress_coef=0.25,
        fact_progress_coef=0.1,
    )

    torch.testing.assert_close(advantages, torch.full((1, 2), 0.125))


def test_mask_zeros_observation_and_padding_even_if_a_span_overlaps_them() -> None:
    response_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    advantages, _ = compute_tau2_potential_grpo_advantage(
        token_level_rewards=torch.zeros((1, 4)),
        response_mask=response_mask,
        index=np.array(["task-1"], dtype=object),
        non_tensor_batch=_credits([_span(0, 4, new_progress=1.0)]),
        progress_coef=0.25,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([[0.25, 0.0, 0.25, 0.0]]),
    )


def test_local_credit_remains_nonzero_when_group_rewards_are_identical() -> None:
    rewards = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    advantages, _ = compute_tau2_potential_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["task-1", "task-1"], dtype=object),
        non_tensor_batch=_credits(
            [_span(0, 2, new_progress=0.5)],
            [_span(0, 2, model_error=True)],
        ),
        progress_coef=0.4,
        error_coef=0.3,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor(
            [
                [0.2, 0.2, 0.0],
                [-0.3, -0.3, 0.0],
            ]
        ),
    )


def test_masked_terminal_reward_still_contributes_to_outcome_advantage() -> None:
    rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )

    advantages, _ = compute_tau2_potential_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["task-1", "task-1"], dtype=object),
        non_tensor_batch={},
        norm_adv_by_std_in_grpo=False,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor(
            [
                [-0.5, -0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
            ]
        ),
    )
