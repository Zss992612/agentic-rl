"""Outcome-anchored, success-only turn credit for Retail GRPO."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    compute_grpo_outcome_advantage,
    register_adv_est,
)


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _turn_scores(
    spans: list[dict[str, Any]],
    response_mask: torch.Tensor,
    *,
    write_weight: float,
    fact_weight: float,
    length_ratio_cap: float,
    local_score_cap: float,
) -> list[tuple[int, int, int, float]]:
    """Return ``(start, end, train_tokens, q)`` for every Assistant turn."""
    response_length = response_mask.shape[0]
    trajectory_tokens = int(response_mask.sum().item())
    coverage = torch.zeros_like(response_mask, dtype=torch.int32)
    scored_turns: list[tuple[int, int, int, float]] = []

    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        if not 0 <= start < end <= response_length:
            raise ValueError(f"Invalid turn-credit span: [{start}, {end})")

        span_mask = response_mask[start:end].bool()
        turn_tokens = int(span_mask.sum().item())
        if turn_tokens == 0:
            raise ValueError(f"Turn-credit span has no trainable tokens: [{start}, {end})")
        coverage[start:end] += span_mask.to(torch.int32)

        if bool(span["model_error"]):
            local_score = 0.0
        else:
            local_score = (
                write_weight * float(span["new_progress"])
                + fact_weight * float(span.get("new_fact_progress", 0.0))
            )

        length_ratio = trajectory_tokens / turn_tokens
        length_scale = math.sqrt(min(length_ratio, length_ratio_cap))
        q = min(max(local_score * length_scale, 0.0), local_score_cap)
        scored_turns.append((start, end, turn_tokens, q))

    expected_coverage = response_mask.bool().to(torch.int32)
    if not torch.equal(coverage, expected_coverage):
        raise ValueError("Turn-credit spans must partition every Assistant loss token")

    return scored_turns


@register_adv_est("tau2_anchored_turn_grpo")
def compute_tau2_anchored_turn_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    non_tensor_batch: dict[str, Any],
    write_weight: float = 0.8,
    fact_weight: float = 0.2,
    lambda_success: float = 0.4,
    length_ratio_cap: float = 16.0,
    local_score_cap: float = 1.0,
    norm_adv_by_std_in_grpo: bool | None = None,
    config: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Redistribute positive outcome advantage within successful trajectories.

    Failed trajectories keep their vanilla token advantages. Successful turns
    are reweighted around the trajectory's token-weighted mean, so the sign and
    mean outcome advantage are preserved.
    """
    if norm_adv_by_std_in_grpo is None:
        norm_adv_by_std_in_grpo = _config_value(
            config,
            "norm_adv_by_std_in_grpo",
            True,
        )

    outcome_advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    raw_rewards = token_level_rewards.float().sum(dim=-1)
    advantages = outcome_advantages.float().clone()
    credit_spans = non_tensor_batch["turn_credit_spans"]
    if len(credit_spans) != advantages.shape[0]:
        raise ValueError("Expected one turn-credit span list per trajectory")

    for sample_index, spans in enumerate(credit_spans):
        sample_mask = response_mask[sample_index]
        trajectory_tokens = int(sample_mask.sum().item())
        if trajectory_tokens == 0:
            continue

        scored_turns = _turn_scores(
            spans,
            sample_mask,
            write_weight=write_weight,
            fact_weight=fact_weight,
            length_ratio_cap=length_ratio_cap,
            local_score_cap=local_score_cap,
        )

        outcome = (
            advantages[sample_index].sum()
            / sample_mask.sum().to(advantages.dtype)
        )
        is_shaped_success = (
            raw_rewards[sample_index].item() > 0.5
            and outcome.item() > 0.0
        )
        if not is_shaped_success:
            continue

        q_mean = sum(turn_tokens * q for _, _, turn_tokens, q in scored_turns)
        q_mean /= trajectory_tokens

        for start, end, _, q in scored_turns:
            multiplier = 1.0 + lambda_success * (q - q_mean)
            span_mask = sample_mask[start:end].bool()
            advantages[sample_index, start:end] = torch.where(
                span_mask,
                outcome * multiplier,
                advantages[sample_index, start:end],
            )

    advantages = advantages.to(outcome_advantages.dtype)
    advantages = advantages * response_mask.to(advantages.dtype)
    returns = advantages.clone()
    return advantages, returns
