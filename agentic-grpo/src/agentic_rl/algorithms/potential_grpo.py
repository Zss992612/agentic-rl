"""Turn-local potential shaping layered on top of vanilla GRPO."""

from __future__ import annotations

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


@register_adv_est("tau2_potential_grpo")
def compute_tau2_potential_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    non_tensor_batch: dict[str, Any],
    progress_coef: float = 0.0,
    fact_progress_coef: float = 0.0,
    error_coef: float = 0.0,
    norm_adv_by_std_in_grpo: bool | None = None,
    config: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign verifiable turn-local credit on top of outcome GRPO.

    ``start`` and ``end`` in each turn-credit record are offsets into the
    response tensor. Outcome rewards keep vanilla GRPO semantics, including
    rewards stored at masked observation positions.

    Error turns receive a fixed negative advantage. Turns with verifiable
    positive write or fact progress receive that local credit and keep only
    the non-negative part of their outcome advantage. All other turns retain
    the vanilla outcome advantage.
    """
    if norm_adv_by_std_in_grpo is None:
        norm_adv_by_std_in_grpo = _config_value(
            config,
            "norm_adv_by_std_in_grpo",
            True,
        )

    outcome_advantages, outcome_returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )

    if progress_coef == 0.0 and fact_progress_coef == 0.0 and error_coef == 0.0:
        return outcome_advantages, outcome_returns

    advantages = outcome_advantages.clone()
    for sample_index, spans in enumerate(non_tensor_batch["turn_credit_spans"]):
        for span in spans:
            start = int(span["start"])
            end = int(span["end"])
            positive_local_credit = (
                progress_coef * float(span["new_progress"])
                + fact_progress_coef * float(span.get("new_fact_progress", 0.0))
            )

            if span["model_error"] and error_coef != 0.0:
                advantages[sample_index, start:end] = -error_coef
            elif positive_local_credit > 0.0:
                outcome = outcome_advantages[sample_index, start:end]
                advantages[sample_index, start:end] = (
                    torch.clamp_min(outcome, 0.0) + positive_local_credit
                )

    advantages = advantages * response_mask
    returns = advantages.clone()
    return advantages, returns
