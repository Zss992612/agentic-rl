"""Offline smoke test for the real τ-bench Retail environment and verifier.

This deliberately avoids an LLM user simulator. It executes real Retail tools,
records their ToolMessages, replays the trajectory in a fresh environment, and
checks the official environment and action evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass

from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.data_model.tasks import Action
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.evaluator.evaluator_env import EnvironmentEvaluator


@dataclass(frozen=True)
class Score:
    env_reward: float
    db_reward: float
    action_diagnostic: float
    message_count: int


def build_trajectory(actions: list[Action]) -> list[Message]:
    """Execute actions against a fresh official Retail environment."""
    environment = get_environment()
    messages: list[Message] = []
    for action in actions:
        tool_call = ToolCall(
            id=action.action_id,
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        messages.append(
            AssistantMessage(role="assistant", tool_calls=[tool_call])
        )
        messages.append(environment.get_response(tool_call))
    return messages


def score(actions: list[Action], task) -> Score:
    trajectory = build_trajectory(actions)
    environment_result = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=trajectory,
    )
    action_result = ActionEvaluator.calculate_reward(
        task=task,
        full_trajectory=trajectory,
    )
    assert environment_result.db_check is not None
    return Score(
        env_reward=environment_result.reward,
        db_reward=environment_result.db_check.db_reward,
        action_diagnostic=action_result.reward,
        message_count=len(trajectory),
    )


def main() -> None:
    task = next(task for task in get_tasks(None) if task.id == "0")
    assert task.evaluation_criteria is not None
    actions = task.evaluation_criteria.actions or []
    assert actions, "Retail task 0 should contain reference actions"

    results = {
        "reference": score(actions, task),
        "write_only": score([actions[-1]], task),
        "reads_only": score(actions[:-1], task),
    }

    assert results["reference"].env_reward == 1.0
    assert results["reference"].action_diagnostic == 1.0
    assert results["write_only"].env_reward == 1.0
    assert results["write_only"].action_diagnostic == 0.0
    assert results["reads_only"].env_reward == 0.0

    for name, result in results.items():
        print(
            f"{name}: env_reward={result.env_reward:.0f}, "
            f"db_reward={result.db_reward:.0f}, "
            f"action_diagnostic={result.action_diagnostic:.0f}, "
            f"messages={result.message_count}"
        )


if __name__ == "__main__":
    main()
