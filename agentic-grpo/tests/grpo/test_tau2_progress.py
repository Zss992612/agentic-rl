from __future__ import annotations

from copy import deepcopy

import pytest
from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.environment.environment import Environment

from agentic_rl.envs.tau2_retail.progress import RetailDBProgressTracker


def _task(task_id: str) -> Task:
    return next(task for task in get_tasks(task_split_name=None) if task.id == task_id)


def _initialized_environment(task: Task) -> Environment:
    environment = get_environment()
    initial_state = task.initial_state
    environment.set_state(
        initialization_data=(
            initial_state.initialization_data if initial_state is not None else None
        ),
        initialization_actions=(
            initial_state.initialization_actions if initial_state is not None else None
        ),
        message_history=(
            list(initial_state.message_history or [])
            if initial_state is not None
            else []
        ),
    )
    return environment


def _db(environment: Environment) -> RetailDB:
    assert environment.tools is not None
    assert isinstance(environment.tools.db, RetailDB)
    return environment.tools.db


def _replay_reference_actions(environment: Environment, task: Task) -> None:
    assert task.evaluation_criteria is not None
    assert environment.tools is not None
    for action in task.evaluation_criteria.actions or []:
        if not environment.tools.has_tool(action.name):
            continue
        if not environment.tools.tool_mutates_state(action.name):
            continue
        environment.make_tool_call(
            tool_name=action.name,
            requestor=action.requestor,
            **action.arguments,
        )


def test_correct_writes_accumulate_progress_to_one() -> None:
    # Task 4 has two independent write actions, preceded by read-only lookups.
    task = _task("4")
    tracker = RetailDBProgressTracker.from_task(task)
    environment = _initialized_environment(task)

    assert tracker.target_field_count > 0
    assert tracker.initial_potential == 0.0
    assert tracker.current_potential == 0.0
    assert tracker.best_potential == 0.0

    observations = []
    assert task.evaluation_criteria is not None
    assert environment.tools is not None
    for action in task.evaluation_criteria.actions or []:
        if not environment.tools.has_tool(action.name):
            continue
        if not environment.tools.tool_mutates_state(action.name):
            continue
        environment.make_tool_call(
            tool_name=action.name,
            requestor=action.requestor,
            **action.arguments,
        )
        observations.append(tracker.observe(_db(environment)))

    positive_progress = [item.progress for item in observations if item.progress > 0.0]
    assert len(positive_progress) == 2
    assert sum(positive_progress) == pytest.approx(1.0)
    assert observations[-1].potential == 1.0
    assert observations[-1].best_potential == 1.0


def test_read_and_unchanged_observations_have_no_progress() -> None:
    task = _task("4")
    tracker = RetailDBProgressTracker.from_task(task)
    environment = _initialized_environment(task)
    assert task.evaluation_criteria is not None
    read_action = (task.evaluation_criteria.actions or [])[0]

    before = tracker.observe(_db(environment))
    environment.make_tool_call(
        tool_name=read_action.name,
        requestor=read_action.requestor,
        **read_action.arguments,
    )
    after = tracker.observe(_db(environment))

    assert before.potential == before.progress == 0.0
    assert after.potential == after.progress == 0.0
    assert tracker.best_potential == 0.0


def test_regression_is_not_penalized_and_recovery_is_not_rewarded_twice() -> None:
    task = _task("4")
    tracker = RetailDBProgressTracker.from_task(task)
    environment = _initialized_environment(task)
    initial_db = deepcopy(_db(environment))
    _replay_reference_actions(environment, task)
    gold_db = deepcopy(_db(environment))

    completed = tracker.observe(gold_db)
    regressed = tracker.observe(initial_db)
    restored = tracker.observe(gold_db)

    assert completed.potential == completed.progress == 1.0
    assert regressed.potential == 0.0
    assert regressed.progress == 0.0
    assert regressed.best_potential == 1.0
    assert restored.potential == 1.0
    assert restored.progress == 0.0
    assert restored.best_potential == 1.0


def test_cached_task_spec_keeps_rollout_state_independent() -> None:
    task = _task("4")
    first = RetailDBProgressTracker.from_task(task)
    second = RetailDBProgressTracker.from_task(task)
    environment = _initialized_environment(task)
    _replay_reference_actions(environment, task)

    assert first.observe(_db(environment)).progress == 1.0
    assert first.best_potential == 1.0
    assert second.current_potential == 0.0
    assert second.best_potential == 0.0


def test_task_without_changed_target_fields_stays_at_zero() -> None:
    task = _task("24")
    tracker = RetailDBProgressTracker.from_task(task)
    environment = _initialized_environment(task)

    observation = tracker.observe(_db(environment))

    assert tracker.target_field_count == 0
    assert tracker.initial_potential == 0.0
    assert tracker.current_potential == 0.0
    assert tracker.best_potential == 0.0
    assert observation.potential == 0.0
    assert observation.progress == 0.0
    assert observation.best_potential == 0.0
