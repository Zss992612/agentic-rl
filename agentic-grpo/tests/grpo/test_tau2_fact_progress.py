from __future__ import annotations

from typing import Any

import pytest
from tau2.data_model.tasks import Action, Task
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.environment.environment import Environment

from agentic_rl.envs.tau2_retail.progress import RetailFactProgressTracker


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


def _call(environment: Environment, action: Action) -> Any:
    result = environment.make_tool_call(
        tool_name=action.name,
        requestor=action.requestor,
        **action.arguments,
    )
    # AgentLoop receives ToolMessage.content, not the live Pydantic return.
    return environment.to_json_str(result)


def test_gold_reads_cover_target_facts_once() -> None:
    task = _task("10")
    tracker = RetailFactProgressTracker.from_task(task)
    environment = _initialized_environment(task)
    actions = task.evaluation_criteria.actions or []

    observations = [
        tracker.observe_tool_call(action.name, action.arguments, _call(environment, action))
        for action in actions
        if action.name.startswith(("find_", "get_"))
    ]

    assert tracker.target_fact_count > 0
    assert tracker.covered_fact_count == tracker.target_fact_count
    assert tracker.current_potential == tracker.best_potential == 1.0
    assert sum(item.progress for item in observations) == pytest.approx(1.0)
    assert all(item.progress > 0.0 for item in observations)

    repeated = tracker.observe_tool_call(
        actions[2].name,
        actions[2].arguments,
        _call(environment, actions[2]),
    )
    assert repeated.progress == 0.0
    assert repeated.new_fact_count == 0
    assert repeated.potential == 1.0


def test_returning_old_items_does_not_require_catalog_lookups() -> None:
    task = _task("11")
    tracker = RetailFactProgressTracker.from_task(task)
    environment = _initialized_environment(task)

    for action in task.evaluation_criteria.actions or []:
        if action.name.startswith(("find_", "get_")):
            tracker.observe_tool_call(
                action.name,
                action.arguments,
                _call(environment, action),
            )

    assert tracker.target_fact_count > 0
    assert tracker.covered_fact_count == tracker.target_fact_count
    assert tracker.current_potential == 1.0


def test_equivalent_item_lookup_gets_partial_credit_without_exact_gold_call() -> None:
    task = _task("0")
    tracker = RetailFactProgressTracker.from_task(task)
    environment = _initialized_environment(task)

    item_id = "7706410293"
    result = environment.make_tool_call(
        "get_item_details",
        requestor="assistant",
        item_id=item_id,
    )
    first = tracker.observe_tool_call(
        "get_item_details",
        {"item_id": item_id},
        environment.to_json_str(result),
    )
    second = tracker.observe_tool_call(
        "get_item_details",
        {"item_id": item_id},
        environment.to_json_str(result),
    )

    assert first.new_fact_count > 0
    assert 0.0 < first.progress < 1.0
    assert second.new_fact_count == 0
    assert second.progress == 0.0


def test_irrelevant_or_failed_read_has_no_progress() -> None:
    task = _task("10")
    tracker = RetailFactProgressTracker.from_task(task)
    environment = _initialized_environment(task)
    unrelated_product_id = next(iter(environment.tools.db.products))
    result = environment.make_tool_call(
        "get_product_details",
        requestor="assistant",
        product_id=unrelated_product_id,
    )

    irrelevant = tracker.observe_tool_call(
        "get_product_details",
        {"product_id": unrelated_product_id},
        environment.to_json_str(result),
    )
    failed = tracker.observe_tool_call(
        "find_user_id_by_email",
        {"email": "wrong@example.com"},
        "Error: User not found",
        error=True,
    )

    assert irrelevant.progress == 0.0
    assert failed.progress == 0.0
    assert tracker.covered_fact_count == 0


def test_communicate_only_task_covers_scalar_facts_from_structured_result() -> None:
    task = _task("24")
    tracker = RetailFactProgressTracker.from_task(task)
    environment = _initialized_environment(task)
    order_id = "#W9609649"
    result = environment.make_tool_call(
        "get_order_details",
        requestor="assistant",
        order_id=order_id,
    )

    observation = tracker.observe_tool_call(
        "get_order_details",
        {"order_id": order_id},
        environment.to_json_str(result),
    )

    assert tracker.target_fact_count == 2
    assert observation.new_fact_count == 2
    assert observation.progress == 1.0
    assert observation.potential == 1.0


def test_numeric_calculation_result_can_cover_communicated_fact() -> None:
    task = _task("16")
    tracker = RetailFactProgressTracker.from_task(task)
    environment = _initialized_environment(task)
    calculate = next(
        action
        for action in task.evaluation_criteria.actions or []
        if action.name == "calculate"
    )
    before = tracker.covered_fact_count

    observation = tracker.observe_tool_call(
        calculate.name,
        calculate.arguments,
        _call(environment, calculate),
    )

    assert observation.new_fact_count == 1
    assert observation.covered_fact_count == before + 1
    assert observation.progress > 0.0


def test_valid_identity_lookup_survives_bad_reference_lookup() -> None:
    # Task 38 deliberately contains a failed email lookup followed by the
    # valid name+zip fallback.  Order->user reference expansion keeps the
    # successful authentication fact in the target.
    task = _task("38")
    tracker = RetailFactProgressTracker.from_task(task)
    environment = _initialized_environment(task)
    actions = task.evaluation_criteria.actions or []
    valid_lookup = actions[1]

    result = _call(environment, valid_lookup)
    observation = tracker.observe_tool_call(
        valid_lookup.name,
        valid_lookup.arguments,
        result,
    )

    assert observation.new_fact_count == 1
    assert observation.progress > 0.0


def test_communicated_word_matches_inside_structured_phrase() -> None:
    task = _task("38")
    tracker = RetailFactProgressTracker.from_task(task)
    environment = _initialized_environment(task)
    order_id = "#W9348897"
    result = environment.make_tool_call(
        "get_order_details",
        requestor="assistant",
        order_id=order_id,
    )

    observation = tracker.observe_tool_call(
        "get_order_details",
        {"order_id": order_id},
        environment.to_json_str(result),
    )

    # The task asks for "camera" and "481.50"; the structured result contains
    # the product name "Action Camera" and numeric price 481.5.
    assert tracker.target_fact_count > 3
    assert observation.new_fact_count >= 5
    assert observation.progress > 0.0


def test_communicated_word_from_unrelated_entity_gets_no_credit() -> None:
    tracker = RetailFactProgressTracker.from_task(_task("38"))

    observation = tracker.observe_tool_call(
        "get_product_details",
        {"product_id": "unrelated_product"},
        {
            "product_id": "unrelated_product",
            "name": "Action Camera",
            "variants": {},
        },
    )

    assert observation.new_fact_count == 0
    assert observation.progress == 0.0
    assert tracker.covered_fact_count == 0


def test_write_only_reference_recovers_read_prerequisites_from_db() -> None:
    task = _task("88")
    tracker = RetailFactProgressTracker.from_task(task)
    environment = _initialized_environment(task)
    order_id = "#W8835847"
    order = environment.tools.db.orders[order_id]
    user = environment.tools.db.users[order.user_id]

    calls = [
        (
            "find_user_id_by_name_zip",
            {
                "first_name": user.name.first_name,
                "last_name": user.name.last_name,
                "zip": user.address.zip,
            },
        ),
        ("get_user_details", {"user_id": user.user_id}),
        ("get_order_details", {"order_id": order_id}),
    ]
    for tool_name, arguments in calls:
        result = environment.make_tool_call(
            tool_name,
            requestor="assistant",
            **arguments,
        )
        tracker.observe_tool_call(
            tool_name,
            arguments,
            environment.to_json_str(result),
        )

    assert tracker.target_fact_count > 0
    assert tracker.covered_fact_count == tracker.target_fact_count
    assert tracker.current_potential == 1.0


def test_cached_fact_spec_keeps_rollout_coverage_independent() -> None:
    task = _task("10")
    first = RetailFactProgressTracker.from_task(task)
    second = RetailFactProgressTracker.from_task(task)
    environment = _initialized_environment(task)
    action = (task.evaluation_criteria.actions or [])[0]

    observation = first.observe_tool_call(
        action.name,
        action.arguments,
        _call(environment, action),
    )

    assert observation.progress > 0.0
    assert first.covered_fact_count > 0
    assert second.covered_fact_count == 0
    assert second.current_potential == 0.0


@pytest.mark.parametrize("task_id", ["50", "57"])
def test_task_without_structured_fact_target_stays_at_zero(task_id: str) -> None:
    tracker = RetailFactProgressTracker.from_task(_task(task_id))

    observation = tracker.observe_tool_call(
        "get_order_details",
        {"order_id": "#W0000000"},
        '{"order_id": "#W0000000", "status": "pending"}',
    )

    assert tracker.target_fact_count == 0
    assert tracker.covered_fact_count == 0
    assert observation.potential == 0.0
    assert observation.progress == 0.0
