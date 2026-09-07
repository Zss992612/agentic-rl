"""World-state and knowledge-state progress for τ²-bench Retail.

The official environment evaluator derives its target database by replaying a
task's reference actions on an independently initialized environment.  This
module uses the same construction, then keeps only the leaf fields that differ
between the initial and target databases.  Observations therefore inspect a
small set of paths instead of copying or serializing the complete Retail DB on
every assistant turn.  This first baseline does not penalize collateral writes
to non-target fields or enforce when user confirmation should occur; those are
separate signals rather than properties of this potential.

Read progress uses a second, monotonic potential.  Its target is a compact set
of structured facts exposed by the task's reference reads and needed by later
reference actions.  A rollout earns credit only the first time it observes one
of those facts.  Matching facts rather than exact calls allows equivalent read
paths (for example, ``get_item_details`` instead of a broader product lookup)
without rewarding arbitrary successful queries.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, TypeAlias

from pydantic import BaseModel
from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolType

logger = logging.getLogger(__name__)

PathPart: TypeAlias = str | int
FieldPath: TypeAlias = tuple[PathPart, ...]
NormalizedValue: TypeAlias = tuple[Any, ...]

_MISSING = object()
_MISSING_VALUE: NormalizedValue = ("missing",)

_FACT_TOOL_NAMES = frozenset(
    {
        "calculate",
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "get_user_details",
        "get_order_details",
        "get_product_details",
        "get_item_details",
        "list_all_product_types",
    }
)


@dataclass(frozen=True, slots=True)
class RetailProgressObservation:
    """Potential and newly earned best-so-far progress for one DB state."""

    potential: float
    progress: float
    best_potential: float


@dataclass(frozen=True, slots=True)
class RetailFactProgressObservation:
    """Knowledge potential and first-time fact coverage for one tool call."""

    potential: float
    progress: float
    best_potential: float
    new_fact_count: int
    covered_fact_count: int


@dataclass(frozen=True, slots=True)
class _TargetField:
    path: FieldPath
    value: NormalizedValue


@dataclass(frozen=True, slots=True)
class _RetailFact:
    """One semantic Retail fact, independent of the tool that exposed it."""

    kind: str
    subject: tuple[str, ...]
    value: NormalizedValue = ("none",)


@dataclass(frozen=True, slots=True)
class _GoldRead:
    name: str
    arguments: Mapping[str, Any]
    result: Any


@dataclass(frozen=True, slots=True)
class _TaskProgressSpec:
    target_fields: tuple[_TargetField, ...]
    target_facts: tuple[_RetailFact, ...]
    initial_potential: float


_TASK_PROGRESS_SPECS: dict[str, _TaskProgressSpec] = {}


class RetailDBProgressTracker:
    """Measure partial completion of a Retail task's target DB mutation.

    Potential is the fraction of target leaf fields whose current values equal
    their values in the gold environment.  ``progress`` is monotonic credit:
    only an increase over the highest potential seen so far is returned.  A
    regression is visible in ``potential`` but is not negatively rewarded, and
    restoring an already reached state earns no duplicate credit.
    """

    def __init__(
        self,
        *,
        spec: _TaskProgressSpec,
    ) -> None:
        self._target_fields = spec.target_fields
        self.initial_potential = spec.initial_potential
        self.current_potential = self.initial_potential
        self.best_potential = self.initial_potential

    @classmethod
    def from_task(cls, task: Task) -> "RetailDBProgressTracker":
        """Build a tracker using the same gold-state replay as τ² evaluation."""

        spec = _TASK_PROGRESS_SPECS.get(task.id)
        if spec is None:
            spec = _build_progress_spec(task)
            _TASK_PROGRESS_SPECS[task.id] = spec
        return cls(spec=spec)

    @property
    def target_field_count(self) -> int:
        """Return the number of changed DB leaves used by the potential."""

        return len(self._target_fields)

    def observe(self, db: RetailDB) -> RetailProgressObservation:
        """Observe ``db`` and return current potential plus new monotonic credit."""

        potential = self._measure(db)
        previous_best = self.best_potential
        self.current_potential = potential
        self.best_potential = max(previous_best, potential)
        return RetailProgressObservation(
            potential=potential,
            progress=self.best_potential - previous_best,
            best_potential=self.best_potential,
        )

    def _measure(self, db: RetailDB) -> float:
        return _measure_target_fields(self._target_fields, db)


class RetailFactProgressTracker:
    """Measure first-time coverage of task-relevant structured facts.

    The target facts are immutable and cached per task.  Covered facts are
    rollout-local, so repeated queries and alternative tools returning facts
    already seen earn no duplicate credit.
    """

    def __init__(self, *, spec: _TaskProgressSpec) -> None:
        self._target_facts = frozenset(spec.target_facts)
        self._has_structural_targets = any(
            fact.kind != "communicate_value" for fact in self._target_facts
        )
        self._covered_facts: set[_RetailFact] = set()
        self.current_potential = 0.0
        self.best_potential = 0.0

    @classmethod
    def from_task(cls, task: Task) -> "RetailFactProgressTracker":
        spec = _TASK_PROGRESS_SPECS.get(task.id)
        if spec is None:
            spec = _build_progress_spec(task)
            _TASK_PROGRESS_SPECS[task.id] = spec
        return cls(spec=spec)

    @property
    def target_fact_count(self) -> int:
        return len(self._target_facts)

    @property
    def covered_fact_count(self) -> int:
        return len(self._covered_facts)

    def observe_tool_call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        result: Any,
        *,
        error: bool = False,
    ) -> RetailFactProgressObservation:
        """Record facts returned by one successful Retail read tool call."""

        if error or tool_name not in _FACT_TOOL_NAMES:
            return self._observation(progress=0.0, new_fact_count=0)

        observed = _facts_from_tool_result(tool_name, arguments, result)
        if self._has_structural_targets:
            structural = {
                fact for fact in observed if fact.kind != "communicate_value"
            }
            # A scalar or word only counts when it came from a task-relevant
            # entity/result.  Communicate-only tasks have no entity reference,
            # so they intentionally retain the lightweight value fallback.
            if not structural & self._target_facts:
                observed = structural
        newly_covered = (observed & self._target_facts) - self._covered_facts
        self._covered_facts.update(newly_covered)

        previous_best = self.best_potential
        if self.target_fact_count:
            self.current_potential = (
                self.covered_fact_count / self.target_fact_count
            )
        self.best_potential = max(previous_best, self.current_potential)
        return self._observation(
            progress=self.best_potential - previous_best,
            new_fact_count=len(newly_covered),
        )

    def _observation(
        self,
        *,
        progress: float,
        new_fact_count: int,
    ) -> RetailFactProgressObservation:
        return RetailFactProgressObservation(
            potential=self.current_potential,
            progress=progress,
            best_potential=self.best_potential,
            new_fact_count=new_fact_count,
            covered_fact_count=self.covered_fact_count,
        )


def _build_progress_spec(task: Task) -> _TaskProgressSpec:
    """Construct the immutable per-task target shared by all rollouts."""

    initial_environment = get_environment()
    _set_initial_state(initial_environment, task)

    gold_environment = get_environment()
    _set_initial_state(gold_environment, task)
    actions = (
        task.evaluation_criteria.actions
        if task.evaluation_criteria is not None
        else None
    )
    gold_reads: list[_GoldRead] = []
    for action in actions or []:
        try:
            result = gold_environment.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
            if (
                gold_environment.tools is not None
                and (
                    gold_environment.tools.tool_type(action.name) == ToolType.READ
                    or action.name == "calculate"
                )
            ):
                gold_reads.append(
                    _GoldRead(
                        name=action.name,
                        arguments=action.arguments,
                        # Retail read tools return live Pydantic objects backed
                        # by the environment DB.  Later reference writes can
                        # mutate them, so preserve the result at read time.
                        result=deepcopy(result),
                    )
                )
        except Exception as exc:
            # Keep this aligned with EnvironmentEvaluator: one malformed
            # reference action should not prevent deriving the remaining
            # target state.
            logger.warning(
                "Could not replay Retail reference action %s(%s): %s",
                action.name,
                action.arguments,
                exc,
            )

    initial_db = _environment_db(initial_environment)
    gold_db = _environment_db(gold_environment)
    target_fields = _changed_target_fields(initial_db, gold_db)
    target_facts = _build_target_facts(task, initial_db, gold_reads)
    return _TaskProgressSpec(
        target_fields=target_fields,
        target_facts=target_facts,
        initial_potential=_measure_target_fields(target_fields, initial_db),
    )


@dataclass(slots=True)
class _FactReferences:
    user_ids: set[str]
    order_ids: set[str]
    product_ids: set[str]
    order_item_ids: set[str]
    catalog_item_ids: set[str]
    payment_method_ids: set[str]
    communicate_values: set[NormalizedValue]


def _build_target_facts(
    task: Task,
    initial_db: RetailDB,
    gold_reads: list[_GoldRead],
) -> tuple[_RetailFact, ...]:
    """Build goal-conditioned facts that a valid read path can expose."""

    references = _fact_references(task)
    _expand_references_from_db(initial_db, references)
    # Gold reads define one valid information path.  DB-derived facts are only
    # used for compact prerequisites of downstream writes that the reference
    # path omitted (common in newer Retail tasks).
    db_candidates = _facts_for_referenced_db(initial_db, references)
    gold_candidates: set[_RetailFact] = set()
    for read in gold_reads:
        gold_candidates.update(
            _facts_from_tool_result(read.name, read.arguments, read.result)
        )

    targets = {
        fact
        for fact in gold_candidates
        if _fact_is_relevant(fact, references)
    }
    targets.update(
        fact
        for fact in db_candidates
        if _fact_is_write_prerequisite(fact, references)
    )
    represented_values = {
        fact.value for fact in targets if fact.value != ("none",)
    }
    observable_values = {
        fact.value
        for fact in gold_candidates | db_candidates
        if fact.kind == "communicate_value"
    }
    communicate_only = not targets
    targets.update(
        _RetailFact("communicate_value", (), value)
        for value in references.communicate_values
        if value not in represented_values
        and (value in observable_values or communicate_only)
    )
    return tuple(
        sorted(
            targets,
            key=lambda fact: (fact.kind, fact.subject, repr(fact.value)),
        )
    )


def _fact_references(task: Task) -> _FactReferences:
    references = _FactReferences(
        set(), set(), set(), set(), set(), set(), set()
    )
    criteria = task.evaluation_criteria
    actions = criteria.actions or [] if criteria is not None else []
    for action in actions:
        for name, value in action.arguments.items():
            _add_argument_reference(references, action.name, name, value)

    if criteria is not None:
        for value in criteria.communicate_info or []:
            references.communicate_values.add(_normalize_communicate_value(value))
    return references


def _add_argument_reference(
    references: _FactReferences,
    action_name: str,
    name: str,
    value: Any,
) -> None:
    destinations = {
        "user_id": references.user_ids,
        "order_id": references.order_ids,
        "product_id": references.product_ids,
        "item_ids": references.order_item_ids,
        "new_item_ids": references.catalog_item_ids,
        "payment_method_id": references.payment_method_ids,
    }
    if name == "item_id" and action_name == "get_item_details":
        destinations[name] = references.catalog_item_ids
    destination = destinations.get(name)
    if destination is None:
        return
    values = value if _is_sequence(value) else (value,)
    destination.update(str(item) for item in values)


def _facts_for_referenced_db(
    db: RetailDB,
    references: _FactReferences,
) -> set[_RetailFact]:
    """Recover observable prerequisites for tasks without gold reads."""

    _expand_references_from_db(db, references)
    facts: set[_RetailFact] = {
        _fact("authenticated_user", user_id) for user_id in references.user_ids
    }

    for user_id in references.user_ids:
        user = db.users.get(user_id)
        if user is not None:
            facts.update(
                _facts_from_tool_result(
                    "get_user_details",
                    {"user_id": user_id},
                    user,
                )
            )
    for order_id in references.order_ids:
        order = db.orders.get(order_id)
        if order is not None:
            facts.update(
                _facts_from_tool_result(
                    "get_order_details",
                    {"order_id": order_id},
                    order,
                )
            )
    for product_id in references.product_ids:
        product = db.products.get(product_id)
        if product is not None:
            facts.update(
                _facts_from_tool_result(
                    "get_product_details",
                    {"product_id": product_id},
                    product,
                )
            )
    for item_id in references.catalog_item_ids:
        variant = _variant_from_db(db, item_id)
        if variant is not None:
            facts.update(
                _facts_from_tool_result(
                    "get_item_details",
                    {"item_id": item_id},
                    variant,
                )
            )
    return facts


def _expand_references_from_db(
    db: RetailDB,
    references: _FactReferences,
) -> None:
    for order_id in tuple(references.order_ids):
        order = db.orders.get(order_id)
        if order is None:
            continue
        references.user_ids.add(order.user_id)

    for user_id, user in db.users.items():
        if references.payment_method_ids & user.payment_methods.keys():
            references.user_ids.add(user_id)

def _variant_from_db(db: RetailDB, item_id: str) -> Any | None:
    for product in db.products.values():
        variant = product.variants.get(item_id)
        if variant is not None:
            return variant
    return None


def _fact_is_relevant(
    fact: _RetailFact,
    references: _FactReferences,
) -> bool:
    """Keep compact prerequisites and explicitly requested fact values."""

    if fact.kind in {"authenticated_user", "user_known"}:
        return fact.subject[0] in references.user_ids
    if fact.kind in {"order_known", "order_status", "order_user"}:
        return fact.subject[0] in references.order_ids
    if fact.kind == "user_order":
        return fact.subject[1] in references.order_ids
    if fact.kind in {"user_payment_method", "payment_source", "payment_balance"}:
        return fact.subject[-1] in references.payment_method_ids
    if fact.kind in {
        "order_item",
        "order_item_product",
        "order_item_price",
        "order_item_option",
    }:
        return fact.subject[1] in references.order_item_ids
    if fact.kind == "order_payment_method":
        return fact.subject[1] in references.payment_method_ids
    if fact.kind == "order_tracking":
        return False
    if fact.kind in {
        "product_known",
        "product_name",
        "product_variant_count",
    }:
        return fact.subject[0] in references.product_ids
    if fact.kind in {
        "item_known",
        "item_available",
        "item_price",
        "item_option",
    }:
        return fact.subject[0] in references.catalog_item_ids
    if fact.kind == "product_item":
        return (
            fact.subject[0] in references.product_ids
            and fact.subject[1] in references.catalog_item_ids
        )
    if fact.kind == "calculation_result":
        return bool(references.communicate_values)
    return False


def _fact_is_write_prerequisite(
    fact: _RetailFact,
    references: _FactReferences,
) -> bool:
    """Keep only DB facts that justify a downstream reference write."""

    if fact.kind == "authenticated_user":
        return fact.subject[0] in references.user_ids
    if fact.kind in {"order_known", "order_status", "order_user"}:
        return fact.subject[0] in references.order_ids
    if fact.kind in {
        "order_item",
        "order_item_product",
        "order_item_price",
        "order_item_option",
    }:
        return fact.subject[1] in references.order_item_ids
    if fact.kind == "order_payment_method":
        return fact.subject[1] in references.payment_method_ids
    if fact.kind in {
        "item_known",
        "item_available",
        "item_price",
        "item_option",
    }:
        return fact.subject[0] in references.catalog_item_ids
    return False


def _facts_from_tool_result(
    tool_name: str,
    arguments: Mapping[str, Any],
    result: Any,
) -> set[_RetailFact]:
    value = _decode_tool_result(result)
    if tool_name == "calculate":
        facts = {
            _fact(
                "calculation_result",
                str(arguments.get("expression", "")),
                value=value,
            )
        }
    elif tool_name in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
        facts = {_fact("authenticated_user", str(value))}
    elif tool_name == "get_user_details" and isinstance(value, Mapping):
        facts = _user_facts(value, arguments)
    elif tool_name == "get_order_details" and isinstance(value, Mapping):
        facts = _order_facts(value, arguments)
    elif tool_name == "get_product_details" and isinstance(value, Mapping):
        facts = _product_facts(value, arguments)
    elif tool_name == "get_item_details" and isinstance(value, Mapping):
        facts = _item_facts(value, arguments)
    elif tool_name == "list_all_product_types" and isinstance(value, Mapping):
        facts = {
            _fact("product_name", str(product_id), value=name)
            for name, product_id in value.items()
        } | {
            _fact("product_known", str(product_id))
            for product_id in value.values()
        }
    else:
        facts = set()
    facts.update(_communicate_value_facts(value))
    return facts


def _communicate_value_facts(value: Any) -> set[_RetailFact]:
    if isinstance(value, Mapping):
        facts: set[_RetailFact] = set()
        for key, item in value.items():
            facts.update(_communicate_value_facts(key))
            facts.update(_communicate_value_facts(item))
        return facts
    if _is_sequence(value):
        facts = set()
        for item in value:
            facts.update(_communicate_value_facts(item))
        return facts
    if value is None:
        return set()
    normalized = _normalize_communicate_value(value)
    facts = {
        _RetailFact(
            "communicate_value",
            (),
            normalized,
        )
    }
    if normalized[0] == "string":
        facts.update(
            _RetailFact(
                "communicate_value",
                (),
                ("string", token),
            )
            for token in re.findall(r"[a-z0-9]+", normalized[1])
        )
    return facts


def _normalize_communicate_value(value: Any) -> NormalizedValue:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        if isinstance(decoded, (int, float, bool)):
            value = decoded
        else:
            value = value.strip().casefold()
    return _normalize_value(value)


def _user_facts(
    user: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> set[_RetailFact]:
    user_id = str(user.get("user_id") or arguments.get("user_id"))
    facts = {_fact("user_known", user_id)}
    facts.update(
        _fact("user_order", user_id, str(order_id))
        for order_id in user.get("orders") or []
    )
    for payment_id, payment in (user.get("payment_methods") or {}).items():
        payment_id = str(payment_id)
        facts.add(_fact("user_payment_method", user_id, payment_id))
        if isinstance(payment, Mapping):
            if "source" in payment:
                facts.add(
                    _fact("payment_source", payment_id, value=payment["source"])
                )
            if "balance" in payment:
                facts.add(
                    _fact("payment_balance", payment_id, value=payment["balance"])
                )
    facts.update(_profile_facts(user_id, user))
    return facts


def _profile_facts(user_id: str, user: Mapping[str, Any]) -> set[_RetailFact]:
    facts: set[_RetailFact] = set()
    for field in ("email",):
        if field in user:
            facts.add(_fact(f"user_{field}", user_id, value=user[field]))
    for container_name in ("name", "address"):
        container = user.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for field, value in container.items():
            facts.add(
                _fact(f"user_{container_name}", user_id, str(field), value=value)
            )
    return facts


def _order_facts(
    order: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> set[_RetailFact]:
    order_id = str(order.get("order_id") or arguments.get("order_id"))
    facts = {_fact("order_known", order_id)}
    if "status" in order:
        facts.add(_fact("order_status", order_id, value=order["status"]))
    if "user_id" in order:
        facts.add(_fact("order_user", order_id, value=order["user_id"]))

    for item in order.get("items") or []:
        if not isinstance(item, Mapping) or "item_id" not in item:
            continue
        item_id = str(item["item_id"])
        facts.add(_fact("order_item", order_id, item_id))
        if "product_id" in item:
            facts.add(
                _fact(
                    "order_item_product",
                    order_id,
                    item_id,
                    value=item["product_id"],
                )
            )
        if "price" in item:
            facts.add(
                _fact("order_item_price", order_id, item_id, value=item["price"])
            )
        for option, option_value in (item.get("options") or {}).items():
            facts.add(
                _fact(
                    "order_item_option",
                    order_id,
                    item_id,
                    str(option),
                    value=option_value,
                )
            )

    for payment in order.get("payment_history") or []:
        if isinstance(payment, Mapping) and "payment_method_id" in payment:
            facts.add(
                _fact(
                    "order_payment_method",
                    order_id,
                    str(payment["payment_method_id"]),
                )
            )
    for fulfillment in order.get("fulfillments") or []:
        if not isinstance(fulfillment, Mapping):
            continue
        facts.update(
            _fact("order_tracking", order_id, str(tracking_id))
            for tracking_id in fulfillment.get("tracking_id") or []
        )
    return facts


def _product_facts(
    product: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> set[_RetailFact]:
    product_id = str(product.get("product_id") or arguments.get("product_id"))
    facts = {_fact("product_known", product_id)}
    if "name" in product:
        facts.add(_fact("product_name", product_id, value=product["name"]))
    variants = product.get("variants") or {}
    facts.add(_fact("product_variant_count", product_id, value=len(variants)))
    facts.add(
        _RetailFact(
            "communicate_value",
            (),
            _normalize_communicate_value(len(variants)),
        )
    )
    for item_id, item in variants.items():
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("item_id") or item_id)
        facts.add(_fact("product_item", product_id, item_id))
        facts.update(_item_facts(item, {"item_id": item_id}))
    return facts


def _item_facts(
    item: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> set[_RetailFact]:
    item_id = str(item.get("item_id") or arguments.get("item_id"))
    facts = {_fact("item_known", item_id)}
    if "available" in item:
        facts.add(_fact("item_available", item_id, value=item["available"]))
    if "price" in item:
        facts.add(_fact("item_price", item_id, value=item["price"]))
    for option, option_value in (item.get("options") or {}).items():
        facts.add(
            _fact("item_option", item_id, str(option), value=option_value)
        )
    return facts


def _decode_tool_result(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump(mode="python")
    if not isinstance(result, str):
        return result
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return result


def _fact(
    kind: str,
    *subject: str,
    value: Any = None,
) -> _RetailFact:
    return _RetailFact(
        kind=kind,
        subject=tuple(map(str, subject)),
        value=_normalize_value(value),
    )


def _measure_target_fields(
    target_fields: tuple[_TargetField, ...],
    db: RetailDB,
) -> float:
    if not target_fields:
        return 0.0
    matched = sum(
        _normalize_value(_value_at_path(db, field.path)) == field.value
        for field in target_fields
    )
    return matched / len(target_fields)


def _set_initial_state(environment: Environment, task: Task) -> None:
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


def _environment_db(environment: Environment) -> RetailDB:
    if environment.tools is None or not isinstance(environment.tools.db, RetailDB):
        raise TypeError("Retail progress requires an environment with a RetailDB")
    return environment.tools.db


def _changed_target_fields(
    initial_db: RetailDB,
    gold_db: RetailDB,
) -> tuple[_TargetField, ...]:
    # Full-tree traversal happens once here.  ``observe`` retains only the
    # changed paths and never calls model_dump on the live DB.
    initial_leaves = dict(_iter_leaves(initial_db.model_dump(mode="python")))
    gold_leaves = dict(_iter_leaves(gold_db.model_dump(mode="python")))

    targets = [
        _TargetField(path=path, value=value)
        for path, value in gold_leaves.items()
        if initial_leaves.get(path, _MISSING_VALUE) != value
    ]

    gold_paths = tuple(gold_leaves)
    for path in initial_leaves.keys() - gold_leaves.keys():
        # Include real deletions.  If the container merely changed shape, its
        # gold ancestor/descendants already describe the target structure.
        if any(_paths_overlap(path, gold_path) for gold_path in gold_paths):
            continue
        targets.append(_TargetField(path=path, value=_MISSING_VALUE))

    targets.sort(key=lambda field: tuple(map(str, field.path)))
    return tuple(targets)


def _iter_leaves(
    value: Any,
    path: FieldPath = (),
):
    if isinstance(value, Mapping):
        if not value:
            yield path, _normalize_value(value)
            return
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            yield from _iter_leaves(value[key], (*path, key))
        return

    if _is_sequence(value):
        if not value:
            yield path, _normalize_value(value)
            return
        for index, item in enumerate(value):
            yield from _iter_leaves(item, (*path, index))
        return

    yield path, _normalize_value(value)


def _value_at_path(root: Any, path: FieldPath) -> Any:
    value = root
    for part in path:
        if isinstance(value, BaseModel) and isinstance(part, str):
            value = getattr(value, part, _MISSING)
        elif isinstance(value, Mapping):
            value = value.get(part, _MISSING)
        elif _is_sequence(value) and isinstance(part, int):
            value = value[part] if 0 <= part < len(value) else _MISSING
        else:
            return _MISSING
        if value is _MISSING:
            return _MISSING
    return value


def _normalize_value(value: Any) -> NormalizedValue:
    """Convert supported DB values to stable, immutable comparison values."""

    if value is _MISSING:
        return _MISSING_VALUE
    if isinstance(value, BaseModel):
        return _normalize_value(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return _normalize_value(value.value)
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float, Decimal)):
        if isinstance(value, float) and not math.isfinite(value):
            return ("number", repr(value))
        return ("number", str(Decimal(str(value)).normalize()))
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, (datetime, date)):
        return ("datetime", value.isoformat())
    if isinstance(value, Mapping):
        items = sorted(
            value.items(),
            key=lambda item: (type(item[0]).__name__, repr(item[0])),
        )
        return (
            "mapping",
            tuple(
                (_normalize_value(key), _normalize_value(item)) for key, item in items
            ),
        )
    if _is_sequence(value):
        return ("sequence", tuple(_normalize_value(item) for item in value))
    return (type(value).__qualname__, repr(value))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _paths_overlap(left: FieldPath, right: FieldPath) -> bool:
    common_length = min(len(left), len(right))
    return left[:common_length] == right[:common_length]


__all__ = [
    "RetailDBProgressTracker",
    "RetailFactProgressObservation",
    "RetailFactProgressTracker",
    "RetailProgressObservation",
]
