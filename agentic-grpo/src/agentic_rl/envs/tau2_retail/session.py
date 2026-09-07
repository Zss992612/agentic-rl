"""Stateful τ²-bench Retail session for externally generated agent turns.

The session owns one independent task environment and user simulator.  It keeps
τ²'s message and evaluation contracts intact while leaving agent generation and
concurrency to the future VERL ``AgentLoop`` integration.
"""

from __future__ import annotations

import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    UserMessage,
)
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.domains.retail.environment import get_environment
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import HalfDuplexUser
from tau2.utils.utils import get_now


class RetailProtocolError(ValueError):
    """Raised when an externally generated turn violates the τ² protocol."""


class RetailInfrastructureError(RuntimeError):
    """Raised when the external user simulator cannot produce a response."""


USER_REQUEST_ATTEMPTS = 2


@dataclass(frozen=True)
class RetailSessionTurn:
    """Messages produced after one externally generated assistant message."""

    messages: tuple[Message, ...]
    done: bool
    termination_reason: TerminationReason | None


class RetailSession:
    """Own one isolated τ² Retail environment and conversation trajectory.

    ``start`` generates the task's first simulated-user message.  Thereafter,
    ``step`` accepts exactly one assistant message and returns either its tool
    results or the next simulated-user message.  Calls are synchronous by
    design; the VERL adapter is responsible for moving user generation to a
    worker thread so its event loop remains responsive.
    """

    domain = "retail"

    def __init__(
        self,
        *,
        task: Task,
        user: HalfDuplexUser,
        environment: Environment | None = None,
        max_steps: int = 100,
        max_errors: int = 10,
        timeout: float | None = None,
        seed: int | None = None,
        simulation_id: str | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_errors <= 0:
            raise ValueError("max_errors must be positive")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive when provided")

        self.task = task
        self.user = user
        self.environment = environment or get_environment()
        if self.environment.get_domain_name() != self.domain:
            raise ValueError(
                "RetailSession requires a retail environment, got "
                f"{self.environment.get_domain_name()!r}"
            )

        self.max_steps = max_steps
        self.max_errors = max_errors
        self.timeout = timeout
        self.seed = seed
        self.simulation_id = simulation_id or str(uuid.uuid4())

        self.messages: list[Message] = []
        self.user_state: Any | None = None
        self.step_count = 0
        self.num_errors = 0
        self.done = False
        self.termination_reason: TerminationReason | None = None
        self._started = False
        self._start_time: str | None = None
        self._start_perf: float | None = None

    @classmethod
    def create(
        cls,
        *,
        task: Task,
        user_llm: str,
        user_llm_args: dict[str, Any] | None = None,
        max_steps: int = 100,
        max_errors: int = 10,
        timeout: float | None = None,
        seed: int | None = None,
        simulation_id: str | None = None,
    ) -> "RetailSession":
        """Construct a session with τ²'s standard text user simulator."""

        environment = get_environment()
        user = UserSimulator(
            llm=user_llm,
            llm_args=user_llm_args,
            instructions=str(task.user_scenario),
            tools=None,
        )
        return cls(
            task=task,
            user=user,
            environment=environment,
            max_steps=max_steps,
            max_errors=max_errors,
            timeout=timeout,
            seed=seed,
            simulation_id=simulation_id,
        )

    @property
    def policy(self) -> str:
        """Return the authoritative Retail policy used in the model prompt."""

        return self.environment.get_policy()

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return complete OpenAI-format schemas without simplifying JSON Schema."""

        return [deepcopy(tool.openai_schema) for tool in self.environment.get_tools()]

    def start(self) -> tuple[Message, ...]:
        """Initialize task state and generate the first simulated-user message."""

        if self._started:
            raise RuntimeError("RetailSession.start() may only be called once")

        initial_state = self.task.initial_state
        message_history = (
            deepcopy(initial_state.message_history)
            if initial_state is not None and initial_state.message_history is not None
            else []
        )
        if message_history:
            raise NotImplementedError(
                "Preloaded message history is not supported by the Retail RL adapter"
            )

        self.environment.set_state(
            initialization_data=(
                initial_state.initialization_data if initial_state is not None else None
            ),
            initialization_actions=(
                initial_state.initialization_actions
                if initial_state is not None
                else None
            ),
            message_history=message_history,
        )
        if self.seed is not None:
            self.user.set_seed(self.seed)

        self._started = True
        self._start_time = get_now()
        self._start_perf = time.perf_counter()
        self.user_state = self.user.get_init_state()

        greeting = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
        user_message = self._generate_user_message(greeting)
        self.messages.extend((greeting, user_message))
        self.step_count = 1

        if UserSimulator.is_stop(user_message):
            self.terminate(TerminationReason.USER_STOP)
        else:
            self._apply_limits()
        return (greeting, user_message)

    def step(self, message: AssistantMessage) -> RetailSessionTurn:
        """Apply one externally generated assistant turn to the τ² session."""

        self._require_active()
        self._validate_assistant_message(message)
        self.messages.append(message)
        self.step_count += 1

        produced: list[Message]
        if message.is_tool_call():
            produced = []
            for tool_call in message.tool_calls or []:
                tool_message = self.environment.get_response(tool_call)
                produced.append(tool_message)
                if tool_message.error:
                    self.num_errors += 1
        else:
            user_message = self._generate_user_message(message)
            produced = [user_message]

        self.messages.extend(produced)
        self.step_count += 1
        self.environment.sync_tools()

        if produced and isinstance(produced[-1], UserMessage):
            if UserSimulator.is_stop(produced[-1]):
                self.terminate(TerminationReason.USER_STOP)
        self._apply_limits()

        return RetailSessionTurn(
            messages=tuple(produced),
            done=self.done,
            termination_reason=self.termination_reason,
        )

    def terminate(self, reason: TerminationReason) -> None:
        """End the session with an explicit τ² termination classification."""

        if not self._started:
            raise RuntimeError("Cannot terminate a session before start()")
        self.done = True
        self.termination_reason = reason

    def to_simulation_run(self) -> SimulationRun:
        """Build the τ² simulation object consumed by official evaluators."""

        if not self.done or self.termination_reason is None:
            raise RuntimeError("The session must terminate before it can be finalized")
        if self._start_time is None or self._start_perf is None:
            raise RuntimeError("The session has not been started")

        trajectory = deepcopy(self.messages)
        for turn_idx, message in enumerate(trajectory):
            message.turn_idx = turn_idx
        return SimulationRun(
            id=self.simulation_id,
            task_id=self.task.id,
            start_time=self._start_time,
            end_time=get_now(),
            duration=time.perf_counter() - self._start_perf,
            termination_reason=self.termination_reason,
            messages=trajectory,
            seed=self.seed,
            mode=CommunicationMode.HALF_DUPLEX.value,
        )

    def evaluate(self) -> RewardInfo:
        """Evaluate with the same environment-only binary signal as SFT eval."""

        return evaluate_simulation(
            simulation=self.to_simulation_run(),
            task=self.task,
            evaluation_type=EvaluationType.ENV,
            solo_mode=False,
            domain=self.domain,
            mode=CommunicationMode.HALF_DUPLEX,
        )

    def _require_active(self) -> None:
        if not self._started:
            raise RuntimeError("Call RetailSession.start() before step()")
        if self.done:
            raise RuntimeError("Cannot step a terminated RetailSession")

    @staticmethod
    def _validate_assistant_message(message: AssistantMessage) -> None:
        message.validate()
        if message.has_text_content() and message.is_tool_call():
            raise RetailProtocolError(
                "assistant messages cannot contain both text and tool calls"
            )
        if any(
            tool_call.requestor != "assistant" for tool_call in message.tool_calls or []
        ):
            raise RetailProtocolError(
                "assistant tool calls must use requestor='assistant'"
            )

    @staticmethod
    def _validate_user_message(message: UserMessage) -> None:
        message.validate()
        if message.has_text_content() and message.is_tool_call():
            raise RetailProtocolError(
                "user messages cannot contain both text and tool calls"
            )
        if message.is_tool_call():
            raise RetailProtocolError("Retail user simulator cannot call tools")

    def _generate_user_message(
        self,
        assistant_message: AssistantMessage,
    ) -> UserMessage:
        """Generate a valid user turn without committing state from failed attempts."""

        previous_state = deepcopy(self.user_state)
        last_error: Exception | None = None
        for _ in range(USER_REQUEST_ATTEMPTS):
            try:
                user_message, next_state = self.user.generate_next_message(
                    assistant_message,
                    deepcopy(previous_state),
                )
                self._validate_user_message(user_message)
            except Exception as exc:
                last_error = exc
                continue

            self.user_state = next_state
            return user_message

        raise RetailInfrastructureError(
            "User simulator failed or returned an invalid message after "
            f"{USER_REQUEST_ATTEMPTS} attempts"
        ) from last_error

    def _apply_limits(self) -> None:
        if self.done:
            return
        if self.num_errors >= self.max_errors:
            self.terminate(TerminationReason.TOO_MANY_ERRORS)
            return
        if self.step_count >= self.max_steps:
            self.terminate(TerminationReason.MAX_STEPS)
            return
        if self.timeout is not None and self._start_perf is not None:
            if time.perf_counter() - self._start_perf >= self.timeout:
                self.terminate(TerminationReason.TIMEOUT)
