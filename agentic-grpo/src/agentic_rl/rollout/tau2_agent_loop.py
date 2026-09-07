"""VERL AgentLoop for τ²-bench Retail rollouts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import Task
from tau2.domains.retail.environment import get_tasks
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser

from agentic_rl.envs.tau2_retail import RetailProtocolError, RetailSession
from agentic_rl.envs.tau2_retail.progress import (
    RetailDBProgressTracker,
    RetailFactProgressTracker,
)
from agentic_rl.envs.tau2_retail.session import RetailInfrastructureError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _retail_tasks() -> dict[str, Task]:
    return {task.id: task for task in get_tasks(task_split_name=None)}


def _slot_id(global_step: int, sample_index: int, rollout_index: int) -> str:
    return (
        f"step_{global_step:06d}:"
        f"sample_{sample_index:06d}:"
        f"rollout_{rollout_index:06d}"
    )


class Tau2RetailAgentLoop(AgentLoopBase):
    """Run one policy trajectory against an isolated τ² Retail session."""

    def __init__(
        self,
        *args: Any,
        user_llm: str,
        user_llm_args: dict[str, Any] | None = None,
        max_steps: int = 100,
        max_errors: int = 10,
        max_assistant_tokens_per_turn: int = 2048,
        max_infrastructure_retries: int = 1,
        enable_turn_credit: bool = False,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        self.user_llm = user_llm
        self.user_llm_args = (
            OmegaConf.to_container(user_llm_args, resolve=True)
            if OmegaConf.is_config(user_llm_args)
            else user_llm_args or {}
        )
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.max_assistant_tokens_per_turn = max_assistant_tokens_per_turn
        self.max_infrastructure_retries = max_infrastructure_retries
        self.enable_turn_credit = enable_turn_credit
        self.timeout = timeout
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.calculate_log_probs = self.rollout_config.calculate_log_probs
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.tool_parser = ToolParser.get_tool_parser("hermes", self.tokenizer)

    async def run(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput:
        trajectory_info = kwargs.get("trajectory_info") or {}
        for retry_count in range(self.max_infrastructure_retries + 1):
            request_id = uuid4().hex
            started = time.perf_counter()
            try:
                return await self._run_once(
                    sampling_params,
                    request_id=request_id,
                    infrastructure_retry_count=retry_count,
                    **kwargs,
                )
            except RetailInfrastructureError as exc:
                elapsed_seconds = time.perf_counter() - started
                self._write_infrastructure_error(
                    request_id=request_id,
                    task_id=str(kwargs["task_id"]),
                    trajectory_info=trajectory_info,
                    retry_count=retry_count,
                    elapsed_seconds=elapsed_seconds,
                    error=exc,
                )
                logger.warning(
                    "Retail user simulator infrastructure failure for task %s "
                    "on attempt %s/%s after %.2fs",
                    kwargs["task_id"],
                    retry_count + 1,
                    self.max_infrastructure_retries + 1,
                    elapsed_seconds,
                )
                if retry_count == self.max_infrastructure_retries:
                    raise

        raise RuntimeError("Unreachable infrastructure retry state")

    async def _run_once(
        self,
        sampling_params: dict[str, Any],
        *,
        request_id: str,
        infrastructure_retry_count: int,
        **kwargs: Any,
    ) -> AgentLoopOutput:
        trajectory_started = time.perf_counter()
        task_id = str(kwargs["task_id"])
        trajectory_info = kwargs.get("trajectory_info") or {}
        global_step = int(trajectory_info.get("step", -1))
        sample_index = int(trajectory_info.get("sample_index", -1))
        rollout_index = int(trajectory_info.get("rollout_n", -1))
        slot_id = _slot_id(global_step, sample_index, rollout_index)
        is_validation = bool(trajectory_info.get("validate", False))
        task = _retail_tasks()[task_id]
        tau2_seed = kwargs.get("tau2_seed")
        session = RetailSession.create(
            task=task,
            user_llm=self.user_llm,
            user_llm_args=self.user_llm_args,
            max_steps=self.max_steps,
            max_errors=self.max_errors,
            timeout=self.timeout,
            seed=None if tau2_seed is None else int(tau2_seed),
        )

        started = time.perf_counter()
        initial_messages = await asyncio.to_thread(session.start)
        initial_user_seconds = time.perf_counter() - started
        write_progress_tracker = (
            RetailDBProgressTracker.from_task(task)
            if self.enable_turn_credit
            else None
        )
        fact_progress_tracker = (
            RetailFactProgressTracker.from_task(task)
            if self.enable_turn_credit
            else None
        )
        chat_messages = [
            {"role": "system", "content": session.policy},
            *[self._to_chat_message(message) for message in initial_messages],
        ]
        prompt_ids = await self.apply_chat_template(
            chat_messages,
            tools=session.tool_schemas,
        )
        if len(prompt_ids) > self.prompt_length:
            raise ValueError(
                f"Initial prompt has {len(prompt_ids)} tokens, exceeding "
                f"prompt_length={self.prompt_length}"
            )

        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        assistant_turns = 0
        user_turns = 1
        tool_call_count = 0
        user_observation_tokens = 0
        tool_response_tokens = 0
        generation_seconds = 0.0
        tool_seconds = 0.0
        user_seconds = initial_user_seconds
        num_preempted = 0
        turn_records: list[dict[str, Any]] = []
        turn_credit_spans: list[dict[str, Any]] = []

        def add_turn_credit(
            *,
            start: int,
            end: int,
            new_progress: float = 0.0,
            new_fact_progress: float = 0.0,
            model_error: bool = False,
        ) -> None:
            if not self.enable_turn_credit:
                return
            turn_credit_spans.append(
                {
                    "start": start,
                    "end": end,
                    "new_progress": new_progress,
                    "new_fact_progress": new_fact_progress,
                    "model_error": model_error,
                }
            )

        while not session.done:
            if (
                self.max_assistant_turns is not None
                and assistant_turns >= self.max_assistant_turns
            ):
                session.terminate(TerminationReason.MAX_STEPS)
                break

            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                session.terminate(TerminationReason.CONTEXT_WINDOW_EXCEEDED)
                break

            turn_token_limit = min(remaining, self.max_assistant_tokens_per_turn)
            turn_sampling_params = dict(sampling_params)
            turn_sampling_params["max_tokens"] = turn_token_limit
            started = time.perf_counter()
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids + response_ids,
                sampling_params=turn_sampling_params,
            )
            turn_generation_seconds = time.perf_counter() - started
            generation_seconds += turn_generation_seconds
            num_preempted += output.num_preempted or 0
            assistant_turns += 1

            turn_token_start = len(response_ids)
            generated_ids = output.token_ids[:turn_token_limit]
            response_ids.extend(generated_ids)
            turn_token_end = len(response_ids)
            response_mask.extend([1] * len(generated_ids))
            if self.calculate_log_probs:
                if output.log_probs is None or len(output.log_probs) < len(generated_ids):
                    raise RuntimeError("vLLM did not return log-probs for every generated token")
                response_logprobs.extend(output.log_probs[: len(generated_ids)])

            if len(generated_ids) == turn_token_limit:
                add_turn_credit(
                    start=turn_token_start,
                    end=turn_token_end,
                    model_error=True,
                )
                turn_records.append(
                    {
                        "assistant_turn": assistant_turns,
                        "token_start": turn_token_start,
                        "token_end": turn_token_end,
                        "generated_tokens": len(generated_ids),
                        "observation_tokens": 0,
                        "generation_seconds": turn_generation_seconds,
                        "action": "length_limit",
                        "action_seconds": 0.0,
                        "tool_name": None,
                        "tool_call_count": 0,
                        "new_progress": 0.0,
                        "new_fact_progress": 0.0,
                        "model_error": True,
                    }
                )
                reason = (
                    TerminationReason.CONTEXT_WINDOW_EXCEEDED
                    if turn_token_limit == remaining
                    else TerminationReason.AGENT_ERROR
                )
                session.terminate(reason)
                break

            try:
                assistant_message = await self._parse_assistant_message(generated_ids)
            except RetailProtocolError:
                add_turn_credit(
                    start=turn_token_start,
                    end=turn_token_end,
                    model_error=True,
                )
                turn_records.append(
                    {
                        "assistant_turn": assistant_turns,
                        "token_start": turn_token_start,
                        "token_end": turn_token_end,
                        "generated_tokens": len(generated_ids),
                        "observation_tokens": 0,
                        "generation_seconds": turn_generation_seconds,
                        "action": "protocol_error",
                        "action_seconds": 0.0,
                        "tool_name": None,
                        "tool_call_count": 0,
                        "new_progress": 0.0,
                        "new_fact_progress": 0.0,
                        "model_error": True,
                    }
                )
                session.terminate(TerminationReason.AGENT_ERROR)
                break

            try:
                started = time.perf_counter()
                if assistant_message.is_tool_call():
                    action = "tool"
                    tool_name = assistant_message.tool_calls[0].name
                    turn_tool_call_count = len(assistant_message.tool_calls)
                    turn = session.step(assistant_message)
                    action_seconds = time.perf_counter() - started
                    tool_seconds += action_seconds
                    tool_call_count += turn_tool_call_count
                else:
                    action = "user"
                    tool_name = None
                    turn_tool_call_count = 0
                    turn = await asyncio.to_thread(session.step, assistant_message)
                    action_seconds = time.perf_counter() - started
                    user_seconds += action_seconds
                    user_turns += 1
            except RetailProtocolError:
                add_turn_credit(
                    start=turn_token_start,
                    end=turn_token_end,
                    model_error=True,
                )
                turn_records.append(
                    {
                        "assistant_turn": assistant_turns,
                        "token_start": turn_token_start,
                        "token_end": turn_token_end,
                        "generated_tokens": len(generated_ids),
                        "observation_tokens": 0,
                        "generation_seconds": turn_generation_seconds,
                        "action": "session_error",
                        "action_seconds": time.perf_counter() - started,
                        "tool_name": None,
                        "tool_call_count": 0,
                        "new_progress": 0.0,
                        "new_fact_progress": 0.0,
                        "model_error": True,
                    }
                )
                session.terminate(TerminationReason.USER_ERROR)
                break

            tool_error_count = sum(
                isinstance(message, ToolMessage) and bool(message.error)
                for message in turn.messages
            )
            model_error = tool_error_count > 0
            potential_before = (
                write_progress_tracker.current_potential
                if write_progress_tracker is not None
                else None
            )
            new_progress = 0.0
            potential_after = potential_before
            if write_progress_tracker is not None and action == "tool":
                progress_observation = write_progress_tracker.observe(
                    session.environment.tools.db
                )
                potential_after = progress_observation.potential
                new_progress = progress_observation.progress

            fact_potential_before = (
                fact_progress_tracker.current_potential
                if fact_progress_tracker is not None
                else None
            )
            fact_potential_after = fact_potential_before
            new_fact_progress = 0.0
            new_fact_count = 0
            if fact_progress_tracker is not None and action == "tool":
                tool_messages = {
                    message.id: message
                    for message in turn.messages
                    if isinstance(message, ToolMessage)
                }
                for tool_call in assistant_message.tool_calls or []:
                    tool_message = tool_messages[tool_call.id]
                    fact_observation = fact_progress_tracker.observe_tool_call(
                        tool_call.name,
                        tool_call.arguments,
                        tool_message.content,
                        error=tool_message.error,
                    )
                    new_fact_progress += fact_observation.progress
                    new_fact_count += fact_observation.new_fact_count
                fact_potential_after = fact_progress_tracker.current_potential

            add_turn_credit(
                start=turn_token_start,
                end=turn_token_end,
                new_progress=new_progress,
                new_fact_progress=new_fact_progress,
                model_error=model_error,
            )

            observation_ids = await self.apply_chat_template(
                [self._to_chat_message(message) for message in turn.messages],
                remove_system_prompt=True,
            )
            remaining = self.response_length - len(response_ids)
            added_observation_tokens = min(len(observation_ids), remaining)
            response_ids.extend(observation_ids[:remaining])
            response_mask.extend([0] * added_observation_tokens)
            if self.calculate_log_probs:
                response_logprobs.extend([0.0] * added_observation_tokens)
            if action == "tool":
                tool_response_tokens += added_observation_tokens
            else:
                user_observation_tokens += added_observation_tokens

            turn_records.append(
                {
                    "assistant_turn": assistant_turns,
                    "token_start": turn_token_start,
                    "token_end": turn_token_end,
                    "generated_tokens": len(generated_ids),
                    "observation_tokens": added_observation_tokens,
                    "generation_seconds": turn_generation_seconds,
                    "action": action,
                    "action_seconds": action_seconds,
                    "tool_name": tool_name,
                    "tool_call_count": turn_tool_call_count,
                    "tool_error_count": tool_error_count,
                    "potential_before": potential_before,
                    "potential_after": potential_after,
                    "new_progress": new_progress,
                    "fact_potential_before": fact_potential_before,
                    "fact_potential_after": fact_potential_after,
                    "new_fact_progress": new_fact_progress,
                    "new_fact_count": new_fact_count,
                    "model_error": model_error,
                }
            )

            if len(observation_ids) > remaining:
                session.terminate(TerminationReason.CONTEXT_WINDOW_EXCEEDED)

        started = time.perf_counter()
        reward_info = session.evaluate()
        evaluation_seconds = time.perf_counter() - started
        reward = reward_info.reward
        trajectory_seconds = time.perf_counter() - trajectory_started
        assistant_tokens = sum(response_mask)
        observation_tokens = len(response_ids) - assistant_tokens
        assistant_tokens_per_turn = assistant_tokens / assistant_turns if assistant_turns else 0.0
        tool_response_tokens_per_call = (
            tool_response_tokens / tool_call_count if tool_call_count else 0.0
        )
        turn_credit_summary = None
        if write_progress_tracker is not None and fact_progress_tracker is not None:
            turn_credit_summary = {
                "target_field_count": write_progress_tracker.target_field_count,
                "final_potential": write_progress_tracker.current_potential,
                "best_potential": write_progress_tracker.best_potential,
                "total_new_progress": sum(
                    float(span["new_progress"]) for span in turn_credit_spans
                ),
                "positive_progress_turns": sum(
                    float(span["new_progress"]) > 0.0
                    for span in turn_credit_spans
                ),
                "target_fact_count": fact_progress_tracker.target_fact_count,
                "covered_fact_count": fact_progress_tracker.covered_fact_count,
                "final_fact_potential": fact_progress_tracker.current_potential,
                "best_fact_potential": fact_progress_tracker.best_potential,
                "total_new_fact_progress": sum(
                    float(span["new_fact_progress"]) for span in turn_credit_spans
                ),
                "positive_fact_progress_turns": sum(
                    float(span["new_fact_progress"]) > 0.0
                    for span in turn_credit_spans
                ),
                "model_error_turns": sum(
                    bool(span["model_error"]) for span in turn_credit_spans
                ),
            }
        metrics = AgentLoopMetrics(
            generate_sequences=generation_seconds,
            tool_calls=tool_seconds,
            num_preempted=num_preempted,
        )
        trajectory_summary = {
            "request_id": request_id,
            "run_id": os.environ.get("AGENTIC_RL_RUN_ID"),
            "global_step": global_step,
            "sample_index": sample_index,
            "rollout_index": rollout_index,
            "slot_id": slot_id,
            "is_validation": is_validation,
            "task_id": task_id,
            "reward": reward,
            "reward_info": reward_info.model_dump(mode="json", exclude_none=True),
            "termination_reason": session.termination_reason.value,
            "infrastructure_retry_count": infrastructure_retry_count,
            "num_turns": len(session.messages),
            "assistant_turns": assistant_turns,
            "user_turns": user_turns,
            "tool_call_count": tool_call_count,
            "num_errors": session.num_errors,
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "assistant_tokens": assistant_tokens,
            "observation_tokens": observation_tokens,
            "user_observation_tokens": user_observation_tokens,
            "tool_response_tokens": tool_response_tokens,
            "assistant_tokens_per_turn": assistant_tokens_per_turn,
            "tool_response_tokens_per_call": tool_response_tokens_per_call,
            "trajectory_seconds": trajectory_seconds,
            "initial_user_seconds": initial_user_seconds,
            "generation_seconds": generation_seconds,
            "user_seconds": user_seconds,
            "tool_seconds": tool_seconds,
            "evaluation_seconds": evaluation_seconds,
            "num_preempted": num_preempted,
            "turn_records": turn_records,
            "messages": [
                self._serialize_message(turn_index, message)
                for turn_index, message in enumerate(session.messages)
            ],
        }
        if turn_credit_summary is not None:
            trajectory_summary["turn_credit"] = turn_credit_summary
            trajectory_summary["turn_credit_spans"] = turn_credit_spans
        trajectory_log_dir = os.environ.get("AGENTIC_RL_TRAJECTORY_LOG_DIR")
        if trajectory_log_dir:
            step_name = f"step_{global_step:06d}" if global_step >= 0 else "step_unknown"
            log_dir = Path(trajectory_log_dir) / step_name
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{request_id}.json"
            log_path.write_text(
                json.dumps(trajectory_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        reward_extra_info = {
            "termination_reason": session.termination_reason.value,
            "global_step": float(global_step),
            "sample_index": float(sample_index),
            "rollout_index": float(rollout_index),
            "infrastructure_retry_count": float(infrastructure_retry_count),
            "num_turns": float(len(session.messages)),
            "assistant_turns": float(assistant_turns),
            "user_turns": float(user_turns),
            "tool_call_count": float(tool_call_count),
            "num_errors": float(session.num_errors),
            "prompt_tokens": float(len(prompt_ids)),
            "response_tokens": float(len(response_ids)),
            "assistant_tokens": float(assistant_tokens),
            "observation_tokens": float(observation_tokens),
            "user_observation_tokens": float(user_observation_tokens),
            "tool_response_tokens": float(tool_response_tokens),
            "assistant_tokens_per_turn": assistant_tokens_per_turn,
            "tool_response_tokens_per_call": tool_response_tokens_per_call,
            "trajectory_seconds": trajectory_seconds,
            "initial_user_seconds": initial_user_seconds,
            "generation_seconds": generation_seconds,
            "user_seconds": user_seconds,
            "tool_seconds": tool_seconds,
            "evaluation_seconds": evaluation_seconds,
            "num_preempted": float(num_preempted),
        }
        if turn_credit_summary is not None:
            reward_extra_info.update(
                {
                    f"turn_credit_{key}": float(value)
                    for key, value in turn_credit_summary.items()
                }
            )

        extra_fields = {
            "reward_extra_info": reward_extra_info,
            "request_id": request_id,
            "task_id": task_id,
            "global_step": global_step,
            "sample_index": sample_index,
            "rollout_index": rollout_index,
            "slot_id": slot_id,
        }
        if self.enable_turn_credit:
            extra_fields["turn_credit_spans"] = turn_credit_spans
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if self.calculate_log_probs else None,
            reward_score=reward,
            num_turns=len(session.messages),
            metrics=metrics,
            extra_fields=extra_fields,
        )

    @staticmethod
    def _write_infrastructure_error(
        *,
        request_id: str,
        task_id: str,
        trajectory_info: dict[str, Any],
        retry_count: int,
        elapsed_seconds: float,
        error: RetailInfrastructureError,
    ) -> None:
        trajectory_log_dir = os.environ.get("AGENTIC_RL_TRAJECTORY_LOG_DIR")
        if not trajectory_log_dir:
            return

        global_step = int(trajectory_info.get("step", -1))
        sample_index = int(trajectory_info.get("sample_index", -1))
        rollout_index = int(trajectory_info.get("rollout_n", -1))
        step_name = f"step_{global_step:06d}" if global_step >= 0 else "step_unknown"
        error_dir = Path(trajectory_log_dir) / step_name / "infrastructure_errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        error_path = error_dir / f"{request_id}.json"
        error_path.write_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "run_id": os.environ.get("AGENTIC_RL_RUN_ID"),
                    "global_step": global_step,
                    "sample_index": sample_index,
                    "rollout_index": rollout_index,
                    "slot_id": _slot_id(global_step, sample_index, rollout_index),
                    "task_id": task_id,
                    "termination_reason": TerminationReason.INFRASTRUCTURE_ERROR.value,
                    "infrastructure_retry_count": retry_count,
                    "elapsed_seconds": elapsed_seconds,
                    "error_type": type(error.__cause__ or error).__name__,
                    "error_message": str(error.__cause__ or error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _serialize_message(turn_index: int, message: Message) -> dict[str, Any]:
        payload = message.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"audio_content", "raw_data"},
        )
        payload["turn_index"] = turn_index
        return payload

    async def _parse_assistant_message(self, token_ids: list[int]) -> AssistantMessage:
        content, function_calls = await self.tool_parser.extract_tool_calls(token_ids)
        if not function_calls:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            if "<tool_call>" in text or "</tool_call>" in text:
                raise RetailProtocolError("assistant generated a malformed tool call")
            if not text:
                raise RetailProtocolError("assistant generated an empty message")
            return AssistantMessage(role="assistant", content=text)

        visible_content = content
        for special_token in self.tokenizer.all_special_tokens:
            visible_content = visible_content.replace(special_token, "")
        if visible_content.strip():
            raise RetailProtocolError("assistant generated both text and tool calls")

        tool_calls = [self._to_tau2_tool_call(call) for call in function_calls]
        return AssistantMessage(role="assistant", tool_calls=tool_calls)

    @staticmethod
    def _to_tau2_tool_call(function_call: FunctionCall) -> ToolCall:
        arguments = json.loads(function_call.arguments)
        if not isinstance(arguments, dict):
            raise RetailProtocolError("tool arguments must be a JSON object")
        return ToolCall(
            id=uuid4().hex,
            name=function_call.name,
            arguments=arguments,
            requestor="assistant",
        )

    @staticmethod
    def _to_chat_message(message: Message) -> dict[str, Any]:
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolMessage):
            return {"role": "tool", "content": message.content or ""}
        if isinstance(message, AssistantMessage) and message.is_tool_call():
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in message.tool_calls or []
                ],
            }
        if isinstance(message, AssistantMessage):
            return {"role": "assistant", "content": message.content}
        raise TypeError(f"Unsupported τ² message type: {type(message).__name__}")
