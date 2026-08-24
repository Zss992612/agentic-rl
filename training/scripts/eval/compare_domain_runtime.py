import csv
import json
import time
from datetime import datetime
from math import ceil
from pathlib import Path
from statistics import mean, median

from dotenv import load_dotenv

TRAINING_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = TRAINING_DIR.parent
load_dotenv(WORKSPACE_DIR / "tau2-bench/.env")

MODEL = "deepseek/deepseek-v4-flash"
DOMAINS = ("retail", "airline")
REPORT_DIR = TRAINING_DIR / "results/domain_runtime_comparison"


def run_all_tasks(domain: str):
    from tau2 import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner import get_tasks, run_tasks
    from tau2.utils.utils import DATA_DIR

    run_name = f"runtime_compare_{domain}_deepseek_v4_flash"
    save_dir = DATA_DIR / "simulations" / run_name
    config = TextRunConfig(
        domain=domain,
        task_split_name="base",
        llm_agent=MODEL,
        llm_user=MODEL,
        num_trials=1,
        max_concurrency=1,
        save_to=run_name,
    )
    tasks = get_tasks(domain, task_split_name="base")
    start = time.perf_counter()
    results = run_tasks(
        config,
        tasks,
        save_path=save_dir / "results.json",
        save_dir=save_dir,
        evaluation_type=EvaluationType.ENV,
    )
    return results, time.perf_counter() - start


def token_count(messages: list[dict], role: str, field: str) -> int:
    return sum(
        (message.get("usage") or {}).get(field, 0)
        for message in messages
        if message.get("role") == role
    )


def make_row(domain: str, simulation) -> dict:
    data = simulation.model_dump(mode="json")
    messages = data["messages"]
    reward_info = data.get("reward_info") or {}
    return {
        "domain": domain,
        "task_id": data["task_id"],
        "trial": data["trial"],
        "seed": data["seed"],
        "duration_seconds": round(data["duration"], 3),
        "termination_reason": data["termination_reason"],
        "env_reward": reward_info.get("reward"),
        "assistant_messages": sum(m.get("role") == "assistant" for m in messages),
        "user_messages": sum(m.get("role") == "user" for m in messages),
        "tool_messages": sum(m.get("role") == "tool" for m in messages),
        "tool_calls": sum(len(m.get("tool_calls") or []) for m in messages),
        "tool_errors": sum(
            m.get("role") == "tool" and bool(m.get("error")) for m in messages
        ),
        "agent_input_tokens": token_count(messages, "assistant", "prompt_tokens"),
        "agent_output_tokens": token_count(
            messages, "assistant", "completion_tokens"
        ),
        "user_input_tokens": token_count(messages, "user", "prompt_tokens"),
        "user_output_tokens": token_count(messages, "user", "completion_tokens"),
        "agent_cost": data.get("agent_cost"),
        "user_cost": data.get("user_cost"),
    }


def percentile_90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[ceil(len(ordered) * 0.9) - 1]


def make_summary(rows: list[dict], batch_wall_seconds: float) -> dict:
    completed = [r for r in rows if r["termination_reason"] != "infrastructure_error"]
    durations = [r["duration_seconds"] for r in completed]
    return {
        "task_count": len(rows),
        "completed_count": len(completed),
        "infrastructure_errors": len(rows) - len(completed),
        "env_success_count": sum(r["env_reward"] == 1.0 for r in completed),
        "mean_duration_seconds": round(mean(durations), 3),
        "median_duration_seconds": round(median(durations), 3),
        "p90_duration_seconds": round(percentile_90(durations), 3),
        "batch_wall_seconds": round(batch_wall_seconds, 3),
    }


def main() -> None:
    all_rows = []
    summaries = {}
    for domain in DOMAINS:
        results, wall_seconds = run_all_tasks(domain)
        rows = [make_row(domain, simulation) for simulation in results.simulations]
        all_rows.extend(rows)
        summaries[domain] = make_summary(rows, wall_seconds)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "per_task_metrics.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = REPORT_DIR / "domain_summary.json"
    summary = {
        "generated_at": datetime.now().isoformat(),
        "model": MODEL,
        "evaluation_type": "environment_only",
        "max_concurrency": 1,
        "domains": summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Per-task metrics: {csv_path}")
    print(f"Domain summary: {summary_path}")


if __name__ == "__main__":
    main()
