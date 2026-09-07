"""Build the one-row dataset used by the first real Retail rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tau2.domains.retail.environment import get_tasks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/grpo/smoke_single_rollout.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task_id = str(config["data"]["task_id"])
    task_ids = {task.id for task in get_tasks(task_split_name=None)}
    if task_id not in task_ids:
        raise ValueError(f"Unknown τ² Retail task: {task_id}")

    output_path = PROJECT_ROOT / config["paths"]["task_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "data_source": "tau2_retail",
        "agent_name": "tau2_retail",
        "prompt": [
            {
                "role": "user",
                "content": f"Run τ² Retail task {task_id}.",
            }
        ],
        "ability": "retail",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "task_id": task_id,
        "tau2_seed": int(config["data"]["tau2_seed"]),
        "extra_info": {"split": "smoke", "index": 0},
    }
    output_path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Retail task {task_id}: {output_path}")


if __name__ == "__main__":
    main()
