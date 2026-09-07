import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

TRAINING_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = TRAINING_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal Retail baseline test against a local vLLM server."
    )
    parser.add_argument("config", type=Path, help="Path to an evaluation YAML file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from agentic_rl.data.collection import load_config, register_model_pricing
    from tau2 import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner import get_tasks, run_tasks
    from tau2.utils.utils import DATA_DIR

    config_path = args.config.resolve()
    experiment = load_config(config_path)

    split_path = TRAINING_DIR / experiment["task_selection"]["split_file"]
    split_config = load_config(split_path)
    split_name = experiment["task_selection"]["split_name"]
    selected_split = split_config["splits"][split_name]
    num_tasks = experiment["task_selection"]["num_tasks"]
    task_ids = selected_split["task_ids"][:num_tasks]

    load_dotenv(WORKSPACE_DIR / "tau2-bench/.env")
    register_model_pricing(experiment["agent"]["model"])
    register_model_pricing(experiment["user"]["model"])

    server = experiment["server"]
    agent = experiment["agent"]
    user = experiment["user"]
    sampling = experiment["sampling"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment['experiment_name']}_{timestamp}"
    save_dir = DATA_DIR / "simulations" / run_name

    run_config = TextRunConfig(
        domain=experiment["domain"],
        task_split_name=selected_split["task_split_name"],
        task_ids=task_ids,
        llm_agent=agent["model"],
        llm_user=user["model"],
        llm_args_agent={
            "api_base": server["api_base"],
            "api_key": server["api_key"],
            **{key: value for key, value in agent.items() if key != "model"},
        },
        llm_args_user={
            "api_base": user["api_base"],
            "temperature": user["temperature"],
        },
        num_trials=sampling["num_trials"],
        max_concurrency=sampling["max_concurrency"],
        seed=sampling["seed"],
        save_to=run_name,
    )
    tasks = get_tasks(
        task_set_name=experiment["domain"],
        task_split_name=selected_split["task_split_name"],
        task_ids=task_ids,
    )

    print(f"Model: {server['model_path']}")
    print(f"Tasks: {', '.join(task_ids)}")
    results = run_tasks(
        run_config,
        tasks,
        save_path=save_dir / "results.json",
        save_dir=save_dir,
        evaluation_type=EvaluationType(experiment["evaluation_type"]),
    )

    for simulation in results.simulations:
        print(f"Task {simulation.task_id}: reward={simulation.reward_info.reward}")
    print(f"Raw results: {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
