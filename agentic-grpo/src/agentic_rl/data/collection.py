from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def register_model_pricing(model_name: str) -> None:
    import litellm

    litellm.register_model(
        {
            model_name: {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )


def collect_tier(experiment: dict, tier: dict) -> None:
    from tau2 import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner import get_tasks, run_tasks
    from tau2.utils.utils import DATA_DIR

    sampling = experiment["sampling"]
    task_ids = experiment["task_ids"]
    api_base = experiment["provider"]["api_base"]
    run_name = f"{sampling['run_name_prefix']}_{tier['name']}"
    save_dir = DATA_DIR / "simulations" / run_name
    agent_args = {
        "temperature": tier["temperature"],
        "api_base": api_base,
        **experiment["agent"].get("args", {}),
    }
    run_config = TextRunConfig(
        domain=experiment["domain"],
        task_split_name=experiment["task_split"],
        task_ids=task_ids,
        llm_agent=experiment["agent"]["model"],
        llm_user=experiment["user"]["model"],
        llm_args_agent=agent_args,
        llm_args_user={
            "temperature": experiment["user"]["temperature"],
            "api_base": api_base,
        },
        num_trials=tier["num_trials"],
        max_concurrency=sampling["max_concurrency"],
        auto_resume=sampling["auto_resume"],
        save_to=run_name,
    )

    tasks = get_tasks(
        task_set_name=experiment["domain"],
        task_split_name=experiment["task_split"],
        task_ids=task_ids,
    )
    results = run_tasks(
        run_config,
        tasks,
        save_path=save_dir / "results.json",
        save_dir=save_dir,
        evaluation_type=EvaluationType(experiment["evaluation_type"]),
    )
    print(
        f"{tier['name']}: saved {len(results.simulations)} trajectories to {save_dir}"
    )


def collect_from_config(config_path: Path) -> None:
    experiment = load_config(config_path)
    register_model_pricing(experiment["agent"]["model"])
    print(f"Loaded sampling config: {config_path}")
    for tier in experiment["sampling"]["tiers"]:
        collect_tier(experiment, tier)
