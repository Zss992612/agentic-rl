from pathlib import Path

from dotenv import load_dotenv

from agentic_rl.data.collection import load_config, register_model_pricing

TRAINING_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = TRAINING_DIR.parent
CONFIG_PATH = TRAINING_DIR / "configs/sft/retail_collection_v2_opencode_go.yaml"


def main() -> None:
    from tau2 import TextRunConfig
    from tau2.runner import run_domain

    load_dotenv(WORKSPACE_DIR / "tau2-bench/.env")
    experiment = load_config(CONFIG_PATH)
    model_name = experiment["agent"]["model"]
    register_model_pricing(model_name)
    api_base = experiment["provider"]["api_base"]

    config = TextRunConfig(
        domain=experiment["domain"],
        task_split_name=experiment["task_split"],
        task_ids=["0"],
        llm_agent=model_name,
        llm_user=experiment["user"]["model"],
        llm_args_agent={"temperature": 0.2, "api_base": api_base},
        llm_args_user={
            "temperature": experiment["user"]["temperature"],
            "api_base": api_base,
        },
        num_trials=1,
        max_concurrency=1,
        save_to="retail_task_0_opencode_go_zero_cost",
    )

    simulation = run_domain(config).simulations[0]
    print(f"Task: {simulation.task_id}")
    print(f"Reward: {simulation.reward_info.reward}")


if __name__ == "__main__":
    main()
