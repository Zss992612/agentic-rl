import argparse
from pathlib import Path

from agentic_rl.data.collection import collect_from_config
from dotenv import load_dotenv

TRAINING_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = TRAINING_DIR.parent
DEFAULT_CONFIG_PATH = (
    TRAINING_DIR / "configs/sft/retail_collection_v2_opencode_go.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    load_dotenv(WORKSPACE_DIR / "tau2-bench/.env")
    collect_from_config(args.config)


if __name__ == "__main__":
    main()
