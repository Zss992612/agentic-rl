import argparse
from pathlib import Path

from agentic_rl.data.extraction import extract_candidates

TRAINING_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = TRAINING_DIR.parent
DEFAULT_CONFIG_PATH = (
    TRAINING_DIR / "configs/sft/retail_collection_v2_opencode_go.yaml"
)
SIMULATION_DIR = WORKSPACE_DIR / "tau2-bench/data/simulations"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "data/sft"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()

    count = extract_candidates(
        config_path=config_path,
        simulation_dir=SIMULATION_DIR,
        output_dir=output_dir,
        source_config=str(config_path.relative_to(WORKSPACE_DIR)),
    )
    print(f"Wrote {count} candidates to {output_dir}")


if __name__ == "__main__":
    main()
