import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agentic_rl.data.collection import register_model_pricing
from agentic_rl.data.semantic_screening import build_semantic_summary, run_batch

TRAINING_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = TRAINING_DIR.parent
DEFAULT_CONFIG_PATH = (
    TRAINING_DIR / "configs/sft/retail_semantic_screening_v1.yaml"
)


def resolve_path(path: str) -> Path:
    return TRAINING_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    load_dotenv(WORKSPACE_DIR / "tau2-bench/.env")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    judge_config = config["judge"]
    register_model_pricing(judge_config["model"])

    input_path = resolve_path(config["input_path"])
    manifest_path = resolve_path(config["manifest_path"])
    output_path = resolve_path(config["output_path"])
    summary_path = resolve_path(config["summary_path"])

    with input_path.open(encoding="utf-8") as file:
        candidates = [json.loads(line) for line in file if line.strip()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    errors = run_batch(
        candidates=candidates,
        retail_policy=manifest["retail_policy"],
        config=judge_config,
        output_path=output_path,
    )
    summary_path.write_text(
        json.dumps(
            build_semantic_summary(output_path, errors),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Audited candidates: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
