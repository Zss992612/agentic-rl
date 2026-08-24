import argparse
import json
from pathlib import Path

from agentic_rl.data.selection import build_selection_summary, select_candidates

TRAINING_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = TRAINING_DIR / "data/sft/retail_serial_v3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--min-per-task", type=int, default=2)
    args = parser.parse_args()

    input_path = args.data_dir / "semantic_screened_candidates.jsonl"
    output_path = args.data_dir / f"selected_candidates_{args.target_size}.jsonl"
    summary_path = args.data_dir / "selection_summary.json"

    with input_path.open(encoding="utf-8") as file:
        candidates = [json.loads(line) for line in file if line.strip()]
    selected = select_candidates(
        candidates,
        target_size=args.target_size,
        min_per_task=args.min_per_task,
    )

    with output_path.open("w", encoding="utf-8") as file:
        for candidate in selected:
            file.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(
            build_selection_summary(candidates, selected),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Selected {len(selected)} candidates: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
