import argparse
import json
from pathlib import Path

from agentic_rl.data.screening import screen_and_rank

TRAINING_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = TRAINING_DIR / "data/sft"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    input_path = args.data_dir / "candidates.jsonl"
    output_path = args.data_dir / "ranked_candidates.jsonl"
    summary_path = args.data_dir / "ranking_summary.json"

    with input_path.open(encoding="utf-8") as file:
        candidates = [json.loads(line) for line in file if line.strip()]
    ranked, summary = screen_and_rank(candidates)

    with output_path.open("w", encoding="utf-8") as file:
        for candidate in ranked:
            file.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Ranked {len(ranked)} candidates")
    print(f"Candidates: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
