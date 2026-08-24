import argparse
import json
from pathlib import Path

from agentic_rl.data.sft_format import build_sft_row, build_sft_summary

TRAINING_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = TRAINING_DIR / "data/sft/retail_serial_v3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    selected_path = args.data_dir / f"selected_candidates_{args.size}.jsonl"
    manifest_path = args.data_dir / "manifest.json"
    output_path = args.data_dir / f"sft_train_{args.size}.jsonl"
    summary_path = args.data_dir / f"sft_train_{args.size}_summary.json"

    with selected_path.open(encoding="utf-8") as file:
        candidates = [json.loads(line) for line in file if line.strip()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        build_sft_row(
            candidate,
            system_prompt=manifest["agent_system_prompt"],
            tools=manifest["tool_schemas"],
        )
        for candidate in candidates
    ]

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(
            build_sft_summary(rows, str(selected_path)),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} SFT examples: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
