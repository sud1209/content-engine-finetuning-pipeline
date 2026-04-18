"""
Reads JSONL from stdin (or a file argument), validates each line via ScoredExample,
and appends valid lines to data/raw/scored_tweets_raw.jsonl.

Usage:
    uv run python dataset/generate_labeled_batch.py < batch_01.jsonl
    uv run python dataset/generate_labeled_batch.py batch_01.jsonl
"""

import json
import sys
from pathlib import Path

# Allow running from repo root or from the dataset/ directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dataset.schemas import ScoredExample  # noqa: E402

OUTPUT_PATH = _REPO_ROOT / "data" / "raw" / "scored_tweets_raw.jsonl"


def main() -> None:
    # Determine input source
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
        if not input_file.exists():
            print(f"ERROR: file not found: {input_file}", file=sys.stderr)
            sys.exit(1)
        lines = input_file.read_text(encoding="utf-8").splitlines()
    else:
        lines = sys.stdin.read().splitlines()

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0

    with OUTPUT_PATH.open("a", encoding="utf-8") as out_f:
        for line_num, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue  # skip blank lines silently

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(
                    f"[line {line_num}] REJECTED — JSON parse error: {exc}",
                    file=sys.stderr,
                )
                rejected += 1
                continue

            try:
                example = ScoredExample.model_validate(data)
            except Exception as exc:
                print(
                    f"[line {line_num}] REJECTED — validation error: {exc}",
                    file=sys.stderr,
                )
                rejected += 1
                continue

            out_f.write(example.model_dump_json() + "\n")
            accepted += 1

    # Count total lines currently in the output file
    total = 0
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            total = sum(1 for ln in f if ln.strip())

    print(f"Accepted {accepted}, rejected {rejected}. Total in file: {total}")


if __name__ == "__main__":
    main()
