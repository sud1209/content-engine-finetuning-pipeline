"""
Loads test drafts, calls the production post_scorer to get ground-truth labels,
and writes results to data/raw/test_ground_truth.jsonl.

Usage:
    uv run python dataset/generate_test_ground_truth.py
    uv run python dataset/generate_test_ground_truth.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

# Repo root for local imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Path to the production twitter-content-engine scorer
_TCE_SCRIPTS = Path("C:/Users/sudar/OneDrive/Desktop/twitter-content-engine/scripts")
sys.path.insert(0, str(_TCE_SCRIPTS))

INPUT_PATH = _REPO_ROOT / "data" / "raw" / "test_drafts.jsonl"
OUTPUT_PATH = _REPO_ROOT / "data" / "raw" / "test_ground_truth.jsonl"


def load_drafts(path: Path) -> list[dict]:
    drafts = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                drafts.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                print(
                    f"[line {line_num}] Skipping malformed JSON: {exc}",
                    file=sys.stderr,
                )
    return drafts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ground-truth scores for test drafts using the production scorer."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first draft dict without calling the API and exit.",
    )
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        print(f"ERROR: input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    drafts = load_drafts(INPUT_PATH)
    if not drafts:
        print("ERROR: no valid drafts found in input file.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("--dry-run: first draft dict (no API call made):")
        print(json.dumps(drafts[0], indent=2))
        return

    try:
        from post_scorer import batch_score_posts  # type: ignore[import]
    except ImportError as exc:
        print(
            f"ERROR: could not import batch_score_posts from post_scorer: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # batch_score_posts mutates drafts in place
    batch_score_posts(drafts)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for example in drafts:
            out_f.write(json.dumps(example) + "\n")

    print(f"Scored {len(drafts)} examples. Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
