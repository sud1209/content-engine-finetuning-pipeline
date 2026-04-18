"""
Loads test drafts, calls Claude Haiku with the canonical SCORING_SYSTEM_PROMPT rubric,
and writes ScoredExample-compatible records to data/raw/test_ground_truth.jsonl.

Usage:
    python -m dataset.generate_test_ground_truth
    python -m dataset.generate_test_ground_truth --dry-run
    python -m dataset.generate_test_ground_truth --batch-size 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dataset.schemas import (  # noqa: E402
    RUBRIC_HASH,
    SCORING_SYSTEM_PROMPT,
    ScoreSet,
    ScoredExample,
)

INPUT_PATH = _REPO_ROOT / "data" / "raw" / "test_drafts.jsonl"
OUTPUT_PATH = _REPO_ROOT / "data" / "raw" / "test_ground_truth.jsonl"

SCORE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_BATCH_SIZE = 20


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
                print(f"[line {line_num}] Skipping malformed JSON: {exc}", file=sys.stderr)
    return drafts


def score_batch(drafts: list[dict], client) -> list[dict]:
    """Score a batch of drafts via Claude Haiku. Returns list of scored dicts or None on error."""
    results = []
    for draft in drafts:
        try:
            response = client.messages.create(
                model=SCORE_MODEL,
                max_tokens=400,
                system=SCORING_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Score this tweet draft:\n\n{draft['content']}",
                    }
                ],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            # Extract first complete JSON object (handles extra trailing text)
            decoder = json.JSONDecoder()
            score_data, _ = decoder.raw_decode(raw.strip())
            # Clamp all dims to [1,10] except tone_compliance when never_list_violation=True (where 0 is valid)
            nvl = bool(score_data.get("never_list_violation", False))
            for dim in ("hook_strength", "tone_compliance", "x_algorithm_optimization", "data_specificity", "pillar_alignment", "cta_quality"):
                if dim == "tone_compliance" and nvl:
                    continue
                if isinstance(score_data.get(dim), int) and score_data[dim] < 1:
                    score_data[dim] = 1
            results.append({"draft": draft, "score_data": score_data, "error": None})
        except Exception as exc:
            results.append({"draft": draft, "score_data": None, "error": str(exc)})
    return results


def build_scored_example(draft: dict, score_data: dict) -> ScoredExample | None:
    try:
        scores = ScoreSet.model_validate(score_data)
        quality_tier = scores.tier()
        example = ScoredExample(
            id=draft["id"],
            content=draft["content"],
            pillar=draft["pillar"],
            quality_tier=quality_tier,
            scores=scores,
            rubric_hash=RUBRIC_HASH,
        )
        return example
    except Exception as exc:
        print(f"  [WARN] Could not build ScoredExample for {draft['id']}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ground-truth scores for test drafts using Claude Haiku."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print first draft without API call.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        print(f"ERROR: input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    drafts = load_drafts(INPUT_PATH)
    if not drafts:
        print("ERROR: no valid drafts found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(drafts)} test drafts.")

    if args.dry_run:
        print("--dry-run: first draft (no API call made):")
        print(json.dumps(drafts[0], indent=2))
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    failed = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for batch_start in range(0, len(drafts), args.batch_size):
            batch = drafts[batch_start : batch_start + args.batch_size]
            batch_num = batch_start // args.batch_size + 1
            total_batches = (len(drafts) + args.batch_size - 1) // args.batch_size
            print(f"Scoring batch {batch_num}/{total_batches} ({len(batch)} drafts)...", flush=True)

            results = score_batch(batch, client)

            for res in results:
                if res["error"]:
                    print(f"  [ERROR] {res['draft']['id']}: {res['error']}", file=sys.stderr)
                    failed += 1
                    continue

                example = build_scored_example(res["draft"], res["score_data"])
                if example is None:
                    failed += 1
                    continue

                out_f.write(example.model_dump_json() + "\n")
                accepted += 1

            # Brief pause between batches to avoid rate limits
            if batch_start + args.batch_size < len(drafts):
                time.sleep(1)

    print(f"\nDone. Accepted: {accepted}, Failed: {failed}")
    print(f"Written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
