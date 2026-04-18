"""
Validates data/raw/scored_tweets_raw.jsonl and prints a quality report.

Checks:
  1. Parse every line through ScoredExample — count malformed rows
  2. Flag examples where all 6 integer scores are identical
  3. Flag examples where reasoning word count < 20
  4. Print composite score tier distribution
  5. Verify never_list_violation=True examples have '#' in content
  6. Print pillar distribution
  7. Pass/fail summary — fail if < 2000 valid examples OR any tier has 0 examples

Exit code 0 if pass, 1 if fail.

Usage:
    uv run python dataset/validate_dataset.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dataset.schemas import DIMENSIONS, ScoredExample  # noqa: E402

INPUT_PATH = _REPO_ROOT / "data" / "raw" / "scored_tweets_raw.jsonl"

MIN_VALID_EXAMPLES = 2000
ALL_TIERS = {"ready", "below_target", "failed_floor"}


def word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"ERROR: dataset file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    valid_examples: list[ScoredExample] = []
    malformed_count = 0
    all_scores_identical: list[str] = []
    short_reasoning: list[str] = []
    never_list_mismatch: list[str] = []
    tier_counts: Counter = Counter()
    pillar_counts: Counter = Counter()

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"  [line {line_num}] JSON parse error: {exc}", file=sys.stderr)
                malformed_count += 1
                continue

            try:
                example = ScoredExample.model_validate(data)
            except Exception as exc:
                print(
                    f"  [line {line_num}] Validation error: {exc}", file=sys.stderr
                )
                malformed_count += 1
                continue

            valid_examples.append(example)

            # Check 2: all 6 integer scores identical
            int_scores = [getattr(example.scores, d) for d in DIMENSIONS]
            if len(set(int_scores)) == 1:
                all_scores_identical.append(example.id)

            # Check 3: reasoning word count < 20
            if word_count(example.scores.reasoning) < 20:
                short_reasoning.append(example.id)

            # Check 5: never_list_violation=True must have '#' in content
            if example.scores.never_list_violation and "#" not in example.content:
                never_list_mismatch.append(example.id)

            # Tier and pillar tracking
            tier_counts[example.scores.tier()] += 1
            pillar_counts[example.pillar] += 1

    total_lines = len(valid_examples) + malformed_count

    # ── Report ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total lines processed : {total_lines}")
    print(f"Valid examples        : {len(valid_examples)}")
    print(f"Malformed rows        : {malformed_count}")
    print()

    # Check 2
    print(f"All-identical scores  : {len(all_scores_identical)} examples")
    if all_scores_identical:
        for eid in all_scores_identical[:10]:
            print(f"  - {eid}")
        if len(all_scores_identical) > 10:
            print(f"  ... and {len(all_scores_identical) - 10} more")
    print()

    # Check 3
    print(f"Short reasoning (<20w): {len(short_reasoning)} examples")
    if short_reasoning:
        for eid in short_reasoning[:10]:
            print(f"  - {eid}")
        if len(short_reasoning) > 10:
            print(f"  ... and {len(short_reasoning) - 10} more")
    print()

    # Check 4: tier distribution
    print("Tier distribution:")
    for tier in sorted(ALL_TIERS):
        count = tier_counts.get(tier, 0)
        pct = (count / len(valid_examples) * 100) if valid_examples else 0.0
        print(f"  {tier:<20} {count:>6}  ({pct:.1f}%)")
    print()

    # Check 5
    print(f"never_list / '#' mismatch: {len(never_list_mismatch)} examples")
    if never_list_mismatch:
        for eid in never_list_mismatch[:10]:
            print(f"  - {eid}")
    print()

    # Check 6: pillar distribution
    print("Pillar distribution:")
    for pillar, count in pillar_counts.most_common():
        pct = (count / len(valid_examples) * 100) if valid_examples else 0.0
        print(f"  {pillar:<30} {count:>6}  ({pct:.1f}%)")
    print()

    # ── Pass / Fail ──────────────────────────────────────────────────────────
    failures: list[str] = []

    if len(valid_examples) < MIN_VALID_EXAMPLES:
        failures.append(
            f"Only {len(valid_examples)} valid examples (minimum {MIN_VALID_EXAMPLES} required)"
        )

    missing_tiers = ALL_TIERS - set(tier_counts.keys())
    if missing_tiers:
        failures.append(f"Missing tiers with 0 examples: {', '.join(sorted(missing_tiers))}")

    print("=" * 60)
    if failures:
        print("RESULT: FAIL")
        for reason in failures:
            print(f"  - {reason}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("RESULT: PASS")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
