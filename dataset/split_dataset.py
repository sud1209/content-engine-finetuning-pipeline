"""
split_dataset.py

Loads scored tweet examples, validates them through ScoredExample, splits into
train/val/test, formats for SFTTrainer chat template, saves to data/processed/,
and optionally uploads to HuggingFace Hub as sudar/tweet-scorer-dataset.

Usage:
    python -m dataset.split_dataset            # split + push to HF Hub
    python -m dataset.split_dataset --no-push  # split only (local testing)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from dataset.schemas import RUBRIC_HASH, SCORING_SYSTEM_PROMPT, ScoredExample  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TRAIN_VAL_PATH = RAW_DIR / "scored_tweets_raw.jsonl"
TEST_PATH = RAW_DIR / "test_ground_truth.jsonl"

HF_DATASET_NAME = "sudar/tweet-scorer-dataset"

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_for_training(example: ScoredExample) -> dict:
    scores = example.scores
    return {
        "messages": [
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Score this tweet draft:\n\n{example.content}"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "hook_strength": scores.hook_strength,
                        "tone_compliance": scores.tone_compliance,
                        "x_algorithm_optimization": scores.x_algorithm_optimization,
                        "data_specificity": scores.data_specificity,
                        "pillar_alignment": scores.pillar_alignment,
                        "cta_quality": scores.cta_quality,
                        "never_list_violation": scores.never_list_violation,
                        "reasoning": scores.reasoning,
                    }
                ),
            },
        ],
        "rubric_hash": example.rubric_hash,
        "pillar": example.pillar,
        "quality_tier": example.quality_tier,
        "composite_score": scores.composite_score(),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[ScoredExample]:
    """Parse a .jsonl file into validated ScoredExample objects.

    Invalid rows are skipped with a warning to stderr.
    """
    examples: list[ScoredExample] = []
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                example = ScoredExample.model_validate(raw)
                examples.append(example)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARNING: skipping line {lineno} in {path.name}: {exc}",
                    file=sys.stderr,
                )
    return examples


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Split and format tweet-scorer dataset.")
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip uploading to HuggingFace Hub (local testing).",
    )
    args = parser.parse_args()

    # --- Load & validate ---------------------------------------------------
    print("Loading train+val examples …")
    train_val_examples = load_jsonl(TRAIN_VAL_PATH)
    print(f"  Loaded {len(train_val_examples)} valid examples from {TRAIN_VAL_PATH.name}")

    print("Loading test examples …")
    test_examples = load_jsonl(TEST_PATH)
    print(f"  Loaded {len(test_examples)} valid examples from {TEST_PATH.name}")

    # --- Shuffle & split train/val ----------------------------------------
    random.seed(42)
    random.shuffle(train_val_examples)

    n_total = len(train_val_examples)
    n_train = int(n_total * 0.80)
    # val gets the remaining 20 % (≈ 10 % of notional 90 % if test were included,
    # but per spec train+val is the full non-test corpus so we do 80/20 here)
    train_examples = train_val_examples[:n_train]
    val_examples = train_val_examples[n_train:]

    # --- Format ------------------------------------------------------------
    train_records = [format_for_training(e) for e in train_examples]
    val_records = [format_for_training(e) for e in val_examples]
    test_records = [format_for_training(e) for e in test_examples]

    # --- Save processed splits ---------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_records, PROCESSED_DIR / "train.jsonl")
    save_jsonl(val_records, PROCESSED_DIR / "val.jsonl")
    save_jsonl(test_records, PROCESSED_DIR / "test.jsonl")

    # --- Report ------------------------------------------------------------
    print("\nSplit sizes:")
    print(f"  train : {len(train_records)}")
    print(f"  val   : {len(val_records)}")
    print(f"  test  : {len(test_records)}")
    print(f"\nRubric hash used: {RUBRIC_HASH}")

    # --- HF Hub upload -----------------------------------------------------
    if args.no_push:
        print("\n--no-push flag set; skipping HuggingFace Hub upload.")
        return

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "ERROR: HF_TOKEN environment variable not set. "
            "Set it or pass --no-push to skip upload.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        print(
            "ERROR: 'datasets' package not installed. Run: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nUploading to HuggingFace Hub as '{HF_DATASET_NAME}' …")

    def records_to_hf(records: list[dict]) -> Dataset:
        # DatasetDict expects columnar dicts; convert list-of-dicts to dict-of-lists
        if not records:
            return Dataset.from_dict({})
        keys = records[0].keys()
        columnar = {k: [r[k] for r in records] for k in keys}
        # 'messages' is a list-of-dicts per row — HF handles it as a Sequence
        return Dataset.from_dict(columnar)

    dataset_dict = DatasetDict(
        {
            "train": records_to_hf(train_records),
            "validation": records_to_hf(val_records),
            "test": records_to_hf(test_records),
        }
    )

    dataset_dict.push_to_hub(HF_DATASET_NAME, token=hf_token)
    print(f"Upload complete: https://huggingface.co/datasets/{HF_DATASET_NAME}")


if __name__ == "__main__":
    main()
