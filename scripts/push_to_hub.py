"""
Push trained LoRA adapter to Hugging Face Hub with an eval gate.

Gate logic:
  - Loads benchmark/results/benchmark_results.json (must exist — run benchmark first).
  - Computes metrics via compute_metrics().
  - If the Hub already has sudar/tweet-scorer-llama3-8b with a mean_mae metadata
    field, blocks the push when new mean_mae > old mean_mae + 0.05.
  - --force bypasses the gate entirely.

Usage:
    python scripts/push_to_hub.py
    python scripts/push_to_hub.py --force
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, ModelCard, metadata_load
from unsloth import FastLanguageModel

from benchmark.metrics import compute_metrics
from dataset.schemas import RUBRIC_HASH

REPO_ID = "sud1157/tweet-scorer-llama3-8b"
ADAPTER_DIR = "outputs/llama3-8b-tweet-scorer/final_adapter"
RESULTS_PATH = "benchmark/results/benchmark_results.json"


def _get_published_mean_mae(api: HfApi) -> float | None:
    """Return the mean_mae from the published model card, or None if unavailable."""
    try:
        card_content = api.model_info(REPO_ID).cardData
        if card_content and "mean_mae" in card_content:
            return float(card_content["mean_mae"])
    except Exception:
        pass

    # Fallback: try fetching the raw README.md metadata block
    try:
        path = api.hf_hub_download(repo_id=REPO_ID, filename="README.md")
        meta = metadata_load(path)
        if meta and "mean_mae" in meta:
            return float(meta["mean_mae"])
    except Exception:
        pass

    return None


def _build_model_card(mean_mae: float, rubric_hash: str) -> str:
    return f"""---
language: en
license: llama3
tags:
  - tweet-scorer
  - lora
  - unsloth
mean_mae: {mean_mae}
rubric_hash: {rubric_hash}
---

# tweet-scorer-llama3-8b

Fine-tuned LoRA adapter on top of `meta-llama/Llama-3.1-8B-Instruct` for
scoring tweet drafts across 6 quality dimensions.

## Metrics (vs Claude Haiku ground truth)

| Metric | Value |
|--------|-------|
| mean_mae | {mean_mae} |

## Rubric

Rubric hash: `{rubric_hash}` — regenerate training data if this changes.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Push adapter to HF Hub with eval gate.")
    parser.add_argument("--force", action="store_true", help="Bypass eval gate.")
    parser.add_argument("--results", default=RESULTS_PATH, help="Path to benchmark_results.json")
    parser.add_argument("--adapter-dir", default=ADAPTER_DIR, help="Path to final adapter dir")
    parser.add_argument("--repo-id", default=REPO_ID, help="HF Hub repo id")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: Benchmark results not found at {results_path}. Run benchmark first.")
        sys.exit(1)

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        print(f"ERROR: Adapter directory not found at {adapter_dir}.")
        sys.exit(1)

    # Compute metrics for new adapter
    results = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    if not results:
        # Try loading as a JSON array
        results = json.loads(results_path.read_text())

    metrics = compute_metrics(results)
    new_mean_mae = metrics["finetuned"]["aggregate"]["mean_mae"]
    print(f"New adapter mean_mae: {new_mean_mae}")

    api = HfApi()

    # Eval gate
    if not args.force:
        published_mae = _get_published_mean_mae(api)
        if published_mae is not None:
            print(f"Published adapter mean_mae: {published_mae}")
            threshold = published_mae + 0.05
            if new_mean_mae > threshold:
                print(
                    f"BLOCKED: new mean_mae ({new_mean_mae}) > published ({published_mae}) + 0.05 tolerance ({threshold}). "
                    "Use --force to bypass."
                )
                sys.exit(1)
            print(f"Gate passed: {new_mean_mae} <= {threshold}")
        else:
            print("No existing model found on Hub — skipping gate, proceeding with push.")
    else:
        print("--force flag set: skipping eval gate.")

    # Push adapter weights
    print(f"Pushing adapter from {adapter_dir} to {args.repo_id}...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Upload adapter | mean_mae={new_mean_mae} rubric={RUBRIC_HASH}",
    )

    # Push / update model card with metadata
    card_text = _build_model_card(mean_mae=new_mean_mae, rubric_hash=RUBRIC_HASH)
    api.upload_file(
        path_or_fileobj=card_text.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Update model card with latest eval metrics",
    )

    print(f"Done. Adapter published to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
