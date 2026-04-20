# Tweet Scorer — QLoRA Fine-Tuning Pipeline

Fine-tuned **Llama 3.1 8B** to replace Claude Haiku API calls in a tweet quality scoring system. The goal: own the scoring model instead of paying per inference. This is an exploratory research project — the pipeline is complete, results are benchmarked honestly, and the next iteration path is documented.

**Status:** Research complete. Not in production. Findings inform the next training run.

---

## The Problem

[`twitter-content-engine`](../twitter-bot) scores every tweet draft across 6 quality dimensions before publishing — hook strength, tone compliance, algorithm optimization, data specificity, pillar alignment, and CTA quality. It calls `claude-haiku-4-5-20251001` for every score. That's an external API call on the critical path, with per-call cost and ~800ms latency.

The question this project answers: **can an open-source model learn this rubric well enough to replace those calls?**

---

## Results

Three-way benchmark on 200 held-out examples. Ground truth = Claude Haiku (the model being replaced).

| Model | Mean MAE | Within-1 Agr. | Never-list F1 | Latency |
|---|---|---|---|---|
| Fine-tuned Llama 3.1 8B | 1.974 | 46.4% | **1.000** | ~6s |
| Base Llama 3.1 8B (zero-shot) | 1.923 | 52.3% | 0.833 | ~10s |
| Claude Haiku (ground truth) | 0.0 | 100% | 1.000 | ~800ms |

**What worked:** The fine-tuned model learned rule-based dimensions well. `tone_compliance` MAE of 0.21 with 96% within-1 agreement. `never_list_f1 = 1.000` — perfect precision and recall on hashtag violation detection, which is the highest-stakes binary decision in the rubric. The fine-tuned model also runs **40% faster** than base Llama at inference time due to Unsloth's optimized kernels.

**What didn't:** Judgment-based dimensions — `cta_quality` (MAE 3.42), `x_algorithm_optimization` (MAE 2.67) — are not close enough to replace Haiku. The fine-tuned model is marginally worse than the base model on mean MAE overall.

**Why:** Two compounding data quality issues, documented below. These are solvable — this is not a model capacity problem.

### Per-Dimension Breakdown

| Dimension | MAE | Within-1 | Assessment |
|---|---|---|---|
| `tone_compliance` | **0.21** | **96%** | Production-ready |
| `pillar_alignment` | **1.00** | **77%** | Usable |
| `hook_strength` | 1.87 | 42% | Needs better training data |
| `x_algorithm_optimization` | 2.67 | 12% | Not ready |
| `data_specificity` | 2.68 | 33% | Not ready |
| `cta_quality` | 3.42 | 19% | Not ready |

---

## Root Cause

**Training data repetition.** Batches 09–28 (1,440 of 2,355 examples, or 61% of the dataset) used 20 tweet templates cycled with slightly varied scores. The model learned to associate content fingerprints with scores instead of applying rubric logic. A model that memorizes "this template scores a 7" generalizes poorly to new content.

**Label distribution mismatch.** Training labels came from Claude Sonnet; ground truth labels came from Claude Haiku. Same rubric, different calibration — Sonnet is more conservative on `cta_quality` and more generous on `hook_strength`. The fine-tuned model learned Sonnet's bias and is penalized at benchmark time against Haiku's bias.

Neither of these is a fundamental pipeline failure. Both have clear fixes.

---

## What the Next Run Looks Like

These are evidence-backed interventions, not generic "get more data" advice.

**1. Eliminate template repetition** — Regenerate batches 09–28 with unique tweet content per example. One approach: generate drafts from real AI/ML news headlines, then score. Expected impact: largest gains on `cta_quality` and `x_algorithm_optimization`, since those require rubric generalization.

**2. Align training labels to Haiku** — Score all training examples with Haiku instead of Sonnet. Cost: ~$2 for 2,355 examples. This eliminates the scorer bias entirely and aligns the training distribution with the benchmark distribution.

**3. Chain-of-thought format** — Change training format from `(tweet) → {scores}` to `(tweet) → {reasoning} → {scores}`. The `reasoning` field already exists in every training example. Repositioning it before the scores teaches rubric application step-by-step rather than direct pattern-to-score mapping.

**4. SFT → GRPO** — Once SFT converges better, a GRPO stage can directly optimize for Haiku score agreement. Reward signal: composite MAE vs. Haiku label. TRL's `GRPOTrainer` supports this. Prerequisite: a well-calibrated SFT model — GRPO on a poorly-calibrated base amplifies existing errors.

---

## Architecture

```
Claude Code (Sonnet)          Claude Haiku API
     │                              │
     ▼                              ▼
scored_tweets_raw.jsonl      test_ground_truth.jsonl
(2,355 train+val examples)   (200 held-out examples)
     │                              │
     └──────────────┬───────────────┘
                    ▼
           validate_dataset.py
           (schema + rubric hash validation)
                    │
                    ▼
       sud1157/tweet-scorer-dataset (HF Hub)
                    │
                    ▼
        Unsloth + QLoRA + SFTTrainer
        (Kaggle T4 x1, ~6h, rank=16, 4-bit)
                    │
                    ▼
       sud1157/tweet-scorer-llama3-8b (HF Hub)
                    │
                    ▼
        3-way benchmark: fine-tuned vs base vs Haiku GT
```

---

## Cost

| Stage | Compute | API Cost |
|---|---|---|
| Dataset generation (2,355 examples) | Claude Code subscription | $0 |
| Test ground truth (200 examples) | Claude Haiku API | ~$0.20 |
| QLoRA fine-tuning (~6h) | Kaggle T4 (free tier) | $0 |
| Benchmark | Kaggle T4 (free tier) | $0 |
| **Total** | | **~$0.20** |

---

## Serving Design

The `serving/` directory contains the productionization path — vLLM + FastAPI + guided JSON decoding. Not deployed; documents what production would look like.

Guided decoding (`guided_json`) enforces valid JSON output at the token level — eliminates parse errors and retry logic entirely.

```bash
# vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model sud1157/tweet-scorer-llama3-8b \
    --dtype float16 \
    --max-model-len 2048

# Score a tweet
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"content": "We replaced 3,000 lines of prompt engineering with 500 labeled examples."}'
```

At production scale on an A10G instance (~$1/hr): **~$0 per 1,000 scores vs ~$0.50 for Haiku API**. Latency: ~400ms vs ~800ms.

---

## Reproducing

```bash
# 1. Install dependencies
uv sync
cp .env.example .env  # ANTHROPIC_API_KEY, HF_TOKEN, WANDB_API_KEY

# 2. Validate existing dataset
python dataset/validate_dataset.py

# 3. Push dataset to HF Hub (already published: sud1157/tweet-scorer-dataset)
python -m dataset.split_dataset

# 4. Fine-tune on Kaggle
#    Open notebooks/02_train_kaggle.ipynb
#    Accelerator: GPU T4 x1
#    Add HF_TOKEN + WANDB_API_KEY as Kaggle Secrets
#    Runtime: ~6h. Adapter auto-pushed to sud1157/tweet-scorer-llama3-8b

# 5. Benchmark on Kaggle
#    Open notebooks/03_benchmark_kaggle.ipynb
#    Add test_ground_truth.jsonl as a Kaggle dataset input
#    Download benchmark_results.json + benchmark_card.html from output
```

---

## Artifacts

| Artifact | Location |
|---|---|
| Training dataset | `sud1157/tweet-scorer-dataset` on HF Hub |
| LoRA adapter | `sud1157/tweet-scorer-llama3-8b` on HF Hub |
| Benchmark results | `benchmark/results/benchmark_results.json` |
| Benchmark card | `benchmark/results/benchmark_card.html` |

---

## What This Demonstrates

- **Below the API surface** — synthetic dataset design with schema versioning, rubric hash enforcement, quality tier distribution across training splits
- **QLoRA / PEFT mechanics** — 4-bit NF4 quantization, rank-16 LoRA on all 7 attention + MLP projections, Unsloth gradient checkpointing, packing for T4 utilization
- **Honest benchmarking** — three-way comparison against a real ground truth, per-dimension breakdown, root cause analysis of underperformance with specific next steps
- **Production serving design** — vLLM guided decoding, FastAPI rate limiting, SQLite cost tracking, eval gate before model publish
- **Cost discipline** — full pipeline for ~$0.20 in API spend on free compute
