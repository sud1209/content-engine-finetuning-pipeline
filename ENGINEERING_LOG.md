# Engineering Log — Tweet Scorer Fine-tuning Pipeline

This document captures key design decisions made during implementation, including tradeoffs, alternatives considered, and the reasoning behind non-obvious choices.

---

## 1. Dataset Layer

### 1.1 Ground Truth Strategy: Claude Code for Training, Haiku for Test

**Decision:** Use Claude Code (Sonnet subscription) to generate ~2,500 training labels interactively, and reserve Anthropic Haiku API calls exclusively for the ~500-example test set.

**Rationale:** Dataset generation is a one-time exercise. Running it through the Claude subscription costs $0 in API charges. Haiku API calls for the held-out test set cost ~$0.20–$0.30 total. This avoids spending $3–11 on GPT-4o or $2 on Haiku for training labels that could be generated for free.

**Alternative considered:** Using `post_scorer.py` from `twitter-content-engine` directly (which calls the live API). Rejected because it would consume real API balance for something a subscription already covers, and introduces a runtime dependency on the production scoring system during data generation.

**Tradeoff:** Training labels generated interactively via Claude Code are not versioned in the same way as a batch API call log. This is acceptable because the rubric is version-locked via SHA256 hash (`RUBRIC_HASH` in `schemas.py`), so any label generated under a different rubric version is rejected at ingest.

---

### 1.2 Rubric Versioning via SHA256

**Decision:** Compute `RUBRIC_HASH = hashlib.sha256(SCORING_SYSTEM_PROMPT.encode()).hexdigest()[:8]` at module load time. Every training example stores this hash. Ingest rejects any example where stored hash ≠ current hash.

**Rationale:** The scoring rubric is the ground truth contract. If the rubric changes mid-dataset, you silently mix labels from two different scoring functions — a data poisoning equivalent. Versioning forces intentionality: if the rubric changes, you must regenerate all examples under the new hash.

**Implementation note:** `ScoredExample.validate_rubric_and_tier` is a `model_validator(mode='after')` that raises `ValueError` on hash mismatch. This means stale data fails loudly at ingest, not silently at training.

---

### 1.3 `never_list_violation` Enforcement

**Decision:** When any `#` character appears in tweet text, `composite_score` is forced to `0.0` and `tone_compliance` is forced to `0` regardless of what the LLM scores.

**Rationale:** The production `post_scorer.py` enforces this as a hard rule — hashtags are a categorical disqualifier. The fine-tuned model must learn this invariant. If we let LLMs assign partial scores to hashtag-containing tweets, the model learns wrong behavior.

**Implementation detail:** `ScoreSet.enforce_never_list` is a `model_validator(mode='after')`. `tone_compliance` field is `ge=1` (not `ge=0`) so that Pydantic allows the LLM to supply any valid score, and the post-validator zeroes it when needed. This design means the validator has sole authority over the zero value — the LLM cannot accidentally submit `tone_compliance=0` and have it pass validation without `never_list_violation=True`.

---

### 1.4 Composite Score Formula: `sum(score * weight/100) + 0.5`

**Decision:** The `+0.5` offset in `composite_score()` is intentional.

**Rationale:** This matches the production formula in `twitter-content-engine` exactly. The offset exists in production to shift scores slightly upward from the raw weighted sum, reflecting a calibration choice made when the rubric was first deployed. Replicating it here ensures the fine-tuned model is trained on the same target distribution the production system evaluates against.

**Warning for future maintainers:** Do not remove the `+0.5`. It looks like a bug (off-by-constant). It is not. It is load-bearing.

---

### 1.5 Quality Tier Thresholds

| Tier | Composite Score Range |
|------|----------------------|
| `ready` | ≥ 9.25 |
| `below_target` | 8.0 – 9.24 |
| `failed_floor` | < 8.0 |

These thresholds match production. The `ScoredExample` validator asserts `quality_tier == scores.tier()` — silent tier drift is not permitted.

---

## 2. Training Infrastructure

### 2.1 Kaggle P100 (Not Local GPU)

**Decision:** All training and benchmarking runs on Kaggle free tier (NVIDIA P100, 16GB VRAM, ~30h/week).

**Rationale:** The development machine has an RTX 3060 laptop with 6GB VRAM. Llama 3.1 8B with QLoRA requires ~14GB VRAM at minimum batch sizes. Local execution is not feasible. Kaggle provides adequate hardware for free.

**Consequence:** The `training/config.py` defaults are P100-tuned: `bf16=False` (P100 does not support BF16), `batch_size=2`, `grad_accumulation_steps=8`. These settings should not be changed without re-validating on the target hardware.

---

### 2.2 Unsloth + QLoRA over Full Fine-tune

**Decision:** Use Unsloth's optimized QLoRA kernels via `FastLanguageModel.from_pretrained()` with 4-bit quantization.

**Rationale:** Full fine-tuning of Llama 3.1 8B requires ~80GB VRAM (multiple A100s). QLoRA reduces this to ~14GB by quantizing the base model to 4-bit and training only low-rank adapter matrices. Unsloth provides 2x memory efficiency improvement over vanilla PEFT through custom CUDA kernels.

**LoRA targets:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` — all attention and MLP projection matrices. Targeting all projection layers (not just attention) captures more model capacity for structured output learning.

---

### 2.3 Optuna Sweep: 6 Trials

**Decision:** 6-trial Optuna sweep over `lora_rank ∈ {8, 16, 32}` and `learning_rate ∈ {1e-4, 2e-4, 5e-4}`.

**Rationale:** Kaggle sessions are time-limited. 6 trials is the practical maximum within a single session. The search space covers the two hyperparameters with the highest impact on QLoRA quality. Rank controls adapter expressiveness; learning rate controls convergence stability.

---

## 3. Serving Architecture (Design Artifact)

The serving layer (`serving/`, `docker/`) is implemented as production-ready code but is not deployed as part of this project. It documents what productionization would look like.

### 3.1 vLLM Guided Decoding

**Decision:** Use vLLM's `guided_json` parameter (outlines backend) to enforce the `ScoreSet` JSON schema at the token level.

**Rationale:** Without constrained decoding, LLMs occasionally generate malformed JSON or values outside valid ranges. Guided decoding prevents this at the sampling level — invalid tokens are masked. This eliminates the need for retry logic on parse failures.

**Tradeoff:** Guided decoding adds ~5–10% latency. Acceptable for a scoring use case where correctness matters more than raw throughput.

---

### 3.2 Circuit Breaker Pattern

**Decision:** FastAPI endpoint validates vLLM output via Pydantic `ScoreSet` post-decode. Returns HTTP 502 on validation failure rather than retrying.

**Rationale:** If guided decoding is working correctly, Pydantic failure indicates a model regression (the model learned to generate structurally valid but semantically invalid JSON). Retrying is unlikely to help and would mask the regression. A 502 surfaces it immediately to callers.

---

### 3.3 Cost Tracking via SQLite

**Decision:** Log every inference call to SQLite with token counts, latency, and pro-rated g5.xlarge cost.

**Formula:** `cost = (input_tokens/1000 * 0.0008) + (output_tokens/1000 * 0.0016) + (latency * g5_per_second)` where `g5_per_second = 1.006/3600` (g5.xlarge on-demand rate).

**Rationale:** SQLite requires no external infrastructure (no PostgreSQL, no ClickHouse). For a single-node serving deployment, write throughput is more than adequate. The cost formula enables weekly cost reporting without any external billing API.

---

### 3.4 When to Scale Beyond This Architecture

The serving design handles ~115K calls/day on a single g5.xlarge before saturating. At that point, the appropriate evolution is:

- Horizontal scaling with a load balancer across multiple vLLM instances
- Swap SQLite for a time-series store (ClickHouse, InfluxDB) for cost/latency analytics
- Add a Redis layer for rate limiting state (current `slowapi` state is in-process memory)
- Migrate to Kubernetes for autoscaling

These are out of scope for this repository. The code as written demonstrates the patterns; infrastructure provisioning is a separate concern.

---

## 4. Benchmark Design

### 4.1 Pre-computed Haiku Ground Truth

**Decision:** Test ground truth (`data/processed/test_ground_truth.jsonl`) is generated once via Haiku API and committed. Benchmark reads this file — it does not call any API at benchmark time.

**Rationale:** Benchmark reproducibility requires stable ground truth. If the test set were re-scored on each benchmark run, score drift from model updates would corrupt longitudinal comparisons. Committing the ground truth file makes it an immutable reference.

---

### 4.2 Graceful JSON Parse Fallback

**Decision:** `benchmark_runner_kaggle.py` catches `json.JSONDecodeError` on model outputs and substitutes a zeroed `ScoreSet` rather than raising.

**Rationale:** A single malformed output should not abort the entire benchmark. The zeroed score is obviously wrong (mean_mae will spike), making regressions visible without losing the rest of the benchmark run.

---

### 4.3 Eval Gate in `push_to_hub.py`

**Decision:** Block HF Hub push if `mean_mae` for the new model exceeds the published model's `mean_mae + 0.05`.

**Rationale:** Prevents accidental regression from being deployed. The 0.05 threshold is generous enough to allow normal score variation but tight enough to catch meaningful degradation. `--force` flag available for intentional overrides (e.g., architecture changes that temporarily worsen metrics before recovering).

---

## 5. GGUF Export: Two-Step Process

**Decision:** Export path is `safetensors → F16 GGUF → Q4_K_M quantized GGUF`, not direct quantization.

**Rationale:** `llama.cpp`'s `llama-quantize` tool requires an F16 GGUF as input — it cannot quantize directly from safetensors. The two-step process is not optional; it reflects the tool's requirements.

**Portability:** `scripts/export_gguf.py` uses `shutil.which()` to locate `llama-quantize` and `convert_hf_to_gguf.py` rather than hardcoded paths. If these tools are not on PATH, the script raises `FileNotFoundError` with a clear message.

---

## 6. CI/CD

### 6.1 GitHub Actions → Kaggle Trigger

**Decision:** `.github/workflows/train.yml` triggers on push to `main` when `dataset/**` or `training/**` changes. It calls `kaggle kernels push` which submits the notebook to Kaggle's queue.

**Limitation:** Kaggle kernel execution is asynchronous. The CI job succeeds when the kernel is *submitted*, not when it *completes*. The workflow does not poll for completion or fail on kernel errors.

**Rationale:** Kaggle's API does not support blocking kernel execution. Polling would require a long-running CI job (potentially hours). The current design is a reasonable tradeoff: CI ensures the kernel submission is always fresh on dataset/training changes; actual results must be checked in Kaggle UI or via a separate polling job.

---

## 7. Patterns and Principles

### Rubric as First-Class Citizen
Every design decision that touches scoring — ingest, training, serving, benchmark — treats the rubric as the source of truth. Hash versioning, field validation, and the eval gate all exist to preserve rubric integrity across the pipeline.

### Hardware-First Defaults
Configuration defaults are not aspirational — they reflect the actual hardware where the pipeline runs. Defaults that assume V100s or A100s when the actual target is a P100 cause silent failures (OOM, silent bf16 fallback).

### Observable Training
WandB integration, Optuna sweep logging, and the benchmark card are not nice-to-haves. For a portfolio project, the training run artifacts *are* the deliverable alongside the model. An unlogged training run cannot be reasoned about or presented.

### Fail Loudly at Boundaries
Model validators, circuit breakers, and the eval gate all prefer loud failure over silent degradation. Each boundary (ingest, serving, publication) has an explicit check that terminates rather than propagates corrupt state.

---

*Log written: 2026-04-18. Implementation branch: `feature/implement`.*
