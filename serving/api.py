"""
FastAPI serving layer for the tweet content scoring pipeline.

Targets AWS g5.xlarge (A10G 24 GB) running vLLM at localhost:8001.
Uses vLLM's guided_json decoding to constrain model output to the exact
SCORE_OUTPUT_SCHEMA, then validates the payload through Pydantic as a
circuit breaker before returning to the caller.
"""

import asyncio
import json
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from dataset.schemas import SCORING_SYSTEM_PROMPT, ScoreSet
from serving.cost_tracker import log_request
from serving.guided_decoding import SCORE_OUTPUT_SCHEMA
from serving.schemas import HealthResponse, ScoreRequest, ScoreResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter (slowapi)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Tweet Scorer", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# vLLM client — points at the locally running vLLM server on port 8001.
# The AsyncOpenAI client is reused across requests (connection pooling).
# ---------------------------------------------------------------------------
_VLLM_BASE_URL = "http://localhost:8001/v1"
_MODEL_NAME = "tweet-scorer"

vllm_client = AsyncOpenAI(
    base_url=_VLLM_BASE_URL,
    api_key="vllm",  # vLLM ignores the key; required by the OpenAI client
)


# ---------------------------------------------------------------------------
# Core scoring helper
# ---------------------------------------------------------------------------
async def _score_one(content: str) -> ScoreResponse:
    """
    Send *content* to vLLM, validate the structured output, compute the
    composite score, log to the cost tracker, and return a ScoreResponse.
    """
    t0 = time.monotonic()

    completion = await vllm_client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        extra_body={"guided_json": SCORE_OUTPUT_SCHEMA},
        temperature=0.0,
        max_tokens=256,
    )

    latency_ms = (time.monotonic() - t0) * 1000

    # --- Post-decode circuit breaker ---
    try:
        raw = json.loads(completion.choices[0].message.content)
        scores = ScoreSet(**raw)  # validate via Pydantic
    except Exception as e:
        logger.error(
            "Output validation failed: %r  raw=%r",
            e,
            completion.choices[0].message.content,
        )
        raise HTTPException(status_code=502, detail="Model output failed validation")

    composite = scores.composite_score()

    # Best-effort cost / quality logging
    log_request(
        latency_ms=latency_ms,
        composite_score=composite,
        model_version=_MODEL_NAME,
    )

    return ScoreResponse(
        hook_strength=scores.hook_strength,
        tone_compliance=scores.tone_compliance,
        x_algorithm_optimization=scores.x_algorithm_optimization,
        data_specificity=scores.data_specificity,
        pillar_alignment=scores.pillar_alignment,
        cta_quality=scores.cta_quality,
        never_list_violation=scores.never_list_violation,
        composite_score=composite,
        reasoning=scores.reasoning,
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model=_MODEL_NAME)


@app.post("/score", response_model=ScoreResponse)
@limiter.limit("100/minute")
async def score(request: Request, body: ScoreRequest) -> ScoreResponse:
    """Score a single tweet. Rate-limited to 100 requests/minute per IP."""
    return await _score_one(body.content)


@app.post("/score/batch", response_model=list[ScoreResponse])
async def score_batch(request: Request, bodies: list[ScoreRequest]) -> list[ScoreResponse]:
    """
    Score up to 50 tweets concurrently.

    Requests beyond the 50-item cap are rejected with HTTP 422 before any
    inference is performed.
    """
    if len(bodies) > 50:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(bodies)} exceeds the maximum of 50.",
        )
    results = await asyncio.gather(
        *[_score_one(b.content) for b in bodies],
        return_exceptions=True,
    )

    # Re-raise the first exception encountered so the caller gets a clean error.
    for result in results:
        if isinstance(result, Exception):
            raise result

    return list(results)
