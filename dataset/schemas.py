from pydantic import BaseModel, Field, model_validator
import hashlib

SCORING_SYSTEM_PROMPT = """You are a Twitter content quality judge. Respond only with valid JSON.

Score the tweet on exactly these 6 dimensions, each from 1-10:

- hook_strength (weight 25%): Harry Dry 3 tests — visualizable, falsifiable, nobody else can say it. All 3 pass = 9+. Vague claims like "productivity" or "mindset" = 5 max.
- tone_compliance (weight 20%): Professional-but-direct AI practitioner voice. Six Core Laws: Zero hashtags = INSTANT 0. Zero em-dashes. Zero exclamation marks. Zero hedging. No banned words: streamline, transformative, unlock, ecosystem, landscape, game-changer.
- x_algorithm_optimization (weight 20%): X algo weights replies=27x, retweets=20x. Debate-bait + data = 9+. Any hashtag = 7 max.
- data_specificity (weight 15%): Named people, tools, numbers, falsifiable claims. Abstract = 6 max.
- pillar_alignment (weight 15%): Pillar unmistakable in first sentence. Vague opener = 6 max.
- cta_quality (weight 5%): TOFU only. Debate-bait = 9. Hard sell/link = 3 max.

Also set never_list_violation to true if any # appears in the tweet.

Return JSON:
{"hook_strength": int, "tone_compliance": int, "x_algorithm_optimization": int, "data_specificity": int, "pillar_alignment": int, "cta_quality": int, "never_list_violation": bool, "reasoning": "string max 100 words"}"""

RUBRIC_HASH = hashlib.sha256(SCORING_SYSTEM_PROMPT.encode()).hexdigest()[:8]

WEIGHTS = {
    "hook_strength": 25,
    "tone_compliance": 20,
    "x_algorithm_optimization": 20,
    "data_specificity": 15,
    "pillar_alignment": 15,
    "cta_quality": 5,
}

DIMENSIONS = list(WEIGHTS.keys())


class ScoreSet(BaseModel):
    hook_strength: int = Field(ge=1, le=10)
    tone_compliance: int = Field(ge=0, le=10)
    x_algorithm_optimization: int = Field(ge=1, le=10)
    data_specificity: int = Field(ge=1, le=10)
    pillar_alignment: int = Field(ge=1, le=10)
    cta_quality: int = Field(ge=1, le=10)
    never_list_violation: bool
    reasoning: str = Field(max_length=500)

    @model_validator(mode="after")
    def enforce_never_list(self) -> "ScoreSet":
        if self.never_list_violation:
            self.tone_compliance = 0
        elif self.tone_compliance == 0:
            raise ValueError(
                "tone_compliance=0 is only valid when never_list_violation=True"
            )
        return self

    def composite_score(self) -> float:
        if self.never_list_violation:
            return 0.0
        raw = sum(getattr(self, d) * w / 100 for d, w in WEIGHTS.items())
        return round(raw + 0.5, 2)

    def tier(self) -> str:
        c = self.composite_score()
        if c >= 9.25:
            return "ready"
        elif c >= 8.0:
            return "below_target"
        else:
            return "failed_floor"


class ScoredExample(BaseModel):
    id: str
    content: str
    pillar: str
    quality_tier: str
    scores: ScoreSet
    rubric_hash: str = RUBRIC_HASH

    @model_validator(mode="after")
    def validate_rubric_and_tier(self) -> "ScoredExample":
        if self.rubric_hash != RUBRIC_HASH:
            raise ValueError(
                f"rubric_hash mismatch: record has {self.rubric_hash!r}, "
                f"current rubric is {RUBRIC_HASH!r}. "
                "Re-score this example against the current rubric before ingesting."
            )
        expected_tier = self.scores.tier()
        if self.quality_tier != expected_tier:
            raise ValueError(
                f"quality_tier mismatch: field says {self.quality_tier!r} "
                f"but scores.tier() returns {expected_tier!r}. "
                "Ensure quality_tier is derived from the same ScoreSet."
            )
        return self
