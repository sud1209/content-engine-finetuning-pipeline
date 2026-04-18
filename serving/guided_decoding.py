"""
JSON schema for vLLM guided decoding of tweet quality scores.

Pass SCORE_OUTPUT_SCHEMA as the `guided_json` argument to the vLLM
completions endpoint so that token sampling is constrained to valid
score objects on every forward pass.
"""

SCORE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "hook_strength":            {"type": "integer", "minimum": 1, "maximum": 10},
        "tone_compliance":          {"type": "integer", "minimum": 0, "maximum": 10},
        "x_algorithm_optimization": {"type": "integer", "minimum": 1, "maximum": 10},
        "data_specificity":         {"type": "integer", "minimum": 1, "maximum": 10},
        "pillar_alignment":         {"type": "integer", "minimum": 1, "maximum": 10},
        "cta_quality":              {"type": "integer", "minimum": 1, "maximum": 10},
        "never_list_violation":     {"type": "boolean"},
        "reasoning":                {"type": "string", "maxLength": 500},
    },
    "required": [
        "hook_strength",
        "tone_compliance",
        "x_algorithm_optimization",
        "data_specificity",
        "pillar_alignment",
        "cta_quality",
        "never_list_violation",
        "reasoning",
    ],
    "additionalProperties": False,
}
