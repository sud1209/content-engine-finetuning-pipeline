from dataclasses import dataclass, field
from typing import List
import yaml

@dataclass
class TrainingConfig:
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "outputs/llama3-8b-tweet-scorer"
    max_seq_length: int = 2048

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training — Kaggle P100 16GB
    epochs: int = 3
    batch_size: int = 2
    grad_accum_steps: int = 8   # effective batch = 16
    learning_rate: float = 2e-4
    use_bf16: bool = False      # P100 does not support bf16

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
