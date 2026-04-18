import wandb
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from training.config import TrainingConfig
from dotenv import load_dotenv
import os

load_dotenv()

def main(config_path: str = "training/configs/llama3_8b_qlora.yaml"):
    cfg = TrainingConfig.from_yaml(config_path)

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "tweet-scorer-finetuning"),
        name=f"llama3-8b-qlora-r{cfg.lora_rank}",
        config=cfg.__dict__,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = load_dataset("sudar/tweet-scorer-dataset")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum_steps,
            warmup_ratio=0.05,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type="cosine",
            fp16=not cfg.use_bf16,
            bf16=cfg.use_bf16,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="wandb",
            dataset_text_field="messages",
            max_seq_length=cfg.max_seq_length,
            packing=True,
            dataset_num_proc=2,
        ),
    )

    trainer.train()

    model.save_pretrained(f"{cfg.output_dir}/final_adapter")
    tokenizer.save_pretrained(f"{cfg.output_dir}/final_adapter")

    hf_repo = os.environ.get("HF_MODEL_REPO", "sudar/tweet-scorer-llama3-8b")
    model.push_to_hub(hf_repo)
    tokenizer.push_to_hub(hf_repo)

    wandb.finish()

if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "training/configs/llama3_8b_qlora.yaml"
    main(config)
