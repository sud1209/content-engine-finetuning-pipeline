import optuna
import wandb
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from training.config import TrainingConfig
from dotenv import load_dotenv
import os

load_dotenv()

N_TRIALS = 6  # 6 trials × ~30min each = ~3h on P100

def objective(trial: optuna.Trial) -> float:
    lora_rank = trial.suggest_categorical("lora_rank", [8, 16, 32])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 2e-4, 5e-4])

    cfg = TrainingConfig(lora_rank=lora_rank, learning_rate=learning_rate, epochs=1)

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "tweet-scorer-finetuning"),
        name=f"sweep-r{lora_rank}-lr{learning_rate}",
        config={"lora_rank": lora_rank, "learning_rate": learning_rate},
        reinit=True,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_rank, target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", use_gradient_checkpointing="unsloth", random_state=42,
    )

    dataset = load_dataset("sudar/tweet-scorer-dataset")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        args=SFTConfig(
            output_dir=f"outputs/sweep-r{lora_rank}-lr{learning_rate}",
            num_train_epochs=1,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum_steps,
            learning_rate=learning_rate,
            fp16=True, bf16=False,
            logging_steps=10, eval_strategy="epoch",
            report_to="wandb",
            dataset_text_field="messages",
            max_seq_length=cfg.max_seq_length,
            packing=True,
        ),
    )

    trainer.train()
    eval_results = trainer.evaluate()
    wandb.finish()
    return eval_results["eval_loss"]

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"Best: rank={study.best_params['lora_rank']}, lr={study.best_params['learning_rate']}, loss={study.best_value:.4f}")
