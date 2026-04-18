import json
import time
from pathlib import Path

from unsloth import FastLanguageModel

from dataset.schemas import SCORING_SYSTEM_PROMPT, DIMENSIONS, WEIGHTS


def score_with_model(model, tokenizer, content: str) -> dict:
    messages = [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {"role": "user", "content": f"Score this tweet draft:\n\n{content}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    start = time.perf_counter()
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=256, temperature=0.0, do_sample=False
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    raw_text = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        result = {d: 5 for d in DIMENSIONS}
        result.update({"never_list_violation": False, "reasoning": "parse_error"})
    result["latency_ms"] = latency_ms
    # Compute composite score
    if result.get("never_list_violation"):
        result["composite_score"] = 0.0
    else:
        raw = sum(result.get(d, 5) * w / 100 for d, w in WEIGHTS.items())
        result["composite_score"] = round(raw + 0.5, 2)
    return result


def run_benchmark(
    ft_repo: str = "sudar/tweet-scorer-llama3-8b",
    base_repo: str = "meta-llama/Llama-3.1-8B-Instruct",
    test_path: str = "data/raw/test_ground_truth.jsonl",
    output_dir: str = "benchmark/results",
) -> None:
    test_data = [json.loads(l) for l in open(test_path)]

    print(f"Loading fine-tuned model from {ft_repo}...")
    ft_model, tokenizer = FastLanguageModel.from_pretrained(
        ft_repo, max_seq_length=2048, load_in_4bit=True
    )
    FastLanguageModel.for_inference(ft_model)

    print(f"Loading base model from {base_repo}...")
    base_model, _ = FastLanguageModel.from_pretrained(
        base_repo, max_seq_length=2048, load_in_4bit=True
    )
    FastLanguageModel.for_inference(base_model)

    results = []
    for i, example in enumerate(test_data):
        content = example["content"]
        results.append(
            {
                "id": example["id"],
                "content": content,
                "haiku": example["scores"],
                "finetuned": score_with_model(ft_model, tokenizer, content),
                "base": score_with_model(base_model, tokenizer, content),
            }
        )
        if (i + 1) % 20 == 0:
            print(f"Scored {i+1}/{len(test_data)}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_path = f"{output_dir}/benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    from benchmark.metrics import compute_metrics
    from benchmark.visualize import generate_benchmark_card

    metrics = compute_metrics(results)
    print(json.dumps(metrics, indent=2))

    card_path = f"{output_dir}/benchmark_card"
    generate_benchmark_card(metrics, card_path)


if __name__ == "__main__":
    run_benchmark()
