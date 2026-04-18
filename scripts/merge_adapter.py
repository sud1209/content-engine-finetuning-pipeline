"""
Merge LoRA adapter weights into the base model using Unsloth.

Reads from:  outputs/llama3-8b-tweet-scorer/final_adapter/
Writes to:   outputs/llama3-8b-tweet-scorer-merged/

The merged directory contains full safetensor weights ready for:
  - GGUF export (scripts/export_gguf.py)
  - vLLM / TGI serving without an adapter runtime
"""

from unsloth import FastLanguageModel

ADAPTER_PATH = "outputs/llama3-8b-tweet-scorer/final_adapter"
MERGED_PATH = "outputs/llama3-8b-tweet-scorer-merged"


def main() -> None:
    print(f"Loading adapter from {ADAPTER_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        ADAPTER_PATH,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged weights to {MERGED_PATH}...")
    model.save_pretrained(MERGED_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_PATH)

    print(f"Merged weights saved to {MERGED_PATH}")


if __name__ == "__main__":
    main()
