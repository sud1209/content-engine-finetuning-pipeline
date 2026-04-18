"""
Export merged safetensor model to GGUF format (Q4_K_M quantization) for
CPU / Ollama deployment.

Requires llama.cpp tools on PATH. Install options:
  - brew install llama.cpp            (macOS)
  - apt install llama-cpp             (Debian/Ubuntu)
  - pip install llama-cpp-python      (Python bindings — convert only, no quantize CLI)
  - Build from source: https://github.com/ggerganov/llama.cpp

Typical usage:
    python scripts/export_gguf.py
    python scripts/export_gguf.py --input outputs/llama3-8b-tweet-scorer-merged \
                                   --output outputs/tweet-scorer-q4km.gguf \
                                   --quant q4_k_m
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_INPUT = "outputs/llama3-8b-tweet-scorer-merged"
DEFAULT_OUTPUT = "outputs/tweet-scorer-q4km.gguf"
DEFAULT_QUANT = "q4_k_m"

# Intermediate unquantized GGUF produced by convert step
_CONVERT_TMP = "outputs/tweet-scorer-f16.gguf"

# Names searched on PATH for the llama.cpp convert script / binary
_CONVERT_CANDIDATES = [
    "llama-convert",           # some distro packages
    "convert_hf_to_gguf.py",  # llama.cpp source tree (run via python)
    "convert.py",             # older llama.cpp source tree name
]
_QUANTIZE_CANDIDATES = [
    "llama-quantize",
    "llama_quantize",
    "quantize",               # built from source, in build/bin/
]


def _find_tool(candidates: list[str]) -> tuple[str, ...] | None:
    """Return a command tuple (interpreter + path) for the first found candidate."""
    for name in candidates:
        path = shutil.which(name)
        if path:
            cmd: tuple[str, ...] = (path,)
            if name.endswith(".py"):
                cmd = (sys.executable, path)
            return cmd
    return None


def _run(cmd: list[str], step: str) -> None:
    print(f"[{step}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {step} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def convert_to_gguf(input_dir: str, tmp_gguf: str) -> None:
    """Convert HF safetensors -> unquantized (F16) GGUF."""
    cmd_tuple = _find_tool(_CONVERT_CANDIDATES)
    if cmd_tuple is None:
        print(
            "ERROR: No llama.cpp convert tool found on PATH. "
            "Install llama.cpp or add convert_hf_to_gguf.py to PATH."
        )
        sys.exit(1)

    cmd = list(cmd_tuple) + [input_dir, "--outfile", tmp_gguf, "--outtype", "f16"]
    _run(cmd, "convert")


def quantize_gguf(tmp_gguf: str, output_path: str, quant_type: str) -> None:
    """Quantize F16 GGUF -> Q4_K_M (or other type)."""
    cmd_tuple = _find_tool(_QUANTIZE_CANDIDATES)
    if cmd_tuple is None:
        print(
            "ERROR: No llama-quantize binary found on PATH. "
            "Build llama.cpp from source or install via your package manager."
        )
        sys.exit(1)

    # llama-quantize syntax: llama-quantize <input> <output> <type>
    cmd = list(cmd_tuple) + [tmp_gguf, output_path, quant_type.upper()]
    _run(cmd, "quantize")


def export_gguf(
    input_dir: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    quant_type: str = DEFAULT_QUANT,
    keep_tmp: bool = False,
) -> None:
    """
    Full pipeline: HF safetensors -> F16 GGUF -> quantized GGUF.

    Parameters
    ----------
    input_dir   : Directory with merged safetensor model (tokenizer + weights).
    output_path : Destination .gguf file path.
    quant_type  : llama.cpp quantization type string, e.g. "q4_k_m", "q8_0".
    keep_tmp    : If True, retain the intermediate F16 GGUF.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_gguf = _CONVERT_TMP

    print(f"Step 1/2 — Converting {input_dir} to F16 GGUF at {tmp_gguf}...")
    convert_to_gguf(input_dir, tmp_gguf)

    print(f"Step 2/2 — Quantizing to {quant_type.upper()} -> {output_path}...")
    quantize_gguf(tmp_gguf, output_path, quant_type)

    if not keep_tmp:
        tmp = Path(tmp_gguf)
        if tmp.exists():
            tmp.unlink()
            print(f"Removed intermediate file {tmp_gguf}")

    size_mb = round(Path(output_path).stat().st_size / 1024 / 1024, 1)
    print(f"Export complete: {output_path} ({size_mb} MB)")
    print(
        "\nTo run with Ollama:\n"
        f"  ollama create tweet-scorer --file {output_path}\n"
        "  ollama run tweet-scorer"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export merged model to GGUF.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Merged safetensor model dir")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .gguf file path")
    parser.add_argument(
        "--quant",
        default=DEFAULT_QUANT,
        help="Quantization type (default: q4_k_m). See llama.cpp docs for options.",
    )
    parser.add_argument("--keep-tmp", action="store_true", help="Keep intermediate F16 GGUF")
    args = parser.parse_args()

    export_gguf(
        input_dir=args.input,
        output_path=args.output,
        quant_type=args.quant,
        keep_tmp=args.keep_tmp,
    )


if __name__ == "__main__":
    main()
