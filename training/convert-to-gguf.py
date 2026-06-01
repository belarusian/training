r"""
Convert trained LoRA model to GGUF Q4_K_M format for llama.cpp inference.

Usage: .\unsloth_env\Scripts\python.exe training\convert-to-gguf.py
       .\unsloth_env\Scripts\python.exe training\convert-to-gguf.py --model <path>

Options:
    --model PATH    Path to trained model (default: training/output-v2/qwen3-4b-effect-codegen-v2)
    --quant Q       Quantization method (default: q4_k_m)
"""

import sys
import argparse
import torch
from unsloth import FastLanguageModel


def convert_to_gguf(model_path, quant_method="q4_k_m"):
    """Convert a trained LoRA model to GGUF format."""
    
    print("=" * 70)
    print(f"Converting {model_path} to GGUF {quant_method}")
    print("=" * 70)
    
    # Load the trained model
    print(f"\nLoading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=False,
        load_in_8bit=False,
        dtype=torch.float16,
    )
    print(f"Model loaded.")
    
    # Merge LoRA adapters
    print("\nMerging LoRA adapters...")
    model = model.merge_and_unload()
    print("LoRA merged.")
    
    # Save as GGUF
    print(f"\nConverting to GGUF {quant_method} format...")
    print("This may take several minutes...")
    model.save_pretrained_gguf(model_path, tokenizer, quantization_method=quant_method)
    
    print(f"\n✅ GGUF model saved to {model_path}")
    print(f"   You can now use this with llama.cpp or Ollama")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert trained model to GGUF")
    parser.add_argument("--model", default="training/output-v2/qwen3-4b-effect-codegen-v2",
                        help="Path to trained model")
    parser.add_argument("--quant", default="q4_k_m",
                        choices=["q4_k_m", "q8_0", "q5_k_m", "f16"],
                        help="Quantization method")
    args = parser.parse_args()
    
    try:
        convert_to_gguf(args.model, args.quant)
    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}")
        print("\nCommon issues:")
        print("  1. Make sure you ran the training first")
        print("  2. Make sure llama.cpp is installed (unsloth handles this)")
        print("  3. Check that the model path is correct")
        sys.exit(1)
