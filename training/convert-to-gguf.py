r"""
Convert trained LoRA model to GGUF Q8_0 format for llama.cpp inference.

Usage: .\unsloth_env\Scripts\python.exe training\convert-to-gguf.py
       .\unsloth_env\Scripts\python.exe training\convert-to-gguf.py --model <path>

Options:
    --model PATH    Path to trained model (default: training/output-v2/qwen3-4b-effect-codegen-v2)
    --quant Q       Quantization method (default: q8_0)
"""

import sys
import argparse
import torch
from unsloth import FastLanguageModel


def convert_to_gguf(model_path, quant_method="q8_0"):
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
    
    # Save as GGUF f16 first
    print(f"\nConverting to GGUF f16 format...")
    print("This may take several minutes...")
    model.save_pretrained_gguf(model_path, tokenizer, quantization_method="f16")
    
    print(f"\n✅ GGUF f16 model saved to {model_path}")
    
    # Now quantize to Q8_0 using gguf library
    print(f"\nQuantizing to GGUF {quant_method} format...")
    print("This may take several minutes...")
    
    import numpy as np
    from gguf import GGUFReader, GGUFWriter, quantize, GGMLQuantizationType
    
    input_path = f"{model_path}/Qwen3-4B.F16.gguf"
    output_path = f"{model_path}/Qwen3-4B.Q8_0.gguf"
    
    print(f"Reading GGUF file: {input_path}")
    reader = GGUFReader(input_path)
    metadata = {k: v.data for k, v in reader.fields.items()}
    
    print(f"Creating quantized GGUF: {output_path}")
    writer = GGUFWriter(output_path, "qwen3")
    
    # Copy metadata
    for key, data in metadata.items():
        if key == 'general.architecture':
            writer.add_architecture()
            writer.arch = data[0].decode() if isinstance(data[0], bytes) else str(data[0])
        elif key == 'general.type':
            writer.add_type(data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'general.name':
            writer.add_name(data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'general.version':
            writer.add_version(data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'general.finetune':
            writer.add_finetune(data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'general.basename':
            writer.add_basename(data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'general.quantized_by':
            writer.add_quantized_by(data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'general.size_label':
            writer.add_size_label(data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'general.repo_url':
            writer.add_repo_url(data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'general.tags':
            tags = [t.decode() if isinstance(t, bytes) else str(t) for t in data]
            writer.add_tags(tags)
        elif key == 'general.base_model.count':
            writer.add_base_model_count(data[0])
        elif key == 'general.base_model.0.name':
            writer.add_base_model_name(0, data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'general.base_model.0.organization':
            writer.add_base_model_organization(0, data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'general.base_model.0.repo_url':
            writer.add_base_model_repo_url(0, data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'qwen3.block_count':
            writer.add_block_count(data[0])
        elif key == 'qwen3.context_length':
            writer.add_context_length(data[0])
        elif key == 'qwen3.embedding_length':
            writer.add_embedding_length(data[0])
        elif key == 'qwen3.feed_forward_length':
            writer.add_feed_forward_length(data[0])
        elif key == 'qwen3.attention.head_count':
            writer.add_head_count(data[0])
        elif key == 'qwen3.attention.head_count_kv':
            writer.add_head_count_kv(data[0])
        elif key == 'qwen3.rope.freq_base':
            writer.add_rope_freq_base(data[0])
        elif key == 'qwen3.attention.layer_norm_rms_epsilon':
            writer.add_layer_norm_rms_eps(data[0])
        elif key == 'qwen3.attention.key_length':
            writer.add_key_length(data[0])
        elif key == 'qwen3.attention.value_length':
            writer.add_value_length(data[0])
        elif key == 'general.file_type':
            writer.add_file_type(data[0])
        elif key == 'general.quantization_version':
            writer.add_quantization_version(data[0])
        elif key == 'tokenizer.ggml.model':
            writer.add_tokenizer_model(data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'tokenizer.ggml.pre':
            writer.add_tokenizer_pre(data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
        elif key == 'tokenizer.ggml.tokens':
            tokens = [t.decode() if isinstance(t, bytes) else str(t) for t in data]
            writer.add_token_list(tokens)
        elif key == 'tokenizer.ggml.merges':
            merges = [m.decode() if isinstance(m, bytes) else str(m) for m in data]
            writer.add_token_merges(merges)
        elif key == 'tokenizer.ggml.eos_token_id':
            writer.add_eos_token_id(data[0])
        elif key == 'tokenizer.ggml.padding_token_id':
            writer.add_pad_token_id(data[0])
        elif key == 'tokenizer.ggml.add_bos_token':
            writer.add_add_bos_token(data[0])
        elif key == 'tokenizer.chat_template':
            writer.add_chat_template(data[0].decode() if isinstance(data[0], bytes) else str(data[0]))
    
    # Quantize and write tensors
    print(f"Quantizing {len(reader.tensors)} tensors to Q8_0...")
    for i, tensor in enumerate(reader.tensors):
        if i % 50 == 0:
            print(f"  [{i}/{len(reader.tensors)}] Quantizing {tensor.name} ({tensor.shape})...")
        
        data = tensor.data
        quantized_data = quantize(data, GGMLQuantizationType.Q8_0)
        
        writer.add_tensor(
            name=tensor.name,
            tensor=quantized_data,
            raw_dtype=GGMLQuantizationType.Q8_0
        )
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print(f"\n✅ GGUF {quant_method} model saved to {output_path}")
    print(f"   You can now use this with llama.cpp or Ollama")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert trained model to GGUF")
    parser.add_argument("--model", default="training/output-v2/qwen3-4b-effect-codegen-v2",
                        help="Path to trained model")
    parser.add_argument("--quant", default="q8_0",
                        choices=["q8_0", "f16"],
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
