"""
Test Unsloth GPU setup for training
"""

import torch
from unsloth import FastLanguageModel

print("Testing Unsloth GPU setup...")

# Check CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test model loading
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=4096,
        load_in_4bit=False,
        fast_inference=False,  # Don't use vllm for testing
    )
    print("[OK] Model loaded successfully!")
    
    # Test LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=128,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("[OK] LoRA configured successfully!")
    
    print("\nAll tests passed! Ready for training.")
    
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    raise
