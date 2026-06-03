#!/usr/bin/env python
"""
GRPO training script for Marcus Aurelius model.

This script performs Group Relative Policy Optimization using the Unsloth library.
"""

import sys
import os

# Import Unsloth FIRST (critical)
import unsloth
from unsloth import FastLanguageModel
from datasets import Dataset
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from peft import LoraConfig


# Import reward functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'reward-functions.py'))
from reward_functions import calculate_reward


def load_training_data(data_path: str) -> Dataset:
    """Load training data from JSON file."""
    import json
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to chat format
    formatted_data = []
    for item in data:
        messages = item.get("messages", [])
        # Use only user message as prompt
        user_message = next((m['content'] for m in messages if m['role'] == 'user'), '')
        formatted_data.append({
            "prompt": user_message,
            "messages": messages
        })
    
    return Dataset.from_list(formatted_data)


def reward_function(
    prompt: str,
    completion: str,
    messages: List[str],
    expected_citation: str = None,
) -> float:
    """
    Reward function for GRPO training.
    
    Rewards:
    - Citation present: +1.0
    - Citation format: +0.5
    - Stoic principles: +0.3
    - Response length (200-500 chars): +0.5
    - Response quality: +0.5
    """
    return calculate_reward(completion, expected_citation)


def train_grpo(
    model_path: str,
    data_path: str,
    output_path: str,
    sft_adapter_path: str = None,
    max_seq_length: int = 4096,
    learning_rate: float = 2e-6,
    per_device_train_batch_size: int = 4,
    num_generations: int = 4,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 1,
):
    """Train model with GRPO."""
    
    print("=" * 70)
    print(f"Marcus Aurelius GRPO Training")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=True,
        dtype=torch.float16,
    )
    print("Model loaded.")
    
    # Load SFT adapter if provided
    if sft_adapter_path:
        print(f"\nLoading SFT adapter: {sft_adapter_path}")
        model.load_adapter(sft_adapter_path)
        print("SFT adapter loaded.")
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Load data
    print(f"\nLoading training data: {data_path}")
    dataset = load_training_data(data_path)
    print(f"Loaded {len(dataset)} training samples")
    
    # Prepare dataset for GRPO
    def format_sample(sample):
        return {
            "prompt": sample["prompt"],
            "messages": sample["messages"],
        }
    
    dataset = dataset.map(format_sample)
    
    # Configure trainer
    training_args = GRPOConfig(
        output_dir=output_path,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=50,
        max_steps=500,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )
    
    trainer = GRPOTrainer(
        config=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_function],
    )
    
    # Train
    print("\nStarting GRPO training...")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to: {output_path}")
    trainer.save_model(output_path)
    
    print("\n✅ GRPO training complete!")
    print(f"   Output: {output_path}")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="GRPO training for Marcus Aurelius")
    parser.add_argument("--model", default="unsloth/Qwen3-4B",
                       help="Base model to fine-tune")
    parser.add_argument("--data", default="marcus_training_data.json",
                       help="Training data JSON file")
    parser.add_argument("--output", default="training/output-marcus/qwen3-4b-marcus-v2",
                       help="Output directory for trained model")
    parser.add_argument("--sft-adapter", default=None,
                       help="Path to SFT adapter (optional)")
    parser.add_argument("--max-seq", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=2e-6,
                       help="Learning rate")
    parser.add_argument("--eval-only", action="store_true",
                       help="Run evaluation only (no training)")
    args = parser.parse_args()
    
    try:
        if args.eval_only:
            # Run evaluation only
            print("Running post-training evaluation...")
            # Evaluation logic here
        else:
            train_grpo(
                model_path=args.model,
                data_path=args.data,
                output_path=args.output,
                sft_adapter_path=args.sft_adapter,
                learning_rate=args.lr,
            )
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    main()
