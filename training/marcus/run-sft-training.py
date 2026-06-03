#!/usr/bin/env python
"""
SFT training script for Marcus Aurelius model.

This script performs Supervised Fine-Tuning using the Unsloth library.
"""

import sys
import os

# Import Unsloth FIRST (critical)
import unsloth
from unsloth import FastLanguageModel
from datasets import Dataset
import torch
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from peft import LoraConfig


def load_training_data(data_path: str) -> Dataset:
    """Load training data from JSON file."""
    import json
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to chat format expected by SFTTrainer
    formatted_data = []
    for item in data:
        messages = item.get("messages", [])
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)


def train_sft(
    model_path: str,
    data_path: str,
    output_path: str,
    max_seq_length: int = 4096,
    num_train_epochs: int = 2,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
):
    """Train model with SFT."""
    
    print("=" * 70)
    print(f"Marcus Aurelius SFT Training")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=True,  # 8-bit for stability
        dtype=torch.float16,
    )
    print("Model loaded.")
    
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
    
    # Configure trainer
    training_args = SFTConfig(
        output_dir=output_path,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=50,
        max_steps=1000,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        dataloader_drop_last=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )
    
    # Train
    print("\nStarting SFT training...")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to: {output_path}")
    trainer.save_model(output_path)
    
    print("\n✅ SFT training complete!")
    print(f"   Output: {output_path}")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="SFT training for Marcus Aurelius")
    parser.add_argument("--model", default="unsloth/Qwen3-4B",
                       help="Base model to fine-tune")
    parser.add_argument("--data", default="marcus_training_data.json",
                       help="Training data JSON file")
    parser.add_argument("--output", default="training/output-marcus/qwen3-4b-marcus-sft",
                       help="Output directory for trained model")
    parser.add_argument("--max-seq", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    args = parser.parse_args()
    
    try:
        train_sft(
            model_path=args.model,
            data_path=args.data,
            output_path=args.output,
            max_seq_length=args.max_seq,
            num_train_epochs=args.epochs,
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
