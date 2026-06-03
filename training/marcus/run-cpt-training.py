#!/usr/bin/env python
"""
Continued Pretraining (CPT) for Marcus Aurelius model.

This script performs continued pretraining on raw text from Meditations.
Unlike SFT which uses Q&A pairs, CPT trains on raw text passages to learn
Marcus' writing style and philosophical reasoning patterns.

Usage:
    .\unsloth_env\Scripts\python.exe training\marcus\run-cpt-training.py
"""

import sys
import os

# Import Unsloth FIRST (critical)
import unsloth
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from datasets import Dataset
import torch
from transformers import TrainingArguments


def load_meditations_text(filepath: str) -> list[str]:
    """Load Meditations text and split into passages."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by "BOOK" to get passages
    passages = []
    current_passage = []
    
    for line in text.split('\n'):
        if line.strip().startswith('BOOK'):
            if current_passage:
                passage_text = '\n'.join(current_passage).strip()
                if len(passage_text) > 100:  # Skip very short passages
                    passages.append(passage_text)
            current_passage = [line.strip()]
        else:
            current_passage.append(line.strip())
    
    # Don't forget the last passage
    if current_passage:
        passage_text = '\n'.join(current_passage).strip()
        if len(passage_text) > 100:
            passages.append(passage_text)
    
    print(f"[CPT] Loaded {len(passages)} passages from Meditations")
    return passages


def format_dataset(passages: list[str], eos_token: str) -> Dataset:
    """Format passages into dataset for CPT."""
    formatted_data = []
    for passage in passages:
        # Add EOS token at end of each passage
        text = passage + eos_token
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)


def train_cpt(
    model_path: str,
    meditations_path: str,
    output_path: str,
    max_seq_length: int = 2048,
    num_train_epochs: int = 1,
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
):
    """Train model with Continued Pretraining on raw text."""
    
    print("=" * 70)
    print(f"Marcus Aurelius Continued Pretraining (CPT)")
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
    
    # Set EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    
    # Load and format data
    print(f"\nLoading Meditations from: {meditations_path}")
    if not os.path.exists(meditations_path):
        # Try relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fallback_path = os.path.join(script_dir, "..", meditations_path)
        if os.path.exists(fallback_path):
            meditations_path = os.path.normpath(fallback_path)
        else:
            raise FileNotFoundError(f"Meditations file not found: {meditations_path}")
    
    print(f"Using Meditations file: {meditations_path}")
    passages = load_meditations_text(meditations_path)
    
    dataset = format_dataset(passages, tokenizer.eos_token)
    print(f"Formatted {len(dataset)} training samples")
    
    # Configure trainer
    training_args = UnslothTrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=50,
        max_steps=500,
        learning_rate=learning_rate,
        embedding_learning_rate=learning_rate / 10,
        fp16=True,
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )
    
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=4,
        args=training_args,
    )
    
    # Train
    print("\nStarting CPT training...")
    trainer_stats = trainer.train()
    
    # Save model
    print(f"\nSaving model to: {output_path}")
    trainer.save_model(output_path)
    
    print("\n✅ CPT training complete!")
    print(f"   Output: {output_path}")
    print(f"   Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"   Final loss: {trainer_stats.metrics['train_loss']:.4f}")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="CPT training for Marcus Aurelius")
    parser.add_argument("--model", default="unsloth/Qwen3-4B",
                       help="Base model to fine-tune")
    parser.add_argument("--meditations", default="../meditations.mb.txt",
                       help="Path to Meditations text file")
    parser.add_argument("--output", default="training/output-marcus/qwen3-4b-marcus-cpt",
                       help="Output directory for trained model")
    parser.add_argument("--max-seq", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    args = parser.parse_args()
    
    try:
        train_cpt(
            model_path=args.model,
            meditations_path=args.meditations,
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
