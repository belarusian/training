"""
Train Qwen3-4B for Effect TypeScript code generation using GRPO
"""

import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer

# Configuration
MAX_SEQ_LENGTH = 4096
LORA_RANK = 64
DATA_PATH = "extracted-code/effect-code-samples.json"

# Reward function
def reward_fn(completions):
    """Reward function to score generated code quality."""
    rewards = []
    for text in completions:
        score = 0.0
        
        # Reward: Has <CODE> tags
        if "<CODE>" in text and "</CODE>" in text:
            score += 1.0
        
        # Reward: Has Effect imports
        if "from effect" in text or "import Effect" in text:
            score += 0.5
        
        # Reward: Has Schema usage
        if "Schema" in text:
            score += 0.3
        
        # Reward: Proper TypeScript syntax indicators
        if "export " in text:
            score += 0.2
        
        # Penalize: Too short responses
        if len(text) < 100:
            score -= 0.5
        
        rewards.append(score)
    
    return torch.tensor(rewards)


def load_training_data():
    """Load training data from JSON file."""
    print(f"Loading training data from {DATA_PATH}...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training samples")
    return data


def format_dataset_for_sft(samples):
    """Format samples for SFT training."""
    system_prompt = """You are an expert TypeScript developer specializing in the Effect framework.
Analyze the requirements and generate high-quality Effect code.
Provide your reasoning between <start_working_out> and <end_working_out>.
Then, provide your complete TypeScript code implementation between <CODE> and </CODE>."""
    
    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    code_start = "<CODE>"
    code_end = "</CODE>"
    
    formatted_samples = []
    for sample in samples:
        prompt = sample.get("prompt", "")
        completion = sample.get("completion", "")
        path = sample.get("path", "unknown")
        source = sample.get("source", "unknown")
        
        # Create reasoning
        reasoning = f"""Analyzing requirement: {prompt}
Source: {source}, Path: {path}
I need to generate TypeScript code using Effect framework patterns.
This will include proper imports from effect/*, Schema definitions, and Effect service patterns."""
        
        # Format output
        final_output = (
            reasoning_start + reasoning + reasoning_end +
            code_start + completion + code_end
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_output},
        ]
        
        formatted_samples.append(messages)
    
    return formatted_samples


def train_sft(model, tokenizer, formatted_samples):
    """Train with SFT (Supervised Fine-Tuning)."""
    print("\n=== Training with SFT ===")
    
    from datasets import Dataset
    
    # Create dataset
    texts = []
    for messages in formatted_samples:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    
    dataset = Dataset.from_dict({"text": texts})
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )
    
    trainer.train()
    return trainer


def train_grpo(model, tokenizer, samples):
    """Train with GRPO (Reinforcement Learning)."""
    print("\n=== Training with GRPO ===")
    
    system_prompt = """You are an expert TypeScript developer specializing in the Effect framework.
Analyze the requirements and generate high-quality Effect code.
Provide your reasoning between <start_working_out> and <end_working_out>.
Then, provide your complete TypeScript code implementation between <CODE> and </CODE>."""
    
    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    code_start = "<CODE>"
    code_end = "</CODE>"
    
    def format_for_grpo(x):
        prompt = x.get("prompt", "")
        completion = x.get("completion", "")
        path = x.get("path", "unknown")
        source = x.get("source", "unknown")
        
        reasoning = f"""Analyzing requirement: {prompt}
Source: {source}, Path: {path}
I need to generate TypeScript code using Effect framework patterns."""
        
        final_output = (
            reasoning_start + reasoning + reasoning_end +
            code_start + completion + code_end
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_output},
        ]
    
    # Create GRPO dataset
    texts = []
    for sample in samples:
        messages = format_for_grpo(sample)
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    
    dataset = Dataset.from_dict({"text": texts})
    
    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-6,
            num_epochs=1,
            warmup_steps=5,
            logging_steps=5,
            report_to="none",
            reward_functions=[reward_fn],
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    return trainer


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Training Qwen3-4B for Effect TypeScript Code Generation")
    print("=" * 60)
    
    # Load data
    samples = load_training_data()
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=True,  # Use vLLM for faster training
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.9,
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Format for SFT
    formatted_samples = format_dataset_for_sft(samples)
    
    # Train with SFT first
    train_sft(model, tokenizer, formatted_samples)
    
    # Train with GRPO
    train_grpo(model, tokenizer, samples)
    
    # Save model
    print("\nSaving model...")
    model.save_pretrained("qwen3-4b-effect-codegen")
    tokenizer.save_pretrained("qwen3-4b-effect-codegen")
    
    print("\n" + "=" * 60)
    print("Training complete! Model saved to: qwen3-4b-effect-codegen/")
    print("=" * 60)


if __name__ == "__main__":
    main()
