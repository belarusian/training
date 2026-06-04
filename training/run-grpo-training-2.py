"""
Training Qwen3-4B for Effect TypeScript Code Generation - Stage 2
Group Relative Policy Optimization (GRPO) reinforcement fine-tuning.

This is Stage 2 of 2 in the two-stage training pipeline.
Requires run-sft-training-1.py to have completed first.

Output: LoRA adapter saved to training/output-v2/qwen3-4b-effect-codegen-v2/
"""

import os
import sys
import json
import time
import torch
from datetime import datetime

# ---------------------------------------------------------------------------
# Pre-flight checks: Ensure SFT was completed
# ---------------------------------------------------------------------------

print("=" * 70)
print("Pre-flight checks")
print("=" * 70)

# Check CUDA
if not torch.cuda.is_available():
    print("FATAL: CUDA is not available.")
    sys.exit(1)
print(f"  CUDA available:       Yes")

# Check required packages
required_packages = {"unsloth": "unsloth", "datasets": "datasets", "trl": "trl"}
missing = []
for pkg_name, import_name in required_packages.items():
    try:
        __import__(import_name)
    except ImportError:
        missing.append(pkg_name)

if missing:
    print(f"FATAL: Missing packages: {', '.join(missing)}")
    sys.exit(1)
print(f"  Required packages:    OK")

WAND_AVAILABLE = False
try:
    import wandb
    WAND_AVAILABLE = True
    print(f"  wandb:                Available (logging enabled)")
except ImportError:
    print(f"  wandb:                Not installed (optional)")

# Check SFT output exists
SFT_OUTPUT_DIR = "training/output-v2/qwen3-4b-effect-codegen-sft"
if not os.path.exists(SFT_OUTPUT_DIR):
    print(f"\nFATAL: SFT adapter not found at: {SFT_OUTPUT_DIR}")
    print("Run run-sft-training-1.py FIRST to train the SFT adapter.")
    print("Then run this script to continue with GRPO training.")
    sys.exit(1)

print(f"  SFT adapter:          Found at {SFT_OUTPUT_DIR}")

# Check training data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "extracted-code", "effect-code-samples.json")

if not os.path.exists(DATA_PATH):
    print(f"FATAL: Training data not found at: {DATA_PATH}")
    sys.exit(1)

print(f"  Training data:        {os.path.getsize(DATA_PATH) / 1e6:.1f} MB")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "effect-codegen-grpo"
os.environ["WANDB_LOG_MODEL"] = "false"

# ---------------------------------------------------------------------------
# 1. Load model from SFT output
# ---------------------------------------------------------------------------

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

MAX_SEQ_LENGTH = 4096
RANK = 64

print(f"Loading SFT adapter from {SFT_OUTPUT_DIR}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_OUTPUT_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
    dtype=torch.float16,
    fast_inference=False,
    max_lora_rank=RANK,
    gpu_memory_utilization=0.85,
)
print("SFT adapter loaded.")

# Merge SFT adapter into base model so GRPO trains on top of SFT-finetuned model
print("Merging SFT adapter into base model...")
model = model.merge_and_unload()
print("SFT adapter merged.\n")

# ---------------------------------------------------------------------------
# 2. Configure LoRA adapters
# ---------------------------------------------------------------------------

model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=RANK,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("GRPO LoRA adapters configured.\n")

# ---------------------------------------------------------------------------
# 3. Load and format training data
# ---------------------------------------------------------------------------

print(f"Loading data from: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples\n")

SYSTEM_PROMPT = """You are an expert TypeScript developer specializing in the Effect framework.
Analyze the requirements and generate high-quality Effect code.
Provide your reasoning first, then provide your complete TypeScript code implementation."""

def format_sample(x):
    prompt = x["prompt"]
    completion = x["completion"]
    path = x.get("path", "unknown")
    source = x.get("source", "unknown")
    
    reasoning = f"Analyzing requirement: {prompt}\nSource: {source}, Path: {path}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"{reasoning}\n\n{completion}"},
    ]
    
    return {"Messages": messages}

hf_dataset = Dataset.from_list([format_sample(s) for s in data])

def to_text(x):
    return {"text": tokenizer.apply_chat_template(x["Messages"], tokenize=False)}

hf_dataset = hf_dataset.map(to_text, batched=False)

# GRPO needs a "prompt" field for generation
def add_prompt(x):
    messages = x["Messages"]
    # Extract just the user message as the prompt
    user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
    return {"prompt": user_message}

hf_dataset = hf_dataset.map(add_prompt, batched=False)

print(f"Formatted dataset: {len(hf_dataset)} samples\n")

# ---------------------------------------------------------------------------
# 4. Reward function
# ---------------------------------------------------------------------------

def compute_reward_details(completions):
    rewards = []
    reward_components = []
    
    for text in completions:
        components = {
            "code_tags": 0.0,
            "effect_imports": 0.0,
            "schema_usage": 0.0,
            "exports": 0.0,
            "length": 0.0,
        }
        
        if "<CODE>" in text and "</CODE>" in text:
            components["code_tags"] = 1.0
        
        if "from effect" in text.lower() or "import Effect" in text:
            components["effect_imports"] = 0.5
        
        if "Schema" in text:
            components["schema_usage"] = 0.3
        
        if "export " in text:
            components["exports"] = 0.2
        
        # Extract code between <CODE> tags
        import re
        code_match = re.search(r'<CODE>(.*?)</CODE>', text, re.DOTALL)
        code_content = code_match.group(1) if code_match else ""
        code_len = len(code_content)
        
        # Penalize very short code content heavily - model needs to generate actual code
        if code_len < 100:
            components["length"] = -2.0  # Heavy penalty for minimal code
        elif code_len < 200:
            components["length"] = -1.0  # Still too short
        elif code_len < 500:
            components["length"] = -0.5
        elif code_len < 1000:
            components["length"] = 0.2
        elif code_len < 2000:
            components["length"] = 0.4
        elif code_len < 5000:
            components["length"] = 0.6
        elif code_len < 10000:
            components["length"] = 0.8
        else:
            components["length"] = 0.5
        
        total = sum(components.values())
        rewards.append(total)
        reward_components.append(components)
    
    return rewards, reward_components


def reward_fn(prompts, completions, **kwargs):
    rewards, _ = compute_reward_details(completions)
    return rewards


test_completion = '<CODE>import * as Effect from "effect"\nexport const test = Effect.succeed(42)</CODE>'
test_reward, test_components = compute_reward_details([test_completion])
print("Test reward:", test_reward[0])
print("Test components:", test_components[0])
print()

# ---------------------------------------------------------------------------
# 5. GRPO Training
# ---------------------------------------------------------------------------

print("=" * 70)
print("GRPO REINFORCEMENT TRAINING")
print("=" * 70)

grpo_config = GRPOConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-6,
    num_train_epochs=1,
    warmup_steps=5,
    logging_steps=5,
    fp16=True,
    bf16=False,
    report_to="wandb" if WAND_AVAILABLE else "none",
    output_dir="training/output-v2/qwen3-4b-effect-codegen-v2",
    save_steps=50,
    save_total_limit=2,
)

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=grpo_config,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
)

print("Starting GRPO training...")
grpo_start = time.time()
grpo_trainer.train()
grpo_time = time.time() - grpo_start

print(f"\nGRPO training complete. Time: {grpo_time/60:.1f} min")

FINAL_OUTPUT_DIR = "training/output-v2/qwen3-4b-effect-codegen-v2"
grpo_trainer.save_model(FINAL_OUTPUT_DIR)
tokenizer.save_pretrained(FINAL_OUTPUT_DIR)

print(f"Final model saved to {FINAL_OUTPUT_DIR}")

print("\n" + "=" * 70)
print("STAGE 2 COMPLETE")
print("=" * 70)
print(f"\nTotal training time: {grpo_time/60:.1f} min")