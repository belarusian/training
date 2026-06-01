"""
Training Qwen3-4B for Effect TypeScript Code Generation - Stage 1
Supervised Fine-Tuning (SFT) on exact code examples.

This is Stage 1 of 2 in the two-stage training pipeline.
Run this FIRST, then run run-grpo-training-2.py.

Output: LoRA adapter saved to training/output-v2/qwen3-4b-effect-codegen-sft/
"""

import os
import sys
import json
import time
import torch
from datetime import datetime

# ---------------------------------------------------------------------------
# Pre-flight checks: CUDA, packages, data
# ---------------------------------------------------------------------------

print("=" * 70)
print("Pre-flight checks")
print("=" * 70)

if not torch.cuda.is_available():
    print("FATAL: CUDA is not available. Training requires a GPU with CUDA support.")
    print(f"   PyTorch version: {torch.__version__}")
    sys.exit(1)

print(f"  CUDA available:       Yes")
print(f"  CUDA version:         {torch.version.cuda}")
print(f"  GPU:                  {torch.cuda.get_device_name(0)}")
print(f"  VRAM:                 {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

required_packages = {
    "unsloth": "unsloth",
    "datasets": "datasets",
    "trl": "trl",
}

WAND_AVAILABLE = False
try:
    import wandb
    WAND_AVAILABLE = True
    print(f"  wandb:                Available (logging enabled)")
except ImportError:
    print(f"  wandb:                Not installed (optional)")

missing = []
for pkg_name, import_name in required_packages.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {pkg_name}:           {version} OK")
    except ImportError:
        print(f"  {pkg_name}:           MISSING")
        missing.append(pkg_name)

if missing:
    print(f"\nFATAL: Missing packages: {', '.join(missing)}")
    sys.exit(1)

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "extracted-code", "effect-code-samples.json")

if not os.path.exists(DATA_PATH):
    print(f"\nFATAL: Training data not found at: {DATA_PATH}")
    sys.exit(1)

print(f"  Training data:        {os.path.getsize(DATA_PATH) / 1e6:.1f} MB")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "effect-codegen-sft"
os.environ["WANDB_LOG_MODEL"] = "false"

# ---------------------------------------------------------------------------
# 1. Load model (FULL PRECISION)
# ---------------------------------------------------------------------------

MAX_SEQ_LENGTH = 4096
RANK = 64
MODEL_NAME = "unsloth/Qwen3-4B"

print(f"Loading {MODEL_NAME} in full precision...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
    dtype=torch.float16,
    fast_inference=False,
    max_lora_rank=RANK,
    gpu_memory_utilization=0.85,
)
print("Model loaded.\n")

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
print("LoRA adapters configured.\n")

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

print(f"Formatted dataset: {len(hf_dataset)} samples")
print("\nFirst 500 chars of training text:")
print(hf_dataset["text"][0][:500])
print("...\n")

# ---------------------------------------------------------------------------
# 4. SFT Training
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUPERVISED FINE-TUNING (SFT)")
print("=" * 70)

sft_config = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=1,
    warmup_steps=5,
    logging_steps=5,
    fp16=True,
    bf16=False,
    report_to="wandb" if WAND_AVAILABLE else "none",
    output_dir="training/output-v2/qwen3-4b-effect-codegen-sft",
    save_steps=50,
    save_total_limit=2,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
)

sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
)

print("Starting SFT training...")
sft_start = time.time()
sft_trainer.train()
sft_time = time.time() - sft_start

print(f"\nSFT training complete. Time: {sft_time/60:.1f} min")

OUTPUT_DIR = "training/output-v2/qwen3-4b-effect-codegen-sft"
sft_trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"SFT model saved to {OUTPUT_DIR}")

print("\n" + "=" * 70)
print("STAGE 1 COMPLETE")
print("=" * 70)
print(f"\nNext step: Run run-grpo-training-2.py to train with GRPO")
