"""
Training Qwen3-4B for Effect TypeScript Code Generation

Fine-tune a model to generate high-quality Effect-style TypeScript code using GRPO.
Hardware: NVIDIA GeForce RTX 4090 (24GB VRAM), CUDA 13.0

Usage:
    .\\unsloth_env\\Scripts\\python.exe training/run-training.py
"""

import os
import sys
import json
import torch

# ---------------------------------------------------------------------------
# Pre-flight checks: CUDA, packages, data
# ---------------------------------------------------------------------------
print("=" * 60)
print("Pre-flight checks")
print("=" * 60)

# 1. CUDA check
if not torch.cuda.is_available():
    print("FATAL: CUDA is not available. Training requires a GPU with CUDA support.")
    print(f"   PyTorch version: {torch.__version__}")
    sys.exit(1)

print(f"  CUDA available:       Yes")
print(f"  CUDA version:         {torch.version.cuda}")
print(f"  GPU:                  {torch.cuda.get_device_name(0)}")
print(f"  VRAM:                 {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 2. Package checks
required_packages = {
    "unsloth": "unsloth",
    "datasets": "datasets",
    "trl": "trl",
}

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
    print("Install them in your venv:")
    print(f"  .\\unsloth_env\\Scripts\\python.exe -m pip install {' '.join(missing)}")
    sys.exit(1)

# 3. Data file check
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "extracted-code", "effect-code-samples.json")

if not os.path.exists(DATA_PATH):
    print(f"\nFATAL: Training data not found at: {DATA_PATH}")
    print("Run the data extraction notebook first, or check the path.")
    sys.exit(1)

print(f"  Training data:        {os.path.getsize(DATA_PATH) / 1e6:.1f} MB")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 4096
RANK = 64
MODEL_NAME = "unsloth/Qwen3-4B"

print(f"Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=False,
    max_lora_rank=RANK,
    gpu_memory_utilization=0.7,
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
# 3. Chat template constants
# ---------------------------------------------------------------------------
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
CODE_START = "<CODE>"
CODE_END = "</CODE>"

SYSTEM_PROMPT = """You are an expert TypeScript developer specializing in the Effect framework.
Analyze the requirements and generate high-quality Effect code.
Provide your reasoning between <start_working_out> and <end_working_out>.
Then, provide your complete TypeScript code implementation between <CODE> and </CODE>."""

# Verify template
test_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Generate an Effect service pattern"},
]
formatted = tokenizer.apply_chat_template(test_messages, tokenize=False)
print("=== Template test ===")
print(formatted[:300])
print("...\n")

# ---------------------------------------------------------------------------
# 4. Load training data
# ---------------------------------------------------------------------------
from datasets import Dataset

print(f"Loading data from: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")
print(f"First sample keys: {list(data[0].keys())}\n")

# ---------------------------------------------------------------------------
# 5. Format data into chat messages
# ---------------------------------------------------------------------------
def format_sample(x):
    prompt = x["prompt"]
    completion = x["completion"]
    path = x.get("path", "unknown")
    source = x.get("source", "unknown")

    reasoning = f"Analyzing requirement: {prompt}\nSource: {source}, Path: {path}\nI need to generate TypeScript code using Effect framework patterns."

    final_output = (
        REASONING_START + reasoning + REASONING_END +
        CODE_START + completion + CODE_END
    )

    return {
        "Messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_output},
        ]
    }

hf_dataset = Dataset.from_list([format_sample(s) for s in data])
print(f"Formatted dataset: {len(hf_dataset)} samples\n")

# ---------------------------------------------------------------------------
# 6. Token counting & filtering
# ---------------------------------------------------------------------------
def count_tokens(x):
    return {"token_count": len(tokenizer.apply_chat_template(x["Messages"], tokenize=True))}

hf_dataset = hf_dataset.map(count_tokens, batched=False)

print("Token count statistics:")
print(hf_dataset["token_count"])

hf_dataset = hf_dataset.filter(lambda x: x["token_count"] <= MAX_SEQ_LENGTH)
print(f"After filtering: {len(hf_dataset)} samples fit in {MAX_SEQ_LENGTH} tokens\n")

# ---------------------------------------------------------------------------
# 7. Convert to text (chat template strings)
# ---------------------------------------------------------------------------
def to_text(x):
    return {"text": tokenizer.apply_chat_template(x["Messages"], tokenize=False)}

hf_dataset = hf_dataset.map(to_text, batched=False)

print("First 500 chars of training text:")
print(hf_dataset["text"][0][:500])
print()

# ---------------------------------------------------------------------------
# 8. GRPO dataset preparation
# ---------------------------------------------------------------------------
hf_dataset_grpo = Dataset.from_list([
    {
        "prompt": s["prompt"],
        "completion": s["completion"],
        "path": s.get("path", "unknown"),
        "source": s.get("source", "unknown"),
    }
    for s in data
])

print(f"GRPO dataset: {len(hf_dataset_grpo)} samples")
print("First sample prompt:", hf_dataset_grpo[0]["prompt"])
print("First sample completion:", hf_dataset_grpo[0]["completion"][:200])
print()

# ---------------------------------------------------------------------------
# 9. Reward function
# ---------------------------------------------------------------------------
def reward_fn(prompts, completions, **kwargs):
    rewards = []
    for text in completions:
        score = 0.0

        if "<CODE>" in text and "</CODE>" in text:
            score += 1.0

        if "from effect" in text or "import Effect" in text:
            score += 0.5

        if "Schema" in text:
            score += 0.3

        if "export " in text:
            score += 0.2

        if len(text) < 100:
            score -= 0.5

        rewards.append(score)

    return rewards

test_completion = '<CODE>import * as Effect from "effect"\nexport const test = Effect.succeed(42)</CODE>'
print("Test reward:", reward_fn([""], [test_completion]))
print()

# ---------------------------------------------------------------------------
# 10. Format for GRPO
# ---------------------------------------------------------------------------
def format_for_grpo(x):
    prompt = x["prompt"]
    completion = x["completion"]
    path = x.get("path", "unknown")
    source = x.get("source", "unknown")

    reasoning = f"Analyzing requirement: {prompt}\nSource: {source}, Path: {path}\nI need to generate TypeScript code using Effect framework patterns."

    final_output = (
        REASONING_START + reasoning + REASONING_END +
        CODE_START + completion + CODE_END
    )

    return {
        "Messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_output},
        ]
    }

hf_dataset_grpo = hf_dataset_grpo.map(format_for_grpo)
hf_dataset_grpo = hf_dataset_grpo.map(
    lambda x: {"text": tokenizer.apply_chat_template(x["Messages"], tokenize=False)}
)

print("GRPO dataset formatted.\n")

# ---------------------------------------------------------------------------
# 11. GRPO Training
# ---------------------------------------------------------------------------
from trl import GRPOConfig, GRPOTrainer

print("Starting GRPO training...")
print("=" * 60)

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=GRPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-6,
        num_train_epochs=1,
        warmup_steps=5,
        logging_steps=5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        output_dir="training/output",
    ),
    train_dataset=hf_dataset_grpo,
    processing_class=tokenizer,
)

grpo_trainer.train()

print("=" * 60)
print("Training complete.\n")

# ---------------------------------------------------------------------------
# 12. Save the model
# ---------------------------------------------------------------------------
OUTPUT_DIR = "training/output/qwen3-4b-effect-codegen"

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")
