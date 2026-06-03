# Qwen3-4B Effect TypeScript Fine-tuning

Fine-tune Qwen3-4B to generate high-quality Effect-style TypeScript code using a two-stage pipeline: SFT (Supervised Fine-Tuning) + GRPO (Group Relative Policy Optimization).

## Marcus Aurelius Philosopher Model

**Note**: Marcus Aurelius training is also available in the [Practice repo](https://github.com/belarusian/training) (`src/industry_ml_lab/training/philosopher*`).

| Repo | Model | Approach | Use Case |
|------|-------|----------|----------|
| **Training** | Qwen3-4B | SFT + GRPO with Unsloth LoRA | Production ML pipeline, advanced RLHF |
| **Practice** | Qwen3-100M | Full fine-tuning with PyTorch | Experimentation, learning, tiny LLM |

**Why both?**
- **Training repo**: Effect TypeScript fine-tuning (primary focus)
- **Practice repo**: Marcus Aurelius as a learning experiment (separate use case)

See `training/marcus/README.md` for Marcus training details in this repo.

## Trained Model

**Download the trained LoRA adapter:**
https://huggingface.co/Kodep/qwen3-4b-effect-codegen-v2

### Model Overview

| Feature | Value |
|---------|-------|
| Base Model | Qwen3-4B (Qwen3ForCausalLM) |
| Number of Layers | 36 |
| Hidden Size | 2560 |
| Attention Heads | 32 (Q), 8 (KV) |
| Vocabulary Size | 151936 |
| Max Position Embeddings | 40960 |
| LoRA Rank | 64 |
| Trainable Parameters | 132M (3.18% of base model) |
| Precision | float16 |
| License | Apache 2.0 |

### Model Performance

| Metric | Value |
|--------|-------|
| **Final Reward** | 0.9775 ± 0.2134 |
| **Reward Improvement** | +0.2012 (+26%) |
| **KL Divergence** | 0.19 |
| **Gradient Norm** | 0.282 |
| **Training Time** | ~63 minutes total |

**Full post-training summary:** See `POST_TRAINING_SUMMARY.md`

### Quick Start (Pre-trained)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Kodep/qwen3-4b-effect-codegen-v2",
    max_seq_length=4096,
    load_in_4bit=True,
)
```

## Quick Start (Training)

```powershell
# Stage 1: SFT (Supervised Fine-Tuning)
.\unsloth_env\Scripts\python.exe training\run-sft-training-1.py

# Stage 2: GRPO (Reinforcement Fine-Tuning)
.\unsloth_env\Scripts\python.exe training\run-grpo-training-2.py
```

See `training/README.md` for detailed training playbook and testing instructions.

## Training Pipeline (Two-Stage)

### Stage 1: SFT (Supervised Fine-Tuning)
Fast pre-training on exact examples to teach the model Effect code patterns.
- **Duration**: ~7.5 minutes on RTX 4090
- **Final loss**: ~0.834
- **Output**: `training/output-v2/qwen3-4b-effect-codegen-sft/`

### Stage 2: GRPO (Reinforcement Fine-Tuning)
Reinforcement learning with reward functions to optimize code quality.
- **Duration**: ~66.6 minutes on RTX 4090
- **Final reward**: ~0.698
- **Output**: `training/output-v2/qwen3-4b-effect-codegen-v2/`

### Reward Functions
- +1.0 for `<CODE>` tags
- +0.5 for Effect imports (`from effect`, `import Effect`)
- +0.3 for Schema usage
- +0.2 for exports
- +0.3 to +0.7 based on output length (shorter = lower, up to 2000 chars)
- -0.5 for responses too short (< 100 chars)

## Command Line Options

```powershell
# Stage 1: SFT only
.\unsloth_env\Scripts\python.exe training\run-sft-training-1.py

# Stage 2: GRPO (requires SFT first)
.\unsloth_env\Scripts\python.exe training\run-grpo-training-2.py

# Test trained model
.\unsloth_env\Scripts\python.exe training\test-training.py --stage sft   # After SFT
.\unsloth_env\Scripts\python.exe training\test-training.py --stage grpo  # After GRPO

# Convert to GGUF for llama.cpp inference
.\unsloth_env\Scripts\python.exe training\convert-to-gguf.py

# Standalone evaluation (uses llama.cpp HTTP server)
.\unsloth_env\Scripts\python.exe training\evaluate-model.py --model training\output-v2\qwen3-4b-effect-codegen-v2

# Compare models (local vs remote)
.\unsloth_env\Scripts\python.exe training\compare-models.py
```

See `training/README.md` for detailed testing instructions.

**Note**: Training uses 16-bit float (no quantization) to enable easy GGUF conversion. After training, convert to GGUF format for fast llama.cpp inference.

## Standalone Scripts

### Evaluate Model
```powershell
.\unsloth_env\Scripts\python.exe training\evaluate-model.py --model training\output-v2\qwen3-4b-effect-codegen-v2
```

### Compare Models (Local vs Remote)
```powershell
.\unsloth_env\Scripts\python.exe training\compare-models.py
```

## Optimizations

This project runs on consumer hardware (RTX 4090, 24GB VRAM) thanks to Unsloth's GPU optimizations:

- **8-bit Quantization** - Uses `load_in_8bit=True` with `load_in_4bit=False` for slightly better quality than 4-bit. Both flags must be set explicitly when loading 8-bit models. Only 0.02% of parameters (the LoRA adapters) are trainable.
- **Unsloth Gradient Checkpointing** (`use_gradient_checkpointing = "unsloth"`) - A custom checkpointing algorithm that asynchronously offloads activations to system RAM during training, reducing VRAM usage by ~30% with only ~2% compute overhead.
- **Custom Triton Kernels** - Unsloth replaces standard PyTorch operations with hand-written GPU kernels in Triton. Fused RoPE and MLP kernels deliver 2–5x faster training.
- **Smart Packing** - Combines multiple short training samples into single tensors, eliminating wasted compute on padding tokens.

## Unsloth GRPO Configuration

Unsloth dynamically patches `GRPOConfig` at runtime. Key settings:
- `per_device_train_batch_size=4` — Must match `num_generations` to avoid config patching conflicts
- `unsloth_num_chunks=2` — Added by Unsloth for chunked processing
- `unsloth_grpo_mini_batch=4` — Added by Unsloth for mini-batch training
- Always use `from trl import GRPOConfig, GRPOTrainer` after importing Unsloth

## Unsloth Import Order

**Critical**: `import unsloth` must be at the top of the file, **before** `trl`, `transformers`, `peft`. Importing Unsloth after TRL causes warnings and potential config patching issues.

## Model Performance Metrics

### Training Metrics (16-bit float, RTX 4090)

| Metric | SFT Stage | GRPO Stage |
|--------|-----------|------------|
| Duration | ~7.5 min | ~66.6 min |
| Final loss | 0.834 | — |
| Avg reward | — | 0.698 |
| Reward std | — | 0.056 |
| KL divergence | — | 0.0005 |
| Grad norm | — | ~0.00002 |
| frac_reward_zero | — | 0.375 |

### Post-Training Evaluation

Run evaluation to get per-sample metrics:
```powershell
.\unsloth_env\Scripts\python.exe training\run-training-v2.py --eval-only
.\unsloth_env\Scripts\python.exe training\evaluate-model.py --model training\output-v2\qwen3-4b-effect-codegen-v2
```

Output is saved to `training/output-v2/eval-metrics.json` with:
- **Per-sample scores**: code_tags, effect_imports, schema_usage, exports, length
- **Component percentages**: what % of responses use each pattern
- **Generation time**: average time per sample
- **Success/fail status**: with timeout protection (120s per sample)

### Model Comparison

Compare against remote models (OpenAI-compatible endpoint):
```powershell
.\unsloth_env\Scripts\python.exe training\compare-models.py
```
Output: `training/output/comparison-results.json`

### Key Learnings

- **SFT is critical for GRPO**: Without SFT pre-training, `frac_reward_zero_std` stays at 1.0 (no reward variance = no learning)
- **GRPO grad_norm was very small** (~0.00002) — model barely updates during GRPO; consider adjusting learning rate
- **8-bit vs 4-bit**: 8-bit loads with `load_in_4bit=False` + `load_in_8bit=True` (both flags required)
- **Post-training eval**: Model should not be loaded twice — causes memory issues

## Project Structure

```
├── training/                            # Training work
│   ├── run-sft-training-1.py            # Stage 1: SFT training
│   ├── run-grpo-training-2.py           # Stage 2: GRPO training (requires SFT)
│   ├── test-training.py                 # Test script (--stage sft | grpo)
│   ├── training/README.md               # Detailed training playbook
│   ├── evaluate-model.py                # Standalone evaluation script
│   ├── compare-models.py                # Local vs remote model comparison
│   ├── main-training.ipynb              # Notebook version (reference)
│   ├── backup/                          # Original Unsloth notebook
│   └── output-v2/                       # Training outputs (gitignored)
│       ├── qwen3-4b-effect-codegen-sft/ # SFT model (~500MB)
│       └── qwen3-4b-effect-codegen-v2/  # GRPO model (~500MB)
├── extracted-code/                      # Training data (428 samples, 18MB)
│   └── effect-code-samples.json
├── examples/                            # Learning notebooks
├── literature/                          # RLHF/GRPO reference material
├── extract-code.ts                      # Data extraction script
├── LICENSE
├── README.md
└── AGENTS.md
```

## Data Extraction

Extract training samples from Effect repositories:

```bash
bun run extract-code.ts
# or with custom directories:
TRAINING_DIR=/path/to/repos OUTPUT_DIR=/path/to/output bun run extract-code.ts
```

## Requirements

- NVIDIA GPU with CUDA support (tested on RTX 4090, 24GB VRAM)
- Python 3.13
- CUDA 13.0
- Unsloth 2026.5.8
- PyTorch 2.10.0+cu130

## Python Environment

**The system Python (`python`) has a CPU-only PyTorch build and cannot train.**

You MUST use the `unsloth_env` venv.

## Post-Training: Convert to GGUF

To enable fast llama.cpp inference, convert your trained model to GGUF format:

```powershell
.\unsloth_env\Scripts\python.exe training\convert-to-gguf.py
```

This creates:
- `Qwen3-4B.F16.gguf` (~8 GB) - Float16 format
- `Qwen3-4B.Q8_0.gguf` (~4.3 GB) - Q8_0 quantized format (50% size reduction)

Use the Q8_0 file with llama.cpp for faster inference with minimal quality loss.

## Running Inference with llama.cpp

Start the llama.cpp server:

```powershell
.\run-inference.ps1
```

Then run evaluation:

```powershell
.\unsloth_env\Scripts\python.exe training\run-training-v2.py --eval-only
```

## Notes

- **vLLM** is not required for training (excluded due to Windows compatibility; vLLM requires transformers >= 4.56.0, Unsloth requires transformers <= 5.5.0)
- **Training uses 16-bit float** (not quantized) to enable easy GGUF conversion for llama.cpp
- **Model weights** (~500MB models) are hosted on Hugging Face, not in this repo
- **Total training time**: ~75 minutes (7.5 min SFT + 66.6 min GRPO) on RTX 4090
- **wandb** logging is optional; metrics are logged to `effect-codegen-grpo` project
- **LSP errors** can be ignored when code runs correctly (static analyzer vs runtime patches)

## License

MIT License - see [LICENSE](LICENSE)
