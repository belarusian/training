# Training Directory AGENTS.md

## Current State (May 30, 2026)

### ✅ Completed

1. **Code Extraction** - Extracted 428 Effect-style TypeScript code samples from:
   - effect-smol: 185 samples
   - effect: 208 samples
   - opencode: 28 samples
   - effect-examples: 7 samples
   - Output: `extracted-code/effect-code-samples.json` (18MB)

2. **Two-Stage Training Pipeline** (V2):
   - **SFT (Supervised Fine-Tuning)**: ~7.5 min, loss 0.834
     - Output: `training/output-v2/qwen3-4b-effect-codegen-sft/` (~500MB)
   - **GRPO (Reinforcement Fine-Tuning)**: ~66.6 min, reward 0.698, reward_std 0.056
     - Output: `training/output-v2/qwen3-4b-effect-codegen-v2/` (~500MB)
   - Script: `training/run-training-v2.py`
   - V1 (GRPO-only baseline still works): `training/run-training.py`

3. **Evaluation Tools**:
   - `training/evaluate-model.py` - Standalone evaluation with JSON + TXT reports
   - `training/compare-models.py` - Local vs remote model comparison

4. **GPU Setup Verified**:
   - CUDA 13.0 available
   - NVIDIA GeForce RTX 4090 (24GB VRAM)
   - PyTorch 2.10.0+cu130

5. **Unsloth Training Environment** - Working with CUDA support:
   - venv at `unsloth_env/` with Python 3.13
   - Unsloth 2026.5.8 with 2x faster fine-tuning
   - PyTorch 2.10.0+cu130 (CUDA-enabled)
   - vLLM NOT installed (conflicts with unsloth on Windows; training works without it)

### Repository Structure

```
├── training/                            # Training work
│   ├── run-training.py                  # V1: GRPO-only (original, working baseline)
│   ├── run-training-v2.py               # V2: SFT + GRPO two-stage pipeline
│   ├── evaluate-model.py                # Standalone evaluation script
│   ├── compare-models.py                # Local vs remote model comparison
│   ├── main-training.ipynb              # Notebook version (reference)
│   ├── backup/                          # Original Unsloth notebook
│   └── output-v2/                       # V2 training outputs (gitignored)
│       ├── qwen3-4b-effect-codegen-sft/ # SFT adapter (~500MB)
│       └── qwen3-4b-effect-codegen-v2/  # GRPO adapter (~500MB)
├── extracted-code/                      # Training data (428 samples, 18MB)
│   └── effect-code-samples.json
├── examples/                            # Learning notebooks
├── literature/                          # RLHF/GRPO reference material
├── extract-code.ts                      # Data extraction script
├── LICENSE
├── README.md
└── .gitignore
```

### Hardware Specs

- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **CUDA**: 13.0
- **PyTorch**: 2.10.0+cu130
- **Unsloth**: 2026.5.8

### Important: Python Environment

**The system Python (`python`) has a CPU-only PyTorch build and cannot train.**

You MUST use the `unsloth_env` venv which has CUDA-enabled PyTorch:

```powershell
# Activate the venv
.\unsloth_env\Scripts\Activate.ps1

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### How to Run Training

**V2 (Two-Stage: SFT + GRPO):**
```powershell
# Full pipeline (SFT + GRPO)
.\unsloth_env\Scripts\python.exe training\run-training-v2.py

# Skip SFT, load existing SFT adapter, train GRPO only
.\unsloth_env\Scripts\python.exe training\run-training-v2.py --skip-sft

# Run post-training evaluation only (no training)
.\unsloth_env\Scripts\python.exe training\run-training-v2.py --eval-only
```

**V1 (GRPO-only, original):**
```powershell
.\unsloth_env\Scripts\python.exe training\run-training.py
```

The scripts include pre-flight checks for:
- CUDA availability (exits with error if not found)
- Required packages (unsloth, datasets, trl)
- Training data file existence

### Training Workflow (V2)

1. **Load model** -> Qwen3-4B with 8-bit quantization (`load_in_4bit=False`, `load_in_8bit=True`)
2. **Configure LoRA** -> rank 64, gradient checkpointing with Unsloth
3. **Format data** -> Convert extracted samples to chat messages with reasoning tags
4. **Stage 1: SFT** -> Supervised fine-tuning (~7.5 min, loss ~0.834)
5. **Stage 2: GRPO** -> Reinforcement learning with reward functions
   - +1.0 for `<CODE>` tags
   - +0.5 for Effect imports
   - +0.3 for Schema usage
   - +0.2 for exports
   - +0.3 to +0.7 based on length
   - -0.5 for responses < 100 chars
6. **Save** -> Export LoRA adapters to `training/output-v2/`
7. **Push** -> Upload to Hugging Face (model weights are too large for GitHub)

### Data Extraction

Extract training samples from Effect repositories:

```bash
bun run extract-code.ts
# or with custom directories:
TRAINING_DIR=/path/to/repos OUTPUT_DIR=/path/to/output bun run extract-code.ts
```

### vLLM Note

vLLM was tried but causes version conflicts on Windows:
- vLLM requires transformers >= 4.56.0
- Unsloth requires transformers <= 5.5.0

Training works fine without vLLM. vLLM is only needed for **faster** inference during GRPO, not for correctness.

### Hugging Face Model

The trained LoRA adapter is published to Hugging Face:
- **Model**: https://huggingface.co/Kodep/qwen3-4b-effect-codegen

To load it for inference:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Kodep/qwen3-4b-effect-codegen",
    max_seq_length=4096,
    load_in_4bit=True,
)
```

### Git Setup

**Remote**: `https://github.com/belarusian/training.git`

### Key Learnings & Gotchas

1. **Unsloth 8-bit loading requires BOTH flags**: `load_in_4bit=False` AND `load_in_8bit=True`
2. **Unsloth import order**: Must import `unsloth` BEFORE `trl`, `transformers`, `peft`
3. **GRPO batch size**: `per_device_train_batch_size=4` must match `num_generations` to avoid config patching conflicts
4. **Unsloth GRPO patches**: `unsloth_num_chunks=2` and `unsloth_grpo_mini_batch=4` are added dynamically at runtime
5. **SFT is critical for GRPO**: Without SFT pre-training, `frac_reward_zero_std` stays at 1.0 (no reward variance = no learning)
6. **LSP errors**: Can be ignored when code runs correctly (static analyzer vs runtime patches)
7. **wandb metrics**: Logged to `effect-codegen-grpo` project, local cache in `wandb/` (gitignored)
8. **Total training time**: ~75 minutes (7.5 min SFT + 66.6 min GRPO) on RTX 4090
9. **GRPO grad_norm was very small** (~0.00002) - model barely updates during GRPO; consider adjusting learning rate
10. **Post-training eval**: Model should not be loaded twice (once for eval, once inside `run_post_training_eval`) — causes memory issues

### Notes for Next Session

**Ready to train!** Run the full two-stage pipeline:
```powershell
.\unsloth_env\Scripts\python.exe training\run-training-v2.py
```

Or for faster iteration (skip SFT, train GRPO only):
```powershell
.\unsloth_env\Scripts\python.exe training\run-training-v2.py --skip-sft
```
