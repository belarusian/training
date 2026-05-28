# Qwen3-4B Effect TypeScript Fine-tuning

Fine-tune Qwen3-4B to generate high-quality Effect-style TypeScript code using GRPO (Group Relative Policy Optimization).

## Trained Model

**Download the trained LoRA adapter:**
https://huggingface.co/Kodep/qwen3-4b-effect-codegen

### Quick Start

```bash
# Setup environment
uv venv unsloth_env --python 3.13
.\unsloth_env\Scripts\activate
uv pip install unsloth datasets trl --torch-backend=auto

# Run training
.\unsloth_env\Scripts\python.exe training/run-training.py
```

Or load the pre-trained model:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Kodep/qwen3-4b-effect-codegen",
    max_seq_length=4096,
    load_in_4bit=True,
)
```

## Project Structure

```
├── training/run-training.py          # Main training script (run this)
├── training/main-training.ipynb      # Notebook version (reference)
├── training/backup/                  # Original Unsloth notebook
├── extracted-code/                   # Training data (428 samples)
├── examples/                         # Learning notebooks
├── literature/                       # RLHF/GRPO reference material
├── extract-code.ts                   # Data extraction script
├── LICENSE
└── AGENTS.md                         # Agent documentation
```

## Training Pipeline

1. **Load model** - Qwen3-4B with 4-bit quantization
2. **Format data** - Convert extracted samples to chat messages
3. **GRPO training** - Reinforcement learning with reward functions:
   - +1.0 for `<CODE>` tags
   - +0.5 for Effect imports
   - +0.3 for Schema usage
   - +0.2 for exports
   - -0.5 for responses too short
4. **Save** - Export LoRA adapter to `training/output/`

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
- CUDA 12.0+

## Notes

- vLLM is not required for training (excluded due to Windows compatibility)
- Model weights (~500MB LoRA adapter) are hosted on Hugging Face, not in this repo
- Training takes ~45-60 minutes on RTX 4090

## License

MIT License - see [LICENSE](LICENSE)
