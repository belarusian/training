# Training Directory AGENTS.md

## Current State (May 28, 2026)

### ✅ Completed

1. **Code Extraction** - Extracted 428 Effect-style TypeScript code samples from:
   - effect-smol: 185 samples
   - effect: 208 samples
   - opencode: 28 samples
   - effect-examples: 7 samples
   - Output: `data/effect-code-samples.json`

2. **Repository Structure** - Git repo initialized at `C:\Users\kodep\Training\`:
   - `examples/` - 4 example notebooks (learning resources)
   - `training/` - Our training work
   - `data/` - Extracted training data
   - Subtrees: effect-smol, effect, effect-examples, opencode

3. **GPU Setup Verified**:
   - CUDA 13.0 available
   - NVIDIA GeForce RTX 4090 (24GB VRAM)
   - PyTorch 2.10.0+cu130

4. **Unsloth Training Environment** - Working with CUDA support:
   - venv at `unsloth_env/` with Python 3.13
   - Unsloth 2026.5.8 with 2x faster fine-tuning
   - PyTorch 2.10.0+cu130 (CUDA-enabled)
   - vLLM NOT installed (conflicts with unsloth on Windows; training works without it)

### ✅ Files Created

```
Training/
├── examples/                    # 4 example notebooks we studied
│   ├── effect-smol/ (subtree)
│   ├── effect/ (subtree)
│   ├── effect-examples/ (subtree)
│   ├── opencode/ (subtree)
│   ├── Qwen3_5_MoE.ipynb
│   ├── Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb
│   ├── Qwen3_(4B)_GRPO.ipynb
│   └── Ministral_3_(3B)_Reinforcement_Learning_Sudoku_Game.ipynb
├── training/                    # Our actual training
│   ├── main-training.ipynb      # Notebook version (for reference)
│   ├── backup/
│   │   └── main-training-original.ipynb  # Original Unsloth notebook
│   ├── run-training.py          # Python script version (run this)
│   └── output/                  # Trained model output
├── unsloth_env/                 # Python venv with CUDA support
├── data/                        # Training data
│   └── effect-code-samples.json # 428 samples
├── AGENTS.md                    # This file
├── README.md                    # Project overview
└── .gitignore                   # Git ignore rules
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

**Option 1: Run the Python script directly**
```powershell
.\unsloth_env\Scripts\python.exe training\run-training.py
```

**Option 2: Activate venv first, then run**
```powershell
.\unsloth_env\Scripts\Activate.ps1
python training\run-training.py
```

The script includes pre-flight checks for:
- CUDA availability (exits with error if not found)
- Required packages (unsloth, datasets, trl)
- Training data file existence

### Training Workflow

1. **Load model** -> Qwen3-4B with LoRA config (4-bit quantized)
2. **Format data** -> Convert extracted samples to chat messages
3. **GRPO training** -> Reinforcement learning with reward functions
   - +1.0 for `<CODE>` tags
   - +0.5 for Effect imports
   - +0.3 for Schema usage
   - +0.2 for exports
   - -0.5 for responses too short
4. **Save** -> Export LoRA adapter to `training/output/`

### vLLM Note

vLLM was tried but causes version conflicts on Windows:
- vLLM requires transformers >= 4.56.0
- Unsloth requires transformers <= 5.5.0

Training works fine without vLLM. vLLM is only needed for **faster** inference during GRPO, not for correctness.

### Git Setup

**Remote**: `https://github.com/belarusian/training.git`

### Notes for Next Session

**Ready to train!** Just run:
```powershell
.\unsloth_env\Scripts\python.exe training\run-training.py
```
