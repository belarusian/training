# Training Directory AGENTS.md

## Current State (May 26, 2026)

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

4. **Unsloth Studio Installed** - Fully working with CUDA support:
   - 2x faster fine-tuning enabled
   - All patches applied

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
│   └── main-training.ipynb      # Modified notebook for Effect TypeScript
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
- **Unsloth**: Installed via Unsloth Studio

### Next Steps

1. Launch Unsloth Studio: `unsloth studio -H 0.0.0.0 -p 8888`
2. Open `training/main-training.ipynb`
3. Run all cells to train the model
4. Save LoRA adapter to `training/output/`

### Git Setup

**Initial Commit**: "Initial commit: Training project structure"
**Initial Commit**: "Initial commit: Training project structure"
**Total Commits**: 1 (cleaned)

**Remote**: `https://github.com/belarusian/training.git`

### Notes for Next Session

**✅ Everything is ready for training!**

Just launch Unsloth Studio and open the main training notebook.
