# Testing Guide: Qwen3-4B Effect TypeScript Model

## Overview

This guide helps you test the trained `Kodep/qwen3-4b-effect-codegen-v2` model for generating Effect-style TypeScript code.

## Prerequisites

- **Hardware**: NVIDIA GPU with at least 16GB VRAM (RTX 4090 recommended)
- **Python**: 3.13 with CUDA support
- **Environment**: Use `unsloth_env` venv (not system Python)

```powershell
# Activate the correct environment
.\unsloth_env\Scripts\Activate.ps1

# Verify CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Test Methods

### Method 1: Quick Test with Unsloth (Recommended)

```powershell
.\unsloth_env\Scripts\python.exe -c "
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='Kodep/qwen3-4b-effect-codegen-v2',
    max_seq_length=4096,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Test 1: Simple Effect pattern
inputs = tokenizer([
    '<|im_start|>system\nYou are an expert TypeScript developer specializing in the Effect framework.<|im_end|>\n<|im_start|>user\nGenerate an Effect service pattern for a user repository<|im_end|>\n<|im_start|>assistant\n'
], return_tensors='pt').to('cuda')

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0]))
"
```

### Method 2: Full Evaluation Script

```powershell
# Run the evaluation script
.\unsloth_env\Scripts\python.exe training\evaluate-model.py --model training\output-v2\qwen3-4b-effect-codegen-v2
```

This will:
- Generate 10 test cases
- Score each response on: code tags, Effect imports, Schema usage, exports, length
- Save results to `training/output-v2/eval-metrics.json`

### Method 3: Compare with Remote Model

```powershell
# Compare local model vs. remote endpoint
.\unsloth_env\Scripts\python.exe training\compare-models.py
```

## Expected Results

### SFT Stage (Baseline)
- ✅ Generates valid TypeScript syntax
- ✅ Uses Effect imports (`from effect`, `import Effect`)
- ⚠️ May miss some advanced patterns

### GRPO Stage (Fine-tuned)
- ✅ All SFT capabilities
- ✅ Consistent `<CODE>` tags
- ✅ Proper Schema definitions
- ✅ Service patterns (`Layer`, `Context`, `provide`)
- ✅ Length-optimized responses (200-1000 chars)

## Troubleshooting

### Out of Memory
```powershell
# Use 4-bit quantization
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='Kodep/qwen3-4b-effect-codegen-v2',
    load_in_4bit=True,  # Reduces VRAM usage
)
"
```

### Model Not Loading
```powershell
# Check if the model exists on Hugging Face
# Visit: https://huggingface.co/Kodep/qwen3-4b-effect-codegen-v2

# Or use local path
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='training/output-v2/qwen3-4b-effect-codegen-v2',
)
"
"
```

### CUDA Errors
```powershell
# Verify CUDA setup
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"

# If CUDA not available, use CPU (slow!)
python -c "
import torch
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='Kodep/qwen3-4b-effect-codegen-v2',
    device_map='cpu',  # Force CPU
)
"
"
```

## Test Prompts to Try

```typescript
// Prompt 1: Simple Effect
"Generate an Effect service pattern for a user repository"

// Prompt 2: Schema Definition
"Create a Schema for a User entity with name, email, and age fields"

// Prompt 3: Effect Gen
"Write an Effect.gen function that fetches user data and processes it"

// Prompt 4: Service Layer
"Implement a UserService with provide, make, and run methods"
```

## Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Inference Speed | < 2s | Time per response |
| VRAM Usage | < 8GB | With 4-bit quantization |
| Code Quality | > 90% | Score from evaluate-model.py |

## Next Steps

After testing:
1. Review `training/output-v2/eval-metrics.json`
2. Check for patterns in failures
3. Consider GRPO fine-tuning if quality is acceptable but consistency needs improvement

---

**Note**: This model is specifically trained on Effect TypeScript code patterns. It is NOT a general-purpose code generator.
