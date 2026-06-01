#!/usr/bin/env -S python.exe
"""
Model Comparison & Metrics Script

Compares the trained Qwen3-4B Effect model against a remote OpenAI-compatible endpoint.
Generates metrics for code quality, Effect pattern usage, and performance.

Usage:
    .\\unsloth_env\\Scripts\\python.exe training/compare-models.py
"""

import os
import sys
import json
import time
import re
import torch
from typing import Optional

# Check CUDA availability
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import torch
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"FATAL: Missing package: {e}")
    print("Install: .\\unsloth_env\\Scripts\\python.exe -m pip install unsloth transformers")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAINED_MODEL_PATH = "training/output/qwen3-4b-effect-codegen"
REMOTE_BASE_URL = "http://10.106.1.89:8080/v1"
REMOTE_MODEL_NAME = "qwen"

# Test prompts covering common Effect patterns
TEST_PROMPTS = [
    "Generate an Effect service pattern for a database connection pool",
    "Create an Effect stream that processes HTTP requests with retries",
    "Generate an Effect schema for a user registration form with validation",
    "Create an Effect effect that makes concurrent API calls with timeout",
    "Generate an Effect layer that combines configuration and database services",
    "Create an Effect pipe that transforms data through multiple steps",
    "Generate an Effect program that handles errors with fallbacks",
    "Create an Effect service pattern for caching with cache-first strategy",
]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def count_effect_imports(code: str) -> int:
    """Count Effect framework imports in generated code."""
    patterns = [
        r'from ["\']effect["\']',
        r'import \* as Effect from ["\']effect["\']',
        r'from ["\']effect/.*["\']',
    ]
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code)
        count += len(matches)
    return count

def count_schema_usage(code: str) -> int:
    """Count Schema-related code."""
    patterns = [
        r'from ["\']effect/Schema["\']',
        r'import \* as Schema from ["\']effect/Schema["\']',
        r'Schema\.',
        r'Schema\.',
    ]
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code)
        count += len(matches)
    return max(1, count // 3)  # Normalize

def count_exports(code: str) -> int:
    """Count export statements."""
    return len(re.findall(r'export\s+(const|function|class|type|interface|enum)', code))

def has_code_tags(response: str) -> bool:
    """Check if response has <CODE> tags."""
    return "<CODE>" in response and "</CODE>" in response

def extract_code(response: str) -> str:
    """Extract code between <CODE> tags."""
    match = re.search(r'<CODE>(.*?)</CODE>', response, re.DOTALL)
    if match:
        return match.group(1)
    return response

def calculate_length_score(text: str) -> float:
    """Score based on text length (optimal: 200-1000 chars)."""
    length = len(text)
    if length < 100:
        return 0.0
    elif length < 200:
        return 0.5
    elif length <= 1000:
        return 1.0
    elif length <= 2000:
        return 0.8
    else:
        return 0.5

# ---------------------------------------------------------------------------
# Local Model Functions
# ---------------------------------------------------------------------------

def load_local_model():
    """Load the trained LoRA model."""
    print(f"Loading trained model from {TRAINED_MODEL_PATH}...")
    # Load in 16-bit float for compatibility with current training (16-bit → GGUF)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=TRAINED_MODEL_PATH,
        max_seq_length=4096,
        load_in_4bit=False,
        load_in_8bit=False,
        dtype=torch.float16,
        fast_inference=False,
        max_lora_rank=64,
    )
    return model, tokenizer

def generate_local(model, tokenizer, prompt: str, max_tokens: int = 2048) -> tuple[str, float]:
    """Generate response using trained local model."""
    start_time = time.time()
    
    inputs = tokenizer([
        f"""You are an expert TypeScript developer specializing in the Effect framework.
Analyze the requirements and generate high-quality Effect code.

{prompt}

Provide your reasoning between <start_working_out> and <end_working_out>.
Then, provide your complete TypeScript code implementation between <CODE> and </CODE>."""
    ], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - start_time
    
    return response, elapsed

# ---------------------------------------------------------------------------
# Remote Model Functions
# ---------------------------------------------------------------------------

def generate_remote(prompt: str, max_tokens: int = 2048) -> tuple[str, float, bool]:
    """Generate response using remote OpenAI-compatible endpoint."""
    try:
        import httpx
        
        start_time = time.time()
        
        payload = {
            "model": REMOTE_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": """You are an expert TypeScript developer specializing in the Effect framework.
Analyze the requirements and generate high-quality Effect code."""
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nProvide your reasoning and TypeScript code implementation."
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        
        response = httpx.post(
            f"{REMOTE_BASE_URL}/chat/completions",
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        
        result = response.json()
        elapsed = time.time() - start_time
        
        content = result["choices"][0]["message"]["content"]
        return content, elapsed, True
        
    except Exception as e:
        print(f"Error calling remote model: {e}")
        return f"Error: {str(e)}", 0.0, False

# ---------------------------------------------------------------------------
# Metrics Collection
# ---------------------------------------------------------------------------

def calculate_metrics(response: str, code: str) -> dict:
    """Calculate quality metrics for a response."""
    metrics = {
        "has_code_tags": has_code_tags(response),
        "code_length": len(code),
        "effect_imports": count_effect_imports(code),
        "schema_usage": count_schema_usage(code),
        "export_count": count_exports(code),
        "length_score": calculate_length_score(code),
        "is_valid": len(code) > 50,
    }
    
    metrics["overall_score"] = (
        (1.0 if metrics["has_code_tags"] else 0.0) +
        (0.5 if metrics["effect_imports"] > 0 else 0.0) +
        (0.3 if metrics["schema_usage"] > 0 else 0.0) +
        (0.2 if metrics["export_count"] > 0 else 0.0) +
        metrics["length_score"]
    ) / 3.0  # Normalize to ~1.0 max
    
    return metrics

# ---------------------------------------------------------------------------
# Main Comparison
# ---------------------------------------------------------------------------

def run_comparison():
    """Run full model comparison."""
    print("=" * 70)
    print("Model Comparison: Trained vs Remote")
    print("=" * 70)
    print(f"Trained model: {TRAINED_MODEL_PATH}")
    print(f"Remote model: {REMOTE_BASE_URL} ({REMOTE_MODEL_NAME})")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    print("=" * 70)
    print()
    
    # Load local model
    try:
        model, tokenizer = load_local_model()
        local_available = True
    except Exception as e:
        print(f"Warning: Could not load local model: {e}")
        local_available = False
        model, tokenizer = None, None
    
    # Results storage
    results = {
        "local": [],
        "remote": [],
        "summary": {},
    }
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Testing: {prompt[:60]}...")
        print("-" * 60)
        
        # Test local model
        if local_available:
            print("  Local model...")
            try:
                response, elapsed = generate_local(model, tokenizer, prompt)
                code = extract_code(response)
                metrics = calculate_metrics(response, code)
                
                results["local"].append({
                    "prompt": prompt,
                    "response": response[:500] + ("..." if len(response) > 500 else ""),
                    "code": code[:500] + ("..." if len(code) > 500 else ""),
                    "metrics": metrics,
                    "elapsed": elapsed,
                })
                
                print(f"    ✓ Generated {len(code)} chars in {elapsed:.2f}s")
                print(f"    Score: {metrics['overall_score']:.2f}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                results["local"].append({
                    "prompt": prompt,
                    "error": str(e),
                    "metrics": {"overall_score": 0.0},
                    "elapsed": 0.0,
                })
        
        # Test remote model
        print("  Remote model...")
        try:
            response, elapsed, success = generate_remote(prompt)
            
            if success:
                code = extract_code(response)
                metrics = calculate_metrics(response, code)
                
                results["remote"].append({
                    "prompt": prompt,
                    "response": response[:500] + ("..." if len(response) > 500 else ""),
                    "code": code[:500] + ("..." if len(code) > 500 else ""),
                    "metrics": metrics,
                    "elapsed": elapsed,
                })
                
                print(f"    ✓ Generated {len(code)} chars in {elapsed:.2f}s")
                print(f"    Score: {metrics['overall_score']:.2f}")
                
            else:
                print(f"    ✗ Failed: {response[:200]}")
                results["remote"].append({
                    "prompt": prompt,
                    "error": response,
                    "metrics": {"overall_score": 0.0},
                    "elapsed": 0.0,
                })
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results["remote"].append({
                "prompt": prompt,
                "error": str(e),
                "metrics": {"overall_score": 0.0},
                "elapsed": 0.0,
            })
    
    # Calculate summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results["local"]:
        local_scores = [r["metrics"]["overall_score"] for r in results["local"] if "error" not in r]
        local_times = [r["elapsed"] for r in results["local"] if r["elapsed"] > 0]
        
        results["summary"]["local"] = {
            "avg_score": sum(local_scores) / len(local_scores) if local_scores else 0,
            "avg_time": sum(local_times) / len(local_times) if local_times else 0,
            "total_samples": len(local_scores),
        }
        
        print(f"\nTrained Model:")
        print(f"  Average score: {results['summary']['local']['avg_score']:.3f}")
        print(f"  Avg generation time: {results['summary']['local']['avg_time']:.2f}s")
        print(f"  Successful samples: {results['summary']['local']['total_samples']}/{len(TEST_PROMPTS)}")
    
    if results["remote"]:
        remote_scores = [r["metrics"]["overall_score"] for r in results["remote"] if "error" not in r]
        remote_times = [r["elapsed"] for r in results["remote"] if r["elapsed"] > 0]
        
        results["summary"]["remote"] = {
            "avg_score": sum(remote_scores) / len(remote_scores) if remote_scores else 0,
            "avg_time": sum(remote_times) / len(remote_times) if remote_times else 0,
            "total_samples": len(remote_scores),
        }
        
        print(f"\nRemote Model:")
        print(f"  Average score: {results['summary']['remote']['avg_score']:.3f}")
        print(f"  Avg generation time: {results['summary']['remote']['avg_time']:.2f}s")
        print(f"  Successful samples: {results['summary']['remote']['total_samples']}/{len(TEST_PROMPTS)}")
    
    # Direct comparison
    if results["local"] and results["remote"]:
        local_avg = results["summary"]["local"]["avg_score"]
        remote_avg = results["summary"]["remote"]["avg_score"]
        
        print(f"\nComparison:")
        print(f"  Difference: {local_avg - remote_avg:+.3f}")
        
        if local_avg > remote_avg:
            print(f"  → Trained model scores HIGHER")
        elif remote_avg > local_avg:
            print(f"  → Remote model scores HIGHER")
        else:
            print(f"  → Equal performance")
    
    # Save results
    output_path = "training/output/comparison-results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull results saved to: {output_path}")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    run_comparison()
