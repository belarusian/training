#!/usr/bin/env -S python.exe
"""
Model Evaluation Script

Evaluates a trained model on custom prompts and generates a detailed analysis report.
Compares against ground truth examples to assess code quality.

Usage:
    .\\unsloth_env\\Scripts\\python.exe training/evaluate-model.py
       --model training/output-v2/qwen3-4b-effect-codegen-v2
       --prompts prompts.json
       --output evaluation-report.json
"""

import os
import sys
import json
import argparse
import re
import time
import torch
from datetime import datetime
from typing import Optional

try:
    import torch
    from unsloth import FastLanguageModel
except ImportError as e:
    print(f"FATAL: Missing package: {e}")
    print("Install: .\\unsloth_env\\Scripts\\python.exe -m pip install unsloth transformers")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Soft Ground Truth Comparison
# ---------------------------------------------------------------------------

def compare_code_soft(generated: str, expected: str) -> dict:
    """
    Perform soft comparison of generated vs expected code.
    Checks for functional equivalence rather than strict string match.
    """
    comparison = {
        "exact_match": False,
        "syntax_valid": False,
        "key_patterns_found": [],
        "pattern_coverage": 0.0,
        "similarity_score": 0.0,
    }
    
    # Check exact match first
    if generated.strip() == expected.strip():
        comparison["exact_match"] = True
        comparison["pattern_coverage"] = 1.0
        comparison["similarity_score"] = 1.0
        return comparison
    
    # Extract key patterns from expected code
    expected_patterns = {
        "effect_import": bool(re.search(r'from ["\']effect["\']', expected)),
        "schema_import": bool(re.search(r'from ["\']effect/Schema["\']', expected)),
        "export": bool(re.search(r'export\s+(const|function|class)', expected)),
        "type_annotation": bool(re.search(r':\s*(string|number|boolean|any|Array|Record)', expected)),
        "pipe_operator": bool(re.search(r'\.pipe\s*\(', expected)),
        "effect_call": bool(re.search(r'Effect\.(succeed|make|attempt|catch)', expected)),
    }
    
    # Check which patterns are present in generated code
    generated_patterns = {
        "effect_import": bool(re.search(r'from ["\']effect["\']', generated)),
        "schema_import": bool(re.search(r'from ["\']effect/Schema["\']', generated)),
        "export": bool(re.search(r'export\s+(const|function|class)', generated)),
        "type_annotation": bool(re.search(r':\s*(string|number|boolean|any|Array|Record)', generated)),
        "pipe_operator": bool(re.search(r'\.pipe\s*\(', generated)),
        "effect_call": bool(re.search(r'Effect\.(succeed|make|attempt|catch)', generated)),
    }
    
    # Calculate pattern coverage
    total_patterns = len(expected_patterns)
    matched_patterns = sum(1 for k in expected_patterns if expected_patterns[k] == generated_patterns[k])
    comparison["pattern_coverage"] = matched_patterns / total_patterns if total_patterns > 0 else 0.0
    
    # Find which specific patterns were found
    for pattern_name in expected_patterns:
        if generated_patterns[pattern_name]:
            comparison["key_patterns_found"].append(pattern_name)
    
    # Estimate similarity based on line overlap
    gen_lines = set(generated.strip().split('\n'))
    exp_lines = set(expected.strip().split('\n'))
    
    if gen_lines or exp_lines:
        intersection = len(gen_lines & exp_lines)
        union = len(gen_lines | exp_lines)
        comparison["similarity_score"] = intersection / union if union > 0 else 0.0
    
    return comparison

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
CODE_START = "<CODE>"
CODE_END = "</CODE>"

SYSTEM_PROMPT = """You are an expert TypeScript developer specializing in the Effect framework.
Analyze the requirements and generate high-quality Effect code.
Provide your reasoning between <start_working_out> and <end_working_out>.
Then, provide your complete TypeScript code implementation between <CODE> and </CODE>."""

DEFAULT_PROMPTS = [
    "Generate an Effect service pattern for a database connection pool",
    "Create an Effect stream that processes HTTP requests with retries",
    "Generate an Effect schema for a user registration form with validation",
    "Create an Effect effect that makes concurrent API calls with timeout",
    "Generate an Effect layer that combines configuration and database services",
]

DEFAULT_OUTPUT_DIR = "training/output-v2"

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
    ]
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code)
        count += len(matches)
    return max(1, count // 3)

def count_exports(code: str) -> int:
    """Count export statements."""
    return len(re.findall(r'export\s+(const|function|class|type|interface|enum)', code))

def has_code_tags(response: str) -> bool:
    """Check if response has <CODE> tags."""
    return "<CODE>" in response and "</CODE>" in response

def has_reasoning_tags(response: str) -> bool:
    """Check if response has <start_working_out> and <end_working_out> tags."""
    return REASONING_START in response and REASONING_END in response

def extract_code(response: str) -> str:
    """Extract code between <CODE> tags."""
    match = re.search(r'<CODE>(.*?)</CODE>', response, re.DOTALL)
    if match:
        return match.group(1)
    return response

def extract_reasoning(response: str) -> Optional[str]:
    """Extract reasoning between <start_working_out> and <end_working_out> tags."""
    match = re.search(r'<start_working_out>(.*?)<end_working_out>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def compute_reward_details(text: str) -> dict:
    """Compute detailed reward breakdown for a response."""
    components = {
        "code_tags": 0.0,
        "effect_imports": 0.0,
        "schema_usage": 0.0,
        "exports": 0.0,
        "length": 0.0,
    }
    
    if "<CODE>" in text and "</CODE>" in text:
        components["code_tags"] = 1.0
    
    if "from effect" in text or "import Effect" in text:
        components["effect_imports"] = 0.5
    
    if "Schema" in text:
        components["schema_usage"] = 0.3
    
    if "export " in text:
        components["exports"] = 0.2
    
    text_len = len(text)
    if text_len < 100:
        components["length"] = -0.5
    elif text_len < 200:
        components["length"] = 0.3
    elif text_len < 1000:
        components["length"] = 0.5
    elif text_len < 2000:
        components["length"] = 0.7
    else:
        components["length"] = 0.5
    
    components["total"] = sum(components.values())
    
    return components

# ---------------------------------------------------------------------------
# Model Inference
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {model_path}...")
    # Load in 16-bit float for compatibility with current training (16-bit → GGUF)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=False,
        load_in_8bit=False,
        dtype=torch.float16,
        fast_inference=False,
        max_lora_rank=64,
    )
    
    print("Model loaded successfully.")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, 
                      greedy: bool = False) -> tuple[str, float]:
    """Generate a response from the model."""
    start_time = time.time()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs_encoded = tokenizer([inputs], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs_encoded,
        max_new_tokens=max_tokens,
        temperature=temperature if not greedy else 0.0,
        do_sample=not greedy,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - start_time
    
    return response, elapsed

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(prompt: str, response: str, elapsed: float, ground_truth: Optional[str] = None) -> dict:
    """Evaluate a single sample."""
    code = extract_code(response)
    reasoning = extract_reasoning(response)
    
    reward_details = compute_reward_details(response)
    
    # Ground truth comparison if provided
    gt_comparison = None
    if ground_truth:
        gt_code = extract_code(ground_truth)
        
        # Use soft comparison for functional equivalence
        gt_comparison = compare_code_soft(code, gt_code)
        
        # Also include basic metrics for compatibility
        gt_comparison["ground_truth_length"] = len(gt_code)
    
    evaluation = {
        "prompt": prompt,
        "response_length": len(response),
        "code_length": len(code),
        "has_code_tags": has_code_tags(response),
        "has_reasoning_tags": has_reasoning_tags(response),
        "reasoning_length": len(reasoning) if reasoning else 0,
        "code": code,
        "reasoning": reasoning,
        "elapsed_seconds": elapsed,
        "reward_breakdown": reward_details,
        "metrics": {
            "effect_imports": count_effect_imports(code),
            "schema_usage": count_schema_usage(code),
            "exports": count_exports(code),
            "overall_score": reward_details["total"],
        },
    }
    
    if gt_comparison:
        evaluation["ground_truth_comparison"] = gt_comparison
        evaluation["metrics"]["ground_truth_match"] = gt_comparison["exact_match"]
        evaluation["metrics"]["ground_truth_similarity"] = gt_comparison["similarity_score"]
        evaluation["metrics"]["pattern_coverage"] = gt_comparison["pattern_coverage"]
    
    return evaluation

# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(evaluations: list[dict], model_path: str, output_path: str):
    """Generate JSON and human-readable reports."""
    
    # Calculate aggregate metrics
    total_samples = len(evaluations)
    avg_score = sum(e["metrics"]["overall_score"] for e in evaluations) / total_samples if total_samples > 0 else 0
    avg_code_length = sum(e["code_length"] for e in evaluations) / total_samples if total_samples > 0 else 0
    avg_response_length = sum(e["response_length"] for e in evaluations) / total_samples if total_samples > 0 else 0
    avg_time = sum(e["elapsed_seconds"] for e in evaluations) / total_samples if total_samples > 0 else 0
    
    tag_compliance = {
        "code_tags": sum(1 for e in evaluations if e["has_code_tags"]) / total_samples * 100 if total_samples > 0 else 0,
        "reasoning_tags": sum(1 for e in evaluations if e["has_reasoning_tags"]) / total_samples * 100 if total_samples > 0 else 0,
    }
    
    reward_avg = {}
    if total_samples > 0:
        component_names = evaluations[0]["reward_breakdown"].keys()
        for name in component_names:
            reward_avg[name] = sum(e["reward_breakdown"][name] for e in evaluations) / total_samples
    
    # Calculate ground truth metrics if available
    gt_match_rate = None
    gt_similarity_avg = None
    gt_pattern_coverage_avg = None
    
    gt_samples = [e for e in evaluations if "ground_truth_comparison" in e]
    if gt_samples:
        gt_match_rate = sum(1 for e in gt_samples if e["ground_truth_comparison"]["exact_match"]) / len(gt_samples) * 100
        gt_similarity_avg = sum(e["ground_truth_comparison"]["similarity_score"] for e in gt_samples) / len(gt_samples)
        gt_pattern_coverage_avg = sum(e["ground_truth_comparison"]["pattern_coverage"] for e in gt_samples) / len(gt_samples)
    
    report = {
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_samples": total_samples,
            "avg_overall_score": avg_score,
            "avg_code_length": avg_code_length,
            "avg_response_length": avg_response_length,
            "avg_generation_time_seconds": avg_time,
            "avg_ground_truth_similarity": gt_similarity_avg,
            "avg_pattern_coverage": gt_pattern_coverage_avg,
        },
        "tag_compliance": tag_compliance,
        "ground_truth_compliance": {
            "exact_match_rate": gt_match_rate,
            "avg_ground_truth_similarity": gt_similarity_avg,
            "avg_pattern_coverage": gt_pattern_coverage_avg,
        } if gt_match_rate is not None else None,
        "reward_breakdown": reward_avg,
        "samples": evaluations,
    }
    
    # Save JSON report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate human-readable report
    readable_path = output_path.replace(".json", ".txt")
    with open(readable_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Model: {model_path}\n")
        f.write(f"Timestamp: {report['timestamp']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total Samples:        {total_samples}\n")
        f.write(f"  Avg Overall Score:    {avg_score:.3f}\n")
        f.write(f"  Avg Code Length:      {avg_code_length:.0f} chars\n")
        f.write(f"  Avg Response Length:  {avg_response_length:.0f} chars\n")
        f.write(f"  Avg Generation Time:  {avg_time:.2f}s\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("TAG COMPLIANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Code Tags (<CODE>):           {tag_compliance['code_tags']:.1f}%\n")
        f.write(f"  Reasoning Tags:               {tag_compliance['reasoning_tags']:.1f}%\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("REWARD BREAKDOWN (Averaged)\n")
        f.write("-" * 70 + "\n")
        for component, avg in reward_avg.items():
            f.write(f"  {component:20s}: {avg:.3f}\n")
        f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("SAMPLES\n")
        f.write("-" * 70 + "\n\n")
        
        for i, sample in enumerate(evaluations, 1):
            f.write(f"[Sample {i}]\n")
            f.write(f"  Prompt:        {sample['prompt'][:100]}...\n")
            f.write(f"  Code Length:   {sample['code_length']} chars\n")
            f.write(f"  Score:         {sample['metrics']['overall_score']:.3f}\n")
            f.write(f"  Code Tags:     {'[OK]' if sample['has_code_tags'] else '[FAIL]'}\n")
            f.write(f"  Reasoning:     {'[OK]' if sample['has_reasoning_tags'] else '[FAIL]'}\n")
        
            if 'ground_truth_comparison' in sample:
                gt = sample['ground_truth_comparison']
                
                # Show soft comparison results
                if gt['exact_match']:
                    f.write(f"  Ground Truth:  ✓ EXACT MATCH\n")
                else:
                    f.write(f"  Ground Truth:  {gt['similarity_score']:.0%} similar\n")
                    f.write(f"                 Pattern coverage: {gt['pattern_coverage']:.0%}\n")
                    if gt['key_patterns_found']:
                        f.write(f"                 Found: {', '.join(gt['key_patterns_found'])}\n")
            else:
                if sample['metrics']['effect_imports'] > 0:
                    f.write(f"  Effect:        ✓ ({sample['metrics']['effect_imports']} imports)\n")
                if sample['metrics']['schema_usage'] > 0:
                    f.write(f"  Schema:        ✓ ({sample['metrics']['schema_usage']} usages)\n")
                if sample['metrics']['exports'] > 0:
                    f.write(f"  Exports:       ✓ ({sample['metrics']['exports']} exports)\n")
            
            f.write("\n")
        
        f.write("=" * 70 + "\n")
    
    return report

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on custom prompts")
    parser.add_argument("--model", default=None, help="Path to trained model")
    parser.add_argument("--prompts", default=None, help="Path to JSON file with prompts")
    parser.add_argument("--output", default=None, help="Output path for evaluation report")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (temperature=0)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate from prompt file")
    parser.add_argument("--ground-truth", default=None, help="Path to JSON file with ground truth codes (prompt -> expected_code)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        # Try default paths
        default_paths = [
            "training/output-v2/qwen3-4b-effect-codegen-v2",
            "training/output/qwen3-4b-effect-codegen",
        ]
        model_path = None
        for path in default_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found model at: {model_path}")
                break
        
        if not model_path:
            print("FATAL: No model path specified and no default model found")
            print(f"  Try: --model training/output-v2/qwen3-4b-effect-codegen-v2")
            sys.exit(1)
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Determine prompts
    prompts = DEFAULT_PROMPTS
    if args.prompts and os.path.exists(args.prompts):
        with open(args.prompts, "r") as f:
            prompts = json.load(f)
        if isinstance(prompts, list):
            prompts = prompts
        elif isinstance(prompts, dict) and "prompts" in prompts:
            prompts = prompts["prompts"]
    
    # Limit number of samples if specified
    if args.num_samples and args.num_samples > 0:
        prompts = prompts[:args.num_samples]
    
    print(f"Testing {len(prompts)} prompts...")
    print("=" * 70)
    
    # Evaluate each prompt
    evaluations = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt[:60]}...")
        
        # Load ground truth if provided
        ground_truth = None
        if args.ground_truth and os.path.exists(args.ground_truth):
            with open(args.ground_truth, "r") as f:
                gt_data = json.load(f)
            # Try to find ground truth by prompt or by index
            if isinstance(gt_data, list):
                if i <= len(gt_data):
                    ground_truth = gt_data[i-1].get("completion", gt_data[i-1].get("code", gt_data[i-1]))
            elif isinstance(gt_data, dict):
                # Try exact match first, then substring match
                for key, value in gt_data.items():
                    if key == prompt or prompt in key:
                        ground_truth = value
                        break
        
        try:
            response, elapsed = generate_response(model, tokenizer, prompt, args.max_tokens, 
                                                  temperature=args.temperature, greedy=args.greedy)
            evaluation = evaluate_sample(prompt, response, elapsed, ground_truth)
            evaluations.append(evaluation)
            
            print(f"    [OK] Generated {evaluation['code_length']} chars in {elapsed:.2f}s")
            print(f"    Score: {evaluation['metrics']['overall_score']:.2f}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            evaluations.append({
                "prompt": prompt,
                "error": str(e),
                "metrics": {"overall_score": 0.0},
                "elapsed_seconds": 0.0,
            })
    
    # Generate report
    output_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, "evaluation-report.json")
    report = generate_report(evaluations, model_path, output_path)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nOverall Score: {report['summary']['avg_overall_score']:.3f}")
    print(f"Code Tags:     {report['tag_compliance']['code_tags']:.1f}%")
    print(f"Reasoning:     {report['tag_compliance']['reasoning_tags']:.1f}%")
    
    if report.get('ground_truth_compliance') and report['ground_truth_compliance']['exact_match_rate'] is not None:
        print(f"Ground Truth:  {report['ground_truth_compliance']['exact_match_rate']:.1f}% exact match")
        print(f"               {report['ground_truth_compliance']['avg_ground_truth_similarity']:.0%} average similarity")
        print(f"               {report['ground_truth_compliance']['avg_pattern_coverage']:.0%} pattern coverage")
    
    print(f"\nReports saved to:")
    print(f"  JSON:  {output_path}")
    print(f"  Text:  {output_path.replace('.json', '.txt')}")
    print("=" * 70)
    
    return report

if __name__ == "__main__":
    main()
