"""
Post-SFT Verification Script
Tests if the SFT model learned to generate code with proper Effect patterns.

Usage:
    python training/verify-sft.py
"""

import sys
import argparse
import re
from unsloth import FastLanguageModel

def verify_sft_model(model_path):
    """Verify SFT model learned to generate Effect code."""
    
    print("=" * 70)
    print("POST-SFT VERIFICATION")
    print("=" * 70)
    
    # Test prompts that should trigger Effect code generation
    test_prompts = [
        "Generate an Effect service pattern for a database connection pool",
        "Create an Effect stream that processes HTTP requests with retries",
        "Generate an Effect schema for a user registration form",
    ]
    
    # Expected patterns in generated code
    required_patterns = [
        (r'import.*effect|from.*effect', "Effect import"),
        (r'Schema\.', "Schema usage"),
        (r'export', "Export statement"),
    ]
    
    print(f"\nLoading model from: {model_path}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"FATAL: Could not load model: {e}")
        print(f"Make sure SFT training completed and model exists at: {model_path}")
        return False
    
    FastLanguageModel.for_inference(model)
    
    all_passed = True
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i}/{len(test_prompts)}] Testing: {prompt}")
        
        messages = [
            {"role": "system", "content": "You are an expert TypeScript developer specializing in the Effect framework. Provide your reasoning first between <start_working_out> and <end_working_out>, then provide your complete TypeScript code implementation between <CODE> and </CODE>."},
            {"role": "user", "content": prompt},
        ]
        
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True)
            response = tokenizer.decode(outputs[0])
            
            # Extract code between <CODE> tags
            code_match = re.search(r'<CODE>(.*?)</CODE>', response, re.DOTALL)
            code = code_match.group(1) if code_match else ""
            
            print(f"  Response length: {len(response)} chars")
            print(f"  Code length: {len(code)} chars")
            
            # Check required patterns
            for pattern, name in required_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    print(f"  [OK] {name} found")
                else:
                    print(f"  [FAIL] {name} MISSING")
                    all_passed = False
            
            # Check code length
            if len(code) < 100:
                print(f"  [FAIL] Code too short ({len(code)} chars)")
                all_passed = False
            elif len(code) < 500:
                print(f"  [WARN] Code somewhat short ({len(code)} chars)")
            else:
                print(f"  [OK] Code length adequate ({len(code)} chars)")
            
            # Show first 200 chars of code
            if code:
                print(f"  Code preview: {code[:200]}...")
            
        except Exception as e:
            print(f"  Error: {e}")
            all_passed = False
        
        print()
    
    print("=" * 70)
    if all_passed:
        print("[OK] VERIFICATION PASSED - SFT model learned to generate Effect code")
    else:
        print("[FAIL] VERIFICATION FAILED - SFT model needs more training or better data")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify SFT model")
    parser.add_argument("--stage", choices=["sft", "grpo"], default="sft",
                        help="Training stage to verify: 'sft' or 'grpo'")
    args = parser.parse_args()
    
    model_path = "training/output-v2/qwen3-4b-effect-codegen-sft" if args.stage == "sft" else "training/output-v2/qwen3-4b-effect-codegen-v2"
    
    print(f"\n{'=' * 70}")
    print(f"VERIFICATION: {args.stage.upper()} MODEL")
    print(f"{'=' * 70}\n")
    
    success = verify_sft_model(model_path)
    
    print(f"\n{'=' * 70}")
    if success:
        print("[OK] VERIFICATION PASSED")
    else:
        print("[FAIL] VERIFICATION FAILED")
    print(f"{'=' * 70}\n")
    print("Next steps:")
    print("1. If SFT verification fails: Re-run SFT training with updated data format")
    print("2. If GRPO verification fails: Check GRPO training logs and reward function")
    print("3. If both pass: Model is ready for production use")
    sys.exit(0 if success else 1)
