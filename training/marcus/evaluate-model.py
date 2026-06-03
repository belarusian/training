#!/usr/bin/env python
"""
Standalone evaluation script for Marcus Aurelius model.

Evaluates the model on a set of test prompts and scores responses.
"""

import sys
import os
import json
import argparse
from typing import List, Dict, Any

# Import Unsloth
from unsloth import FastLanguageModel


def evaluate_model(
    model_path: str,
    test_prompts: List[str] = None,
    output_path: str = "marcus_eval_results.json",
):
    """Evaluate Marcus Aurelius model."""
    
    print("=" * 70)
    print(f"Marcus Aurelius Model Evaluation")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.")
    
    # Default test prompts
    if test_prompts is None:
        test_prompts = [
            "How should I respond when someone insults me?",
            "I'm struggling with anxiety about the future. What should I do?",
            "How can I find peace in a chaotic world?",
            "What does Marcus say about dealing with difficult people?",
            "How should I approach death and mortality?",
        ]
    
    print(f"\nTesting {len(test_prompts)} prompts...")
    
    results = []
    for i, prompt in enumerate(test_prompts):
        print(f"\n[{i+1}/{len(test_prompts)}] Prompt: {prompt[:60]}...")
        
        # Generate response
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response (remove prompt)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Score response
        result = {
            "prompt": prompt,
            "response": response,
            "response_length": len(response),
        }
        results.append(result)
        
        print(f"    Response: {response[:150]}...")
        print(f"    Length: {len(response)} chars")
    
    # Save results
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    avg_length = sum(r["response_length"] for r in results) / len(results)
    print(f"\n✅ Evaluation complete!")
    print(f"   Average response length: {avg_length:.1f} chars")
    print(f"   Results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Marcus Aurelius model")
    parser.add_argument("--model", required=True,
                       help="Path to trained model")
    parser.add_argument("--prompts", default=None,
                       help="JSON file with test prompts (optional)")
    parser.add_argument("--output", default="marcus_eval_results.json",
                       help="Output JSON file path")
    args = parser.parse_args()
    
    # Load prompts if provided
    test_prompts = None
    if args.prompts and os.path.exists(args.prompts):
        with open(args.prompts, 'r', encoding='utf-8') as f:
            test_prompts = json.load(f)
    
    try:
        evaluate_model(
            model_path=args.model,
            test_prompts=test_prompts,
            output_path=args.output,
        )
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
