"""
Test script for SFT and GRPO-trained models
Usage: 
    Stage 1 (SFT):  python training/test-training.py --stage sft
    Stage 2 (GRPO): python training/test-training.py --stage grpo
"""

import sys
import argparse
import torch
from unsloth import FastLanguageModel


def test_sft_model(model_path):
    """Test the trained model with a simple prompt."""
    
    print("=" * 70)
    print(f"Testing Model: {model_path}")
    print("=" * 70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("FATAL: CUDA is not available")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    try:
        print(f"\nLoading model from: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"FATAL: Could not load model: {e}")
        print(f"Make sure you ran the training stage and the model exists at: {model_path}")
        sys.exit(1)
    
    # Set to inference mode
    FastLanguageModel.for_inference(model)
    
    # Test prompts
    test_prompts = [
        "Generate an Effect service pattern with Schema validation",
        "Create an Effect Effect program that validates user input",
        "Write an Effect service that fetches data from an API",
    ]
    
    print("\n" + "=" * 70)
    print("Testing Inference")
    print("=" * 70)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"Prompt: {prompt}")
        
        messages = [
            {"role": "system", "content": "You are an expert TypeScript developer specializing in the Effect framework. Provide your reasoning first between <start_working_out> and <end_working_out>, then provide your complete TypeScript code implementation between <CODE> and </CODE>."},
            {"role": "user", "content": prompt},
        ]
        
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
            )
            
            response = tokenizer.decode(outputs[0])
            print(f"Response:\n{response}")
            
            # Check for key elements
            has_code = "<CODE>" in response
            has_schema = "Schema" in response
            has_effect = "effect" in response.lower()
            
            print(f"\nQuality checks:")
            print(f"  Has <CODE> tags: {has_code}")
            print(f"  Has Schema: {has_schema}")
            print(f"  Has Effect imports: {has_effect}")
            
        except Exception as e:
            print(f"Error during generation: {e}")
    
    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained model")
    parser.add_argument("--stage", choices=["sft", "grpo"], default="sft",
                        help="Training stage to test: 'sft' or 'grpo'")
    args = parser.parse_args()
    
    if args.stage == "sft":
        model_path = "training/output-v2/qwen3-4b-effect-codegen-sft"
    else:  # grpo
        model_path = "training/output-v2/qwen3-4b-effect-codegen-v2"
    
    test_sft_model(model_path)
