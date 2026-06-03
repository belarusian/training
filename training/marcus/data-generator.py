#!/usr/bin/env python
"""
Generate synthetic Q&A data from Marcus Aurelius' Meditations.

This script uses Gemma 4 + RAG as a teacher to generate high-quality
answers to questions about Stoic philosophy.

Usage:
    .\unsloth_env\Scripts\python.exe training\marcus\data-generator.py --mode synthetic
    .\unsloth_env\Scripts\python.exe training\marcus\data-generator.py --mode extract
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re
import requests
from urllib.parse import urljoin


def parse_meditations_into_passages(text: str) -> List[Dict[str, str]]:
    """Parse Meditations into passages with Book/Section metadata."""
    passages = []
    
    # Match patterns like "Book I, Section 1" or "Book 1, Section 1"
    passage_regex = re.compile(
        r'(Book\s+\w+|Book\s+\d+)[,\s]+(?:Section|Chapter)\s+(\d+)[.\s]*\n'
        r'((?:(?!Book\s+\w+|Book\s+\d+)[\s\S])*?)(?=(?:Book\s+\w+|Book\s+\d+)[,\s]+(?:Section|Chapter)\s+\d+|$)',
        re.MULTILINE
    )
    
    matches = passage_regex.finditer(text)
    for match in matches:
        book = match.group(1).strip()
        section = match.group(2).strip()
        content = match.group(3).strip()
        
        if len(content) > 50:  # Skip very short passages
            passages.append({
                "book": book,
                "section": section,
                "text": content
            })
    
    print(f"[DataGenerator] Parsed {len(passages)} passages from Meditations")
    return passages


def generate_questions_from_passage(passages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Generate questions that each passage answers."""
    questions = []
    
    for i, passage in enumerate(passages):
        # Simple question templates based on passage content
        question_templates = [
            f"What does Marcus say about this passage?",
            f"Explain the Stoic principle in Book {passage['section']}.",
            f"How should one apply this teaching today?",
            f"What is the main lesson from {passage['book']} {passage['section']}?",
        ]
        
        # Use first sentence as basis for question
        sentences = passage['text'].split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            question_templates.append(f"What does Marcus mean by: '{first_sentence[:100]}...'?")
        
        questions.append({
            "passage": passage,
            "questions": question_templates
        })
    
    return questions


def load_meditations(filepath: str) -> str:
    """Load Meditations text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def query_gemma4(prompt: str, system_prompt: str = None) -> str:
    """
    Query Gemma 4 API.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
    
    Returns:
        Model response
    """
    # Gemma 4 API configuration
    base_url = "http://127.0.0.1:8888/v1"
    api_key = "sk-unsloth-e1e411873c6d9cbfcff35c3c2ef6c5f8"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Make request
    response = requests.post(
        urljoin(base_url, "chat/completions"),
        headers=headers,
        json={
            "model": "gemma-4",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.95,
        },
        timeout=60
    )
    
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]


def generate_with_teacher(prompt: str, meditations_context: str = None) -> str:
    """
    Generate answer using Gemma 4 + RAG context.
    
    Args:
        prompt: User question
        meditations_context: RAG-retrieved Meditations passages
    
    Returns:
        High-quality Stoic-style answer
    """
    system_prompt = """You are a Stoic philosophy guide that helps users reflect on present-moment challenges using Marcus Aurelius' Meditations as the authoritative source.

## Context
The following is a selection from Marcus Aurelius' Meditations. Use these passages to inform your responses:

{{meditations}}

## Instructions
1. Read the Meditations text above carefully - use relevant passages when answering questions
2. Respond as a Stoic guide - be calm, thoughtful, and practical
3. Cite specific book/section when referencing Meditations (e.g., "As Marcus writes in Book 4, Section 2...")
4. Keep responses focused on the present moment - Stoicism is about responding to now
5. Be concise but profound - use clear language to convey deep philosophical insights
6. End with one of these: AWAITING_USER_INPUT or COMPLETED"""
    
    if meditations_context:
        system_prompt = system_prompt.replace("{{meditations}}", meditations_context)
    else:
        system_prompt = system_prompt.replace("{{meditations}}", "No Meditations context available.")
    
    return query_gemma4(prompt, system_prompt)


def save_dataset(dataset: List[Dict[str, Any]], filepath: str):
    """Save dataset to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    print(f"[DataGenerator] Saved {len(dataset)} samples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate Marcus Aurelius training data")
    parser.add_argument("--mode", choices=["synthetic", "extract"], default="extract",
                       help="Mode: synthetic (generate questions) or extract (parse text)")
    parser.add_argument("--meditations", default="../meditations.mb.txt",
                       help="Path to Meditations text file")
    parser.add_argument("--output", default="marcus_training_data.json",
                       help="Output JSON file path")
    parser.add_argument("--limit", type=int, default=50,
                       help="Maximum number of samples to generate")
    args = parser.parse_args()
    
    # Load Meditations
    meditations_path = Path(__file__).parent.parent.parent / args.meditations
    if not meditations_path.exists():
        print(f"[ERROR] Meditations file not found: {meditations_path}")
        sys.exit(1)
    
    print(f"[DataGenerator] Loading Meditations from {meditations_path}")
    text = load_meditations(str(meditations_path))
    
    # Parse into passages
    passages = parse_meditations_into_passages(text)
    
    if args.mode == "extract":
        # Just extract passages
        dataset = [{
            "messages": [
                {"role": "user", "content": f"Read this passage from Marcus Aurelius: {p['text'][:500]}..."},
                {"role": "assistant", "content": f"This is from {p['book']}, Section {p['section']}. The passage discusses Stoic principles about life, virtue, and resilience."}
            ]
        } for p in passages[:args.limit]]
        
    elif args.mode == "synthetic":
        # Generate questions with Gemma 4 teacher
        questions = generate_questions_from_passage(passages)
        
        dataset = []
        for i, q_info in enumerate(questions[:args.limit]):
            passage = q_info["passage"]
            question = q_info["questions"][0]  # Use first template
            
            print(f"\n[{i+1}/{args.limit}] Generating with Gemma 4...")
            print(f"    Question: {question[:60]}...")
            
            # Generate answer with Gemma 4
            try:
                answer = generate_with_teacher(
                    question,
                    meditations_context=f"From {passage['book']}, Section {passage['section']}: {passage['text'][:500]}..."
                )
                print(f"    Answer: {answer[:150]}...")
                
                dataset.append({
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                })
            except Exception as e:
                print(f"    [ERROR] Failed to generate: {e}")
                # Fallback to template answer
                dataset.append({
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": f"As Marcus writes in {passage['book']}, Section {passage['section']}: '{passage['text'][:200]}...'\n\nThis teaches us that we should focus on what is within our control and accept what is not."}
                    ]
                })
    
    # Save dataset
    save_dataset(dataset, args.output)


if __name__ == "__main__":
    main()
