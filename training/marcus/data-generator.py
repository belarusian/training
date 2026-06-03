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


def generate_questions_from_passage(passages: List[Dict[str, str]]) -> List[Dict[str, str]]:
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
        } for p in passages[:100]]  # Limit to first 100 for quick extraction
        
    elif args.mode == "synthetic":
        # Generate questions (this would use Gemma 4 + RAG in production)
        # For now, we create a placeholder dataset
        questions = generate_questions_from_passage(passages)
        
        dataset = []
        for q_info in questions[:50]:  # Limit to first 50
            passage = q_info["passage"]
            question = q_info["questions"][0]  # Use first template
            
            dataset.append({
                "messages": [
                    {"role": "user", "content": f"{question}\n\nPassage: {passage['text'][:300]}..."},
                    {"role": "assistant", "content": f"As Marcus writes in {passage['book']}, Section {passage['section']}: '{passage['text'][:200]}...'\n\nThis teaches us that we should focus on what is within our control and accept what is not."}
                ]
            })
    
    # Save dataset
    save_dataset(dataset, args.output)


if __name__ == "__main__":
    main()
