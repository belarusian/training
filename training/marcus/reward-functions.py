#!/usr/bin/env python
"""
Reward functions for Marcus Aurelius training.

These functions evaluate responses for:
- Citation accuracy
- Philosophical consistency
- Response quality
"""

import re
from typing import List, Dict, Any


def check_citation_present(response: str) -> float:
    """Check if response contains a citation (Book X, Section Y format)."""
    citation_pattern = r'Book\s+(?:\w+|\d+)[,\s]+(?:Section|Chapter)\s+\d+'
    if re.search(citation_pattern, response, re.IGNORECASE):
        return 1.0
    return 0.0


def check_citation_format(response: str) -> float:
    """Check if citation follows proper format."""
    citation_pattern = r'Book\s+(?:I+|IV|V?I{0,3})\s+(?:Section|Chapter)\s+\d+'
    if re.search(citation_pattern, response, re.IGNORECASE):
        return 1.0
    return 0.5


def check_stoic_principles(response: str) -> float:
    """Check if response aligns with Stoic principles."""
    stoic_keywords = [
        'virtue', 'reason', 'nature', 'control', 'accept',
        'indifferent', 'judgment', 'present', 'death', 'time'
    ]
    
    response_lower = response.lower()
    keyword_count = sum(1 for kw in stoic_keywords if kw in response_lower)
    
    # Reward based on keyword density
    if keyword_count >= 4:
        return 1.0
    elif keyword_count >= 2:
        return 0.7
    elif keyword_count >= 1:
        return 0.4
    return 0.0


def check_response_length(response: str) -> float:
    """Reward concise but complete responses (200-500 chars)."""
    length = len(response.strip())
    
    if 200 <= length <= 500:
        return 1.0
    elif 100 <= length < 200:
        return 0.5
    elif length < 100:
        return -0.5
    elif 500 < length <= 1000:
        return 0.7
    return 0.3


def check_response_quality(response: str) -> float:
    """Check overall response quality."""
    quality_score = 0.0
    
    # Check for complete sentences
    sentences = response.strip().split('.')
    if len([s for s in sentences if len(s.strip()) > 20]) >= 2:
        quality_score += 0.3
    
    # Check for proper punctuation
    if response.count('.') >= 1 and response.count(',') >= 1:
        quality_score += 0.2
    
    # Check for clear structure
    if '\n' in response:
        quality_score += 0.2
    
    # Check for closing statement
    closing_phrases = ['AWAITING_USER_INPUT', 'COMPLETED', 'I hope this helps', 'Does this answer your question']
    if any(phrase.lower() in response.lower() for phrase in closing_phrases):
        quality_score += 0.3
    
    return quality_score


def calculate_reward(response: str, expected_citation: str = None) -> float:
    """
    Calculate total reward for a response.
    
    Args:
        response: The model's response
        expected_citation: Expected citation (optional, for accuracy check)
    
    Returns:
        Total reward score
    """
    reward = 0.0
    
    # Citation present: +1.0
    reward += check_citation_present(response)
    
    # Citation format: +0.5
    reward += check_citation_format(response)
    
    # Stoic principles: +0.3
    reward += check_stoic_principles(response)
    
    # Response length: +0.5 max
    reward += check_response_length(response)
    
    # Response quality: +0.5 max
    reward += check_response_quality(response)
    
    return reward


def batch_calculate_rewards(responses: List[str], expected_citations: List[str] = None) -> List[float]:
    """Calculate rewards for a batch of responses."""
    if expected_citations is None:
        expected_citations = [None] * len(responses)
    
    return [calculate_reward(r, e) for r, e in zip(responses, expected_citations)]


if __name__ == "__main__":
    # Test the reward functions
    test_response = """As Marcus writes in Book 4, Section 1: "When another blames you or hates you..."
    
The Stoic approach is to pause, recognize that others act based on their own perceptions, and choose your response deliberately. The insult only harms you if you allow it to disturb your inner peace.

AWAITING_USER_INPUT"""
    
    reward = calculate_reward(test_response)
    print(f"Test response reward: {reward:.2f}")
    
    print("\nComponent breakdown:")
    print(f"  Citation present: {check_citation_present(test_response)}")
    print(f"  Citation format: {check_citation_format(test_response)}")
    print(f"  Stoic principles: {check_stoic_principles(test_response)}")
    print(f"  Response length: {check_response_length(test_response)}")
    print(f"  Response quality: {check_response_quality(test_response)}")
