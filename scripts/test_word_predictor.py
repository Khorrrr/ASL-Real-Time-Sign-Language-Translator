"""Test the word predictor."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))

from word_predictor import WordPredictor

predictor = WordPredictor("gpt2")

test_cases = [
    ("", "H"),
    ("", "HE"),
    ("", "HEL"),
    ("", "HELL"),
    ("Hello", "M"),
    ("Hello my", "N"),
    ("Hello my name", "I"),
]

for sentence, letters in test_cases:
    suggestions = predictor.get_suggestions(sentence, letters)
    context = f'"{sentence} {letters}"' if sentence else f'"{letters}"'
    print(f"Context: {context:30s} -> Suggestions: {suggestions}")