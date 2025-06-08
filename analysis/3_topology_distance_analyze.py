#!/usr/bin/env python3
"""
Topology and Distance Results Accuracy Analysis
This script calculates accuracy scores for topology+distance queries that can have:
- None answers or single state answers
- Blank/empty model responses (treated as correct for None expected answers)
- Bracketed format in expected answers that needs to be cleaned

Performs exact match accuracy calculation with special handling for blank answers.
Additionally, any entry that is not a correct match will have its expected
and model answers printed for debugging.
"""

import json
import argparse
import re
from typing import Dict, List, Tuple, Optional


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return data


def clean_bracketed_answer(answer) -> Optional[str]:
    """Clean the bracketed format from expected answers."""
    if answer is None:
        return None
    
    if isinstance(answer, list):
        # If it's already a list with one item, extract it
        if len(answer) == 1:
            return str(answer[0]).strip()
        elif len(answer) == 0:
            return None
        else:
            # Multiple items - take the first one
            return str(answer[0]).strip()
    
    if isinstance(answer, str):
        answer = answer.strip()
        
        if answer.lower() in ['none', 'null', '']:
            return None
        
        # Remove brackets if present
        if answer.startswith('[') and answer.endswith(']'):
            inner = answer[1:-1].strip()
            if inner.startswith('"') and inner.endswith('"'):
                inner = inner[1:-1]
            elif inner.startswith("'") and inner.endswith("'"):
                inner = inner[1:-1]
            
            if inner.lower() in ['none', 'null', '']:
                return None
            return inner.strip() if inner else None
        
        return answer if answer else None
    
    return None


def normalize_answer(answer: str) -> Optional[str]:
    """Normalize an answer for comparison."""
    if not answer:
        return None
    
    normalized = answer.lower().strip()
    
    if normalized in ['none', 'null', 'n/a', '', 'no applicable state', 'no valid answer']:
        return None
    
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def calculate_accuracy(data: List[Dict]) -> Tuple[float, int, int, int, int, List[Tuple[int, Optional[str], Optional[str]]]]:
    """
    Calculate accuracy score for topology+distance results, and collect mismatches.
    
    Returns:
        Tuple of (accuracy, correct_count, total_entries, none_expected_count, blank_model_count, mismatches)
        where mismatches is a list of tuples (index, expected_normalized, model_normalized).
    """
    total_entries = len(data)
    correct_count = 0
    none_expected_count = 0
    blank_model_count = 0
    mismatches: List[Tuple[int, Optional[str], Optional[str]]] = []
    
    for idx, entry in enumerate(data, start=1):
        raw_answer = entry.get('answer')
        raw_model = entry.get('model_answer', '')
        
        expected = clean_bracketed_answer(raw_answer)
        expected_norm = normalize_answer(expected) if expected is not None else None
        
        model_raw = raw_model.strip()
        model_norm = normalize_answer(model_raw) if model_raw else None
        
        if expected_norm is None:
            none_expected_count += 1
        
        if not model_raw:
            blank_model_count += 1
        
        is_correct = False
        if expected_norm is None and model_norm is None:
            is_correct = True
        elif expected_norm is None and not model_raw:
            is_correct = True
        elif expected_norm is not None and model_norm is not None:
            is_correct = (expected_norm == model_norm)
        
        if is_correct:
            correct_count += 1
        else:
            mismatches.append((idx, expected_norm, model_norm))
    
    accuracy = correct_count / total_entries if total_entries > 0 else 0.0
    return accuracy, correct_count, total_entries, none_expected_count, blank_model_count, mismatches


def analyze_results(data: List[Dict]) -> None:
    """Analyze and display detailed results, printing all mismatches."""
    (accuracy, correct, total, none_expected, blank_model, mismatches) = calculate_accuracy(data)
    
    print("=" * 70)
    print("TOPOLOGY AND DISTANCE ACCURACY ANALYSIS")
    print("=" * 70)
    print(f"Total entries in file: {total}")
    print(f"Expected 'None' answers: {none_expected}")
    print(f"Blank model responses: {blank_model}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 70)
    
    # Show some examples of mismatches
    print(f"\nAll mismatched entries (expected_norm vs. model_norm):")
    print("-" * 50)
    
    if not mismatches:
        print("No mismatches found!")
    else:
        for idx, exp_norm, mod_norm in mismatches:
            entry = data[idx - 1]
            query = entry.get('query', 'N/A')
            raw_expected = entry.get('answer')
            raw_model = entry.get('model_answer', '')
            print(f"Entry {idx}:")
            print(f"  Query: {query}")
            print(f"  Raw Expected: {repr(raw_expected)}")
            print(f"  Cleaned Expected: {repr(exp_norm)}")
            print(f"  Raw Model: {repr(raw_model)}")
            print(f"  Model Normalized: {repr(mod_norm)}\n")
    
        print(f"Total mismatches: {len(mismatches)}")


def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy for topology and distance results')
    parser.add_argument('jsonl_file', help='Path to the JSONL file containing results')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.jsonl_file}")
    data = load_jsonl(args.jsonl_file)
    if not data:
        print("No data loaded. Exiting.")
        return
    
    if args.detailed:
        analyze_results(data)
    else:
        accuracy, correct, total, none_expected, blank_model, mismatches = calculate_accuracy(data)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Correct: {correct}/{total}")
        print(f"None expected: {none_expected}/{total}")
        print(f"Blank model responses: {blank_model}/{total}")
        
        print(f"\nMismatched entries (expected_norm vs. model_norm):")
        print("-" * 50)
        if not mismatches:
            print("No mismatches found!")
        else:
            for idx, exp_norm, mod_norm in mismatches:
                entry = data[idx - 1]
                query = entry.get('query', 'N/A')
                raw_expected = entry.get('answer')
                raw_model = entry.get('model_answer', '')
                print(f"Entry {idx}:")
                print(f"  Query: {query}")
                print(f"  Raw Expected: {repr(raw_expected)}")
                print(f"  Cleaned Expected: {repr(exp_norm)}")
                print(f"  Raw Model: {repr(raw_model)}")
                print(f"  Model Normalized: {repr(mod_norm)}\n")
            
            print(f"Total mismatches: {len(mismatches)}")


if __name__ == "__main__":
    main()
