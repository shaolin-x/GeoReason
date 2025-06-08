#!/usr/bin/env python3
"""
Direction and Distance Results Accuracy Analysis
This script calculates accuracy scores by comparing expected answers with model answers
using exact string matching. It also prints all mismatched entries with their correct
and model-provided answers.
"""

import json
import argparse
from typing import Dict, List, Tuple


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


def calculate_accuracy(data: List[Dict]) -> Tuple[float, int, int, int, int, List[Tuple[int, str, str]]]:
    """
    Calculate accuracy score using exact match between answer and model_answer.
    
    Returns:
        Tuple of (accuracy, correct_count, total_valid, total_entries, null_count, mismatches)
        where mismatches is a list of (index, expected_answer, model_answer).
    """
    total_entries = len(data)
    correct_count = 0
    valid_entries = 0  # Entries where ground truth answer is not null
    null_count = 0     # Count of entries with null ground truth
    mismatches: List[Tuple[int, str, str]] = []
    
    for i, entry in enumerate(data):
        answer = entry.get('answer')
        model_answer = entry.get('model_answer')
        
        # Handle different types of null/empty values
        if answer is None or answer == 'null' or answer == '':
            null_count += 1
            continue
            
        valid_entries += 1
        
        # Perform exact string match (case-sensitive)
        if model_answer is not None and answer == model_answer:
            correct_count += 1
        else:
            # Record mismatch (index in file, expected, got)
            mismatches.append((i + 1, answer, model_answer if model_answer is not None else "None"))
    
    # Calculate accuracy
    accuracy = correct_count / valid_entries if valid_entries > 0 else 0.0
    
    return accuracy, correct_count, valid_entries, total_entries, null_count, mismatches


def analyze_results(data: List[Dict]) -> None:
    """Analyze and display detailed results, including all mismatches."""
    accuracy, correct, valid, total, null_count, mismatches = calculate_accuracy(data)
    
    print("=" * 60)
    print("DIRECTION AND DISTANCE ACCURACY ANALYSIS")
    print("=" * 60)
    print(f"Total entries in file: {total}")
    print(f"Valid entries (with ground truth): {valid}")
    print(f"Entries with null/empty ground truth: {null_count}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)
    
    # Show null entries for debugging
    print(f"\nNull/empty ground truth entries:")
    print("-" * 40)
    
    null_examples = 0
    for i, entry in enumerate(data):
        answer = entry.get('answer')
        if answer is None or answer == 'null' or answer == '':
            null_examples += 1
            if null_examples <= 3:  # Show first 3 null entries
                query = entry.get('query', 'N/A')
                model_answer = entry.get('model_answer', 'N/A')
                print(f"Entry {i+1}:")
                print(f"  Query: {query}")
                print(f"  Ground Truth: {repr(answer)}")
                print(f"  Model Answer: {model_answer}")
                print()
    
    if null_examples == 0:
        print("No null entries found!")
    elif null_examples > 3:
        print(f"... and {null_examples - 3} more null entries")
    
    # Show all mismatches
    print(f"\nAll mismatched entries (Expected vs. Model):")
    print("-" * 40)
    if mismatches:
        for idx, expected, got in mismatches:
            print(f"Entry {idx}: Expected: {expected} | Got: {got}")
    else:
        print("No mismatches found!")


def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy for direction and distance results')
    parser.add_argument('jsonl_file', help='Path to the JSONL file containing results')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.jsonl_file}")
    data = load_jsonl(args.jsonl_file)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    # Calculate and display results
    if args.detailed:
        analyze_results(data)
    else:
        accuracy, correct, valid, total, null_count, mismatches = calculate_accuracy(data)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Correct: {correct}/{valid} (valid entries)")
        print(f"Null entries: {null_count}/{total}")
        # Print mismatches even in non-detailed mode
        if mismatches:
            print("\nMismatched entries:")
            for idx, expected, got in mismatches:
                print(f"Entry {idx}: Expected: {expected} | Got: {got}")


if __name__ == "__main__":
    main()
