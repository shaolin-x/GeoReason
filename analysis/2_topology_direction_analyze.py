#!/usr/bin/env python3
"""
Topology and Direction Results Accuracy Analysis
This script calculates accuracy scores for topology+direction queries that can have:
- None/null answers
- Multiple park names as answers (in arrays or comma-separated strings)
- Blank/empty model responses

Two metrics are calculated:
1) Completely correct responses (exact match of all parks)
2) Partially correct responses (at least one park matches)

Additionally, any entry that is not a complete match will have its expected
and model answers printed for debugging.
"""

import json
import argparse
import re
from typing import Dict, List, Tuple, Set, Optional


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


def normalize_park_name(park_name: str) -> str:
    """Normalize park name for comparison."""
    if not park_name:
        return ""
    
    # Convert to lowercase and strip whitespace
    name = park_name.lower().strip()
    
    # Remove common suffixes and variations
    suffixes_to_remove = [
        'national park',
        'national park and preserve',
        'national park & preserve',
        'and preserve',
        '& preserve',
        'national monument',
        'national monument and preserve',
        'national historic park',
        'national historical park',
        'national recreation area',
        'national seashore',
        'national lakeshore'
    ]
    
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
            break
    
    # Remove extra whitespace and normalize punctuation
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'[^\w\s-]', '', name)
    
    return name.strip()


def parse_parks_from_text(text: str) -> Set[str]:
    """Parse park names from text (handles various formats)."""
    if not text or text.strip() == "":
        return set()
    
    # Handle None case
    if text.lower().strip() in ['none', 'null', 'n/a']:
        return set()
    
    parks = set()
    
    # Split by common delimiters
    delimiters = [',', ';', '\n', ' and ', ' & ']
    for delimiter in delimiters:
        if delimiter in text:
            parts = text.split(delimiter)
            for part in parts:
                normalized = normalize_park_name(part)
                if normalized:
                    parks.add(normalized)
            return parks
    
    # If no delimiters found, treat as single park
    normalized = normalize_park_name(text)
    if normalized:
        parks.add(normalized)
    
    return parks


def parse_answer_field(answer) -> Set[str]:
    """Parse the answer field which can be a string, list, or None."""
    if answer is None:
        return set()
    
    if isinstance(answer, list):
        parks = set()
        for item in answer:
            if isinstance(item, str):
                normalized = normalize_park_name(item)
                if normalized:
                    parks.add(normalized)
        return parks
    
    if isinstance(answer, str):
        if answer.lower().strip() in ['none', 'null', '']:
            return set()
        return parse_parks_from_text(answer)
    
    return set()


def calculate_accuracy_metrics(data: List[Dict]) -> Tuple[float, float, int, int, int, int, int]:
    """
    Calculate accuracy metrics for topology+direction results.
    
    Returns:
        Tuple of (complete_accuracy, partial_accuracy, complete_correct, partial_correct, 
                 valid_entries, total_entries, null_ground_truth_count)
    """
    total_entries = len(data)
    complete_correct = 0
    partial_correct = 0
    valid_entries = 0
    null_ground_truth_count = 0
    
    for entry in data:
        answer = entry.get('answer')
        model_answer = entry.get('model_answer', '')
        
        # Parse ground truth answer
        ground_truth_parks = parse_answer_field(answer)
        
        # Skip entries where ground truth is None or empty
        if not ground_truth_parks and (answer is None or 
                                      (isinstance(answer, str) and answer.lower().strip() in ['none', 'null', ''])):
            null_ground_truth_count += 1
            continue
        
        valid_entries += 1
        
        # Parse model answer
        model_parks = parse_parks_from_text(model_answer)
        
        # Calculate matches
        if ground_truth_parks == model_parks:
            # Exact match (completely correct)
            complete_correct += 1
            partial_correct += 1
        elif ground_truth_parks.intersection(model_parks):
            # At least one park matches (partially correct)
            partial_correct += 1
    
    # Calculate accuracy rates
    complete_accuracy = complete_correct / valid_entries if valid_entries > 0 else 0.0
    partial_accuracy = partial_correct / valid_entries if valid_entries > 0 else 0.0
    
    return (complete_accuracy, partial_accuracy, complete_correct, partial_correct, 
            valid_entries, total_entries, null_ground_truth_count)


def analyze_results(data: List[Dict]) -> None:
    """Analyze and display detailed results, printing all mismatches."""
    (complete_acc, partial_acc, complete_correct, partial_correct, 
     valid, total, null_count) = calculate_accuracy_metrics(data)
    
    print("=" * 70)
    print("TOPOLOGY AND DIRECTION ACCURACY ANALYSIS")
    print("=" * 70)
    print(f"Total entries in file: {total}")
    print(f"Valid entries (with non-null ground truth): {valid}")
    print(f"Entries with null/None ground truth: {null_count}")
    print()
    print("ACCURACY METRICS:")
    print(f"1) Complete accuracy (exact match): {complete_acc:.4f} ({complete_acc*100:.2f}%)")
    print(f"   Completely correct responses: {complete_correct}/{valid}")
    print()
    print(f"2) Partial accuracy (≥1 correct park): {partial_acc:.4f} ({partial_acc*100:.2f}%)")
    print(f"   Partially correct responses: {partial_correct}/{valid}")
    print("=" * 70)
    
    # Show examples of null entries
    print(f"\nSample null/None ground truth entries:")
    print("-" * 50)
    
    null_examples = 0
    for i, entry in enumerate(data):
        answer = entry.get('answer')
        if (answer is None or 
            (isinstance(answer, str) and answer.lower().strip() in ['none', 'null', ''])):
            null_examples += 1
            if null_examples <= 3:
                query = entry.get('query', 'N/A')[:70] + "..." if len(entry.get('query', '')) > 70 else entry.get('query', 'N/A')
                model_answer = entry.get('model_answer', 'N/A')[:50] + "..." if len(entry.get('model_answer', '')) > 50 else entry.get('model_answer', 'N/A')
                print(f"Entry {i+1}:")
                print(f"  Query: {query}")
                print(f"  Ground Truth: {repr(answer)}")
                print(f"  Model Answer: {model_answer}")
                print()
    
    if null_examples == 0:
        print("No null entries found!")
    elif null_examples > 3:
        print(f"... and {null_examples - 3} more null entries")
    
    # Print all mismatches (those that are not a complete match)
    print(f"\nAll mismatched entries (expected vs. model):")
    print("-" * 50)
    
    mismatch_count = 0
    for i, entry in enumerate(data):
        answer = entry.get('answer')
        model_answer = entry.get('model_answer', '')
        
        ground_truth_parks = parse_answer_field(answer)
        
        # Skip null entries
        if not ground_truth_parks and (answer is None or 
                                      (isinstance(answer, str) and answer.lower().strip() in ['none', 'null', ''])):
            continue
        
        model_parks = parse_parks_from_text(model_answer)
        
        if ground_truth_parks != model_parks:
            mismatch_count += 1
            query = entry.get('query', 'N/A')[:70] + "..." if len(entry.get('query', '')) > 70 else entry.get('query', 'N/A')
            print(f"Entry {i+1}:")
            print(f"  Query: {query}")
            print(f"  Expected: {sorted(list(ground_truth_parks))}")
            print(f"  Got:      {sorted(list(model_parks))}")
            print()
    
    if mismatch_count == 0:
        print("No mismatches found!")
    else:
        print(f"Total mismatches: {mismatch_count}")
    

def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy for topology and direction results')
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
        (complete_acc, partial_acc, complete_correct, partial_correct, 
         valid, total, null_count) = calculate_accuracy_metrics(data)
        
        print(f"Complete accuracy: {complete_acc:.4f} ({complete_acc*100:.2f}%)")
        print(f"Partial accuracy: {partial_acc:.4f} ({partial_acc*100:.2f}%)")
        print(f"Complete correct: {complete_correct}/{valid}")
        print(f"Partial correct: {partial_correct}/{valid}")
        print(f"Null entries: {null_count}/{total}")
        
        # Also print mismatches in non-detailed mode
        print(f"\nMismatched entries (expected vs. model):")
        print("-" * 50)
        
        mismatch_count = 0
        for i, entry in enumerate(data):
            answer = entry.get('answer')
            model_answer = entry.get('model_answer', '')
            
            ground_truth_parks = parse_answer_field(answer)
            
            # Skip null entries
            if not ground_truth_parks and (answer is None or 
                                          (isinstance(answer, str) and answer.lower().strip() in ['none', 'null', ''])):
                continue
            
            model_parks = parse_parks_from_text(model_answer)
            
            if ground_truth_parks != model_parks:
                mismatch_count += 1
                query = entry.get('query', 'N/A')[:70] + "..." if len(entry.get('query', '')) > 70 else entry.get('query', 'N/A')
                print(f"Entry {i+1}:")
                print(f"  Query: {query}")
                print(f"  Expected: {sorted(list(ground_truth_parks))}")
                print(f"  Got:      {sorted(list(model_parks))}")
                print()
        
        if mismatch_count == 0:
            print("No mismatches found!")
        else:
            print(f"Total mismatches: {mismatch_count}")


if __name__ == "__main__":
    main()
