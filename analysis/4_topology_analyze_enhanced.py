#!/usr/bin/env python3
"""
Topology Results Difference Score Analysis
This script calculates step differences between model answers and expected answers
based on the conceptual neighborhood of topological relations as defined in Figure 1
of the paper (https://arxiv.org/pdf/2505.17136).

The script uses the 9-IM topology conceptual neighborhood graph where relationships
have specific neighbor connections and step distances between them.

Enhanced with better handling of verbose model responses and edge cases.
"""

import json
import argparse
import re
import statistics
import os
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, deque, Counter

# File name configurations
file_name_1 = ["deepseekchat.jsonl", "deepseekreasoner.jsonl", "gpt-4.1.jsonl", "o3-mini.jsonl"]
file_name_2 = ["gemini-2.5-pro-preview-05-06","gemini-2.5-flash-preview-05-20","claude-sonnet-4-20250514"]
file_name_3 = ["Llama-3.3-70B-Instruct-Turbo-Free"]

def extract_model_name(file_path):
    """Extract model name from file path for clearer output"""
    if "deepseekchat.jsonl" in file_path:
        return "DeepSeek-Chat"
    elif "deepseekreasoner.jsonl" in file_path:
        return "DeepSeek-Reasoner"
    elif "gpt-4.1.jsonl" in file_path:
        return "GPT-4.1"
    elif "o3-mini.jsonl" in file_path:
        return "O3-Mini"
    elif "gemini-2.5-pro-preview-05-06" in file_path:
        return "Gemini-2.5-Pro-Preview"
    elif "gemini-2.5-flash-preview-05-20" in file_path:
        return "Gemini-2.5-Flash-Preview"
    elif "claude-sonnet-4-20250514" in file_path:
        return "Claude-Sonnet-4"
    elif "Llama-3.3-70B-Instruct-Turbo-Free" in file_path:
        return "Llama-3.3-70B-Instruct"
    else:
        # Extract filename as fallback
        return os.path.basename(file_path).replace('.jsonl', '').replace('_queries_output', '')

def generate_input_file(cur_type):
    """Generate list of input files for analysis"""
    input_files = []
    # cur_type = "topology"
    for i in file_name_1:
        input_files.append("./tier1_results/"+cur_type+"_merged_results_"+i)
    for i in file_name_2:
        input_files.append("./tier1_results/"+"model_output/"+i + "/" + cur_type+"_queries_output.jsonl")
    for i in file_name_3:
        input_files.append("./tier1_results/"+i + "/" + cur_type+"_queries_output.jsonl")
    return input_files

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


def build_topology_graph() -> Dict[str, Set[str]]:
    """
    Build the conceptual neighborhood graph based on Figure 1b (9-IM).
    Returns a dictionary where keys are topology relations and values are sets of directly connected neighbors.
    
    Fixed to ensure the graph is properly connected and includes all synonyms.
    """
    # Based on Figure 1b (9-IM conceptual neighborhood) - ensuring connectivity
    base_graph = {
        'disjoint': {'meet'},
        'meet': {'disjoint', 'overlap'},
        'overlap': {'meet', 'equal', 'coveredBy', 'covers'},
        'equal': {'overlap', 'coveredBy', 'covers'},
        'coveredBy': {'overlap', 'inside', 'equal'},
        'inside': {'coveredBy'},
        'covers': {'overlap', 'contains', 'equal'},
        'contains': {'covers'}
    }

    # Synonyms and variations mapping
    synonyms = {
        'within': 'inside',
        'covered by': 'coveredBy',
        'touch': 'meet',
        'touches': 'meet',
        'overlaps': 'overlap'
    }

    # Create reverse mapping for canonical to synonyms
    canonical_to_synonyms = defaultdict(set)
    for syn, canonical in synonyms.items():
        canonical_to_synonyms[canonical].add(syn)

    # Build expanded graph
    expanded_graph: Dict[str, Set[str]] = {}
    
    # Add all canonical relations and their synonyms
    all_relations = set(base_graph.keys())
    for canonical in base_graph.keys():
        all_relations.update(canonical_to_synonyms[canonical])
    
    # For each relation (canonical and synonyms), add neighbors
    for relation in all_relations:
        # Get canonical form
        canonical = synonyms.get(relation, relation)
        
        if canonical in base_graph:
            neighbors = set()
            # Add canonical neighbors
            for neighbor in base_graph[canonical]:
                neighbors.add(neighbor)
                # Add synonyms of neighbors
                neighbors.update(canonical_to_synonyms[neighbor])
            
            expanded_graph[relation] = neighbors
        else:
            # If not found in base graph, it might be an unknown relation
            expanded_graph[relation] = set()

    return expanded_graph


def extract_topology_relations(text: str) -> List[str]:
    """
    Extract topology relation names from potentially verbose text.
    Handles cases where the model provides explanations mixed with relation names.
    """
    if not text:
        return []
    
    # List of all possible topology relations (canonical and synonyms)
    all_relations = {
        'disjoint', 'meet', 'overlap', 'equal', 'coveredBy', 'inside', 'covers', 'contains',
        'within', 'covered by', 'touch', 'touches', 'overlaps', 'touching', 'intersects', 
        'intersect', 'separate', 'separated', 'same', 'identical', 'encompasses', 
        'encompass', 'enclosed by', 'encloses'
    }
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    found_relations = []
    
    # Sort relations by length (longest first) to avoid partial matches
    sorted_relations = sorted(all_relations, key=len, reverse=True)
    
    for relation in sorted_relations:
        relation_lower = relation.lower()
        
        # Use word boundaries to avoid partial matches
        # Handle both single words and multi-word relations
        if ' ' in relation_lower:
            # Multi-word relation - look for exact phrase
            pattern = r'\b' + re.escape(relation_lower) + r'\b'
        else:
            # Single word relation - use word boundaries
            pattern = r'\b' + re.escape(relation_lower) + r'\b'
        
        if re.search(pattern, text_lower):
            found_relations.append(relation)
            # Remove found relation from text to avoid duplicates
            text_lower = re.sub(pattern, '', text_lower)
    
    return found_relations


def normalize_topology_relation(relation: str) -> str:
    """
    Normalize topology relation to canonical form.
    Enhanced to handle verbose responses with explanatory text.
    """
    if not relation:
        return ""

    # First, try to extract actual topology relations from the text
    extracted_relations = extract_topology_relations(relation)
    
    if extracted_relations:
        # If we found relations, use the first one
        relation = extracted_relations[0]
    else:
        # Fall back to original approach if no relations found
        relation = relation.lower().strip()
        
        # Handle multiple relations by taking the first one
        if ',' in relation:
            relation = relation.split(',')[0].strip()
        
        # Clean up whitespace
        relation = re.sub(r'\s+', ' ', relation)
        
        # Remove common prefixes/suffixes that might cause issues
        relation = re.sub(r'^(the\s+|a\s+|an\s+)', '', relation)
        relation = re.sub(r'\s+(relation|relationship)$', '', relation)

    # Canonical mapping
    canonical_mapping = {
        'within': 'inside',
        'covered by': 'coveredBy',
        'touch': 'meet',
        'touches': 'meet',
        'overlaps': 'overlap',
        'touching': 'meet',
        'intersects': 'overlap',
        'intersect': 'overlap',
        'separate': 'disjoint',
        'separated': 'disjoint',
        'same': 'equal',
        'identical': 'equal',
        'encompasses': 'contains',
        'encompass': 'contains',
        'enclosed by': 'inside',
        'encloses': 'contains'
    }
    
    relation = relation.lower().strip()
    return canonical_mapping.get(relation, relation)


def calculate_shortest_distance(graph: Dict[str, Set[str]], start: str, end: str) -> int:
    """
    Calculate shortest path distance between two topology relations using BFS.
    Returns the number of steps (edges) between start and end relations.
    """
    if start == end:
        return 0
    
    # Check if both relations exist in graph
    if start not in graph or end not in graph:
        print(f"Warning: Relation not found in graph - start: '{start}', end: '{end}'")
        return float('inf')

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current, dist = queue.popleft()
        
        if current not in graph:
            continue
            
        for neighbor in graph[current]:
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf')


def verify_graph_connectivity(graph: Dict[str, Set[str]]) -> bool:
    """Verify that the graph is connected."""
    if not graph:
        return False
    
    # Get all canonical relations (the main 8 relations)
    canonical_relations = {
        'disjoint', 'meet', 'overlap', 'equal',
        'coveredBy', 'inside', 'covers', 'contains'
    }
    
    # Check connectivity between all pairs of canonical relations
    start_node = next(iter(canonical_relations))
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    
    while queue:
        current = queue.popleft()
        if current in graph:
            for neighbor in graph[current]:
                if neighbor in canonical_relations and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    missing = canonical_relations - visited
    if missing:
        print(f"Warning: Graph is not connected. Missing nodes: {missing}")
        return False
    
    return True


def calculate_model_scores(data: List[Dict]) -> List[int]:
    """
    Calculate difference scores for all entries for a single model.
    Returns a list of difference scores.
    """
    graph = build_topology_graph()
    scores = []
    
    for entry in data:
        expected_raw = entry.get('answer', '')
        model_raw = entry.get('model_answer', '')

        # Skip entries where model answer is None, empty, or "none"
        if not model_raw or model_raw.lower().strip() in ['none', 'null', '']:
            continue

        expected_norm = normalize_topology_relation(expected_raw)

        # Handle comma-separated predictions
        if ',' in model_raw:
            preds_raw = [r.strip() for r in model_raw.split(',')]
            preds = []
            for pred_raw in preds_raw:
                pred_norm = normalize_topology_relation(pred_raw)
                if pred_norm and pred_norm.lower() not in ['none', 'null', '']:
                    preds.append(pred_norm)
            
            if not preds:  # All predictions were empty/none
                continue
                
            distances = []
            for pred in preds:
                d = calculate_shortest_distance(graph, pred, expected_norm)
                if d == float('inf'):
                    d = 10  # Penalty for unparseable
                distances.append(d)
            distance = int(sum(distances) / len(distances)) if distances else 10
        else:
            pred_norm = normalize_topology_relation(model_raw)
            if not pred_norm or pred_norm.lower() in ['none', 'null', '']:
                continue
                
            d = calculate_shortest_distance(graph, pred_norm, expected_norm)
            if d == float('inf'):
                distance = 10  # Penalty for unparseable
            else:
                distance = d

        scores.append(distance)
    
    return scores


def calculate_statistics(scores: List[int]) -> Dict[str, float]:
    """Calculate statistical measures for a list of scores."""
    if not scores:
        return {
            'count': 0,
            'mean': 0.0,
            'sum': 0,
            'min': 0.0,
            'max': 0.0,
            'std': 0.0
        }
    return {
        'count': len(scores),
        'mean': statistics.mean(scores),
        'sum': sum(scores),
        'min': min(scores),
        'max': max(scores),
        'std': statistics.stdev(scores) if len(scores) > 1 else 0.0
    }


def main():
    """Main analysis function."""
    input_files = generate_input_file("topology")
    
    print("=" * 80)
    print("TOPOLOGY ANALYSIS RESULTS")
    print("=" * 80)
    print()
    
    for file_path in input_files:
        try:
            assert (os.path.isfile(file_path))
        except:
            print(f"Warning: File not found - {file_path}")
            continue
            
        model_name = extract_model_name(file_path)
        print(f"📊 MODEL: {model_name}")
        print("-" * 50)
        
        data = load_jsonl(file_path)
        scores = calculate_model_scores(data)
        
        if not scores:
            print("❌ No valid data found for this model")
            print()
            continue
            
        # Calculate statistics
        stats = calculate_statistics(scores)
        
        # Calculate score distribution
        score_counts = Counter(scores)
        
        # Display results
        print(f"📈 Topology Distance Statistics:")
        print(f"   • Mean:     {stats['mean']:.4f}")
        print(f"   • Sum:      {int(stats['sum'])}")
        print(f"   • Min:      {int(stats['min'])}")
        print(f"   • Max:      {int(stats['max'])}")
        print(f"   • Std Dev:  {stats['std']:.4f}")
        print(f"   • Count:    {int(stats['count'])}")
        print()
        
        print(f"📊 Topology Score Distribution (0=Perfect, higher=worse):")
        max_score = int(stats['max']) if stats['max'] > 0 else 4
        for score in range(max_score + 1):  # 0 to max score
            count = score_counts.get(score, 0)
            percentage = (count / len(scores)) * 100 if scores else 0
            print(f"   • Score {score}: {count:3d} ({percentage:5.1f}%)")
        print()
        
        print("=" * 50)
        print()
        
    print("✅ Analysis completed!")


if __name__ == "__main__":
    main()