#!/usr/bin/env python3
"""
Distance Results Difference Score Analysis
This script calculates distance difference scores by comparing the actual distances
from queried parks to model answers vs expected answers.

For each query:
1. Calculate distance from queried park to model answer park (d1)
2. Calculate distance from queried park to expected answer park (d2)  
3. Difference score = |d1 - d2|

Then compute statistics (mean, sum, min, max, std) for each model.
"""

import json
import argparse
import math
import statistics
import re
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

# National Park coordinates (latitude, longitude) from official dataset
PARK_COORDINATES = {
    "Acadia National Park": (44.35, -68.21),
    "National Park of American Samoa": (-14.25, -170.68),
    "Arches National Park": (38.68, -109.57),
    "Badlands National Park": (43.75, -102.5),
    "Big Bend National Park": (29.25, -103.25),
    "Biscayne National Park": (25.65, -80.08),
    "Black Canyon of the Gunnison National Park": (38.57, -107.72),
    "Bryce Canyon National Park": (37.57, -112.18),
    "Canyonlands National Park": (38.2, -109.93),
    "Capitol Reef National Park": (38.2, -111.17),
    "Carlsbad Caverns National Park": (32.17, -104.44),
    "Channel Islands National Park": (34.01, -119.42),
    "Congaree National Park": (33.78, -80.78),
    "Crater Lake National Park": (42.94, -122.1),
    "Cuyahoga Valley National Park": (41.24, -81.55),
    "Death Valley National Park": (36.24, -116.82),
    "Denali National Park": (63.33, -150.5),
    "Dry Tortugas National Park": (24.63, -82.87),
    "Everglades National Park": (25.32, -80.93),
    "Gates of the Arctic National Park": (67.78, -153.3),
    "Gateway Arch National Park": (38.63, -90.19),
    "Glacier National Park": (48.8, -114.0),
    "Glacier Bay National Park": (58.5, -137.0),
    "Grand Canyon National Park": (36.06, -112.14),
    "Grand Teton National Park": (43.73, -110.8),
    "Great Basin National Park": (38.98, -114.3),
    "Great Sand Dunes National Park": (37.73, -105.51),
    "Great Smoky Mountains National Park": (35.68, -83.53),
    "Guadalupe Mountains National Park": (31.92, -104.87),
    "Haleakala National Park": (20.72, -156.17),
    "Hawaii Volcanoes National Park": (19.38, -155.2),
    "Hawai'i Volcanoes National Park": (19.38, -155.2),  # Alternative spelling
    "Hot Springs National Park": (34.51, -93.05),
    "Indiana Dunes National Park": (41.6533, -87.0524),
    "Isle Royale National Park": (48.1, -88.55),
    "Joshua Tree National Park": (33.79, -115.9),
    "Katmai National Park": (58.5, -155.0),
    "Kenai Fjords National Park": (59.92, -149.65),
    "Kings Canyon National Park": (36.8, -118.55),
    "Kobuk Valley National Park": (67.55, -159.28),
    "Lake Clark National Park": (60.97, -153.42),
    "Lassen Volcanic National Park": (40.49, -121.51),
    "Mammoth Cave National Park": (37.18, -86.1),
    "Mesa Verde National Park": (37.18, -108.49),
    "Mount Rainier National Park": (46.85, -121.75),
    "New River Gorge National Park": (38.07, -81.08),
    "North Cascades National Park": (48.7, -121.2),
    "Olympic National Park": (47.97, -123.5),
    "Petrified Forest National Park": (35.07, -109.78),
    "Pinnacles National Park": (36.48, -121.16),
    "Redwood National Park": (41.3, -124.0),
    "Rocky Mountain National Park": (40.4, -105.58),
    "Saguaro National Park": (32.25, -110.5),
    "Sequoia National Park": (36.43, -118.68),
    "Shenandoah National Park": (38.53, -78.35),
    "Theodore Roosevelt National Park": (46.97, -103.45),
    "Virgin Islands National Park": (18.33, -64.73),
    "Voyageurs National Park": (48.5, -92.88),
    "White Sands National Park": (32.78, -106.17),
    "Wind Cave National Park": (43.57, -103.48),
    "Wrangell–St. Elias National Park": (61.0, -142.0),
    "Yellowstone National Park": (44.6, -110.5),
    "Yosemite National Park": (37.83, -119.5),
    "Zion National Park": (37.3, -113.05),
}

# International parks that some models might mention (we skip these entries)
INTERNATIONAL_PARKS = {
    # Canadian Parks
    "Pukaskwa National Park",
    "Waterton Lakes National Park",
    "Kluane National Park and Reserve",
    "Roosevelt Campobello International Park",
    "Fundy National Park",
    # South African Parks
    "Cape Agulhas National Park",
    "Table Mountain National Park",
    # Mauritius Parks
    "Black River Gorges National Park",
    # US National Monuments, Preserves, and Recreation Areas (not National Parks)
    "Buck Island Reef National Monument",
    "Buffalo National River",
    "Curecanti National Recreation Area",
    "Mojave National Preserve",
    "Oregon Caves National Monument and Preserve",
    "Organ Pipe Cactus National Monument",
    "Ross Lake National Recreation Area",
    "Santa Monica Mountains National Recreation Area",
    # US National Historic Sites and other non-National Parks
    "San Juan National Historic Site",
    # Incorrect/Non-existent park names mentioned by models
    "White Mountain National Park",
}

# File name configurations
file_name_1 = ["deepseekchat.jsonl", "deepseekreasoner.jsonl", "gpt-4.1.jsonl", "o3-mini.jsonl"]
file_name_2 = ["gemini-2.5-pro-preview-05-06","gemini-2.5-flash-preview-05-20","claude-sonnet-4-20250514"]
file_name_3 = ["Llama-3.3-70B-Instruct-Turbo-Free"]

def extract_model_name(file_path):
    """Extract model name from file path for clearer output"""
    # Handle distance files with farthest/nearest suffixes
    if "distance_farthest" in file_path or "distance_nearest" in file_path:
        distance_type = "Farthest" if "farthest" in file_path else "Nearest"
        
        if "deepseekchat.jsonl" in file_path:
            return f"DeepSeek-Chat ({distance_type})"
        elif "deepseekreasoner.jsonl" in file_path:
            return f"DeepSeek-Reasoner ({distance_type})"
        elif "gpt-4.1.jsonl" in file_path:
            return f"GPT-4.1 ({distance_type})"
        elif "o3-mini.jsonl" in file_path:
            return f"O3-Mini ({distance_type})"
        elif "gemini-2.5-pro-preview-05-06" in file_path:
            return f"Gemini-2.5-Pro-Preview ({distance_type})"
        elif "gemini-2.5-flash-preview-05-20" in file_path:
            return f"Gemini-2.5-Flash-Preview ({distance_type})"
        elif "claude-sonnet-4-20250514" in file_path:
            return f"Claude-Sonnet-4 ({distance_type})"
        elif "Llama-3.3-70B-Instruct-Turbo-Free" in file_path:
            return f"Llama-3.3-70B-Instruct ({distance_type})"
    else:
        # Regular model name extraction for non-distance files
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
    
    # Extract filename as fallback
    return os.path.basename(file_path).replace('.jsonl', '').replace('_queries_output', '')

def generate_input_file(cur_type):
    """Generate list of input files for analysis"""
    input_files = []
    
    # For distance analysis, we have both farthest and nearest files
    if cur_type == "distance":
        # Add farthest distance files
        for i in file_name_1:
            input_files.append("./tier1_results/distance_farthest_merged_results_"+i)
            input_files.append("./tier1_results/distance_nearest_merged_results_"+i)
        for i in file_name_2:
            input_files.append("./tier1_results/"+"model_output/"+i + "/distance_farthest_queries_output.jsonl")
            input_files.append("./tier1_results/"+"model_output/"+i + "/distance_nearest_queries_output.jsonl")
        for i in file_name_3:
            input_files.append("./tier1_results/"+i + "/distance_farthest_queries_output.jsonl")
            input_files.append("./tier1_results/"+i + "/distance_nearest_queries_output.jsonl")
    else:
        # For other types (direction, topology, etc.)
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
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error loading file: {e}")
        return []
    
    return data

def normalize_park_name(name: str) -> str:
    """Normalize park name for coordinate lookup."""
    if not name or name.lower() in ['none', 'null', '']:
        return None
        
    name = name.strip()
    
    # Handle common variations for Hawai'i / Hawaii with different characters
    if ("Hawaii" in name or "Hawai'i" in name or "Hawaiʻi" or "Hawai'i" in name) and "Volcanoes" in name:
        return "Hawaii Volcanoes National Park"
    
    # Handle Haleakala variations with special characters
    if ("Haleakala" in name or "Haleakalā" in name) and "National Park" in name:
        return "Haleakala National Park"
    
    # Handle American Samoa variations
    if "American Samoa" in name and "National Park" in name:
        return "National Park of American Samoa"
    
    # Handle Wrangell-St. Elias variations (with different dash types and preserve suffix)
    if "Wrangell" in name and "St. Elias" in name and "National Park" in name:
        return "Wrangell–St. Elias National Park"  # Use the em dash version
    
    # Handle Redwood variations
    if "Redwood" in name and ("National and State Parks" in name or "National Park" in name):
        return "Redwood National Park"
    
    # Remove "and Preserve" suffix for parks that have both designations
    if "National Park and Preserve" in name:
        name = name.replace("and Preserve", "").strip()
    
    # Add "National Park" if not present (and not a preserve)
    if not name.endswith("National Park") and "National Park" not in name:
        name += " National Park"
    
    return name

def get_park_coordinates(park_name: str) -> Optional[Tuple[float, float]]:
    """Get coordinates for a park name."""
    if not park_name:
        return None
        
    normalized_name = normalize_park_name(park_name)
    if not normalized_name:
        return None
        
    # Direct lookup
    if normalized_name in PARK_COORDINATES:
        return PARK_COORDINATES[normalized_name]
    
    # Fuzzy matching for common variations
    for coord_park, coords in PARK_COORDINATES.items():
        if park_name.lower() in coord_park.lower() or coord_park.lower() in park_name.lower():
            return coords
    
    return None

def calc_park_to_park_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance given lats and longs of two parks using Haversine formula."""
    R = 6371.0  # Radius of Earth in kilometers

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def extract_queried_park_from_query(query: str) -> str:
    """Extract the park name being queried from the question."""
    # Look for patterns like "farthest from [PARK NAME] in straight"
    pattern = r"farthest from ([^?]+?) in straight"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Look for patterns like "closest to [PARK NAME] in straight"
    pattern = r"closest to ([^?]+?) in straight"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for "from [PARK NAME]" or "to [PARK NAME]"
    pattern = r"(?:from|to) ([^?]+?)(?:\s+in|\s*\?)"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None

def is_international_park(park_name: str) -> bool:
    """Check if a park name refers to an international (non-US) national park."""
    if not park_name:
        return False
    
    # Check exact match first
    if park_name in INTERNATIONAL_PARKS:
        return True
    
    # Check normalized name
    normalized_name = normalize_park_name(park_name)
    return normalized_name in INTERNATIONAL_PARKS

def validate_park_names(data: List[Dict]) -> None:
    """
    Validate that all park names in the dataset have coordinates available.
    Canadian parks are identified and will be skipped in analysis.
    """
    missing_parks = set()
    all_original_parks = set()  # Original names for reference
    all_normalized_parks = set()  # Normalized names for accurate counting
    international_parks_found = set()
    unmatched_us_parks = set()  # Parks that aren't international but also don't have coordinates
    
    for i, entry in enumerate(data):
        query = entry.get('query', '')
        expected = entry.get('answer', '')
        model_answer = entry.get('model_answer', '')
        
        # Extract queried park from the question
        queried_park = extract_queried_park_from_query(query)
        
        if queried_park:
            all_original_parks.add(queried_park)
            if is_international_park(queried_park):
                international_parks_found.add(queried_park)
            elif not get_park_coordinates(queried_park):
                unmatched_us_parks.add(queried_park)
                missing_parks.add(queried_park)
            else:
                # Add normalized name for US parks
                normalized = normalize_park_name(queried_park)
                all_normalized_parks.add(normalized)
        
        if expected:
            all_original_parks.add(expected)
            if is_international_park(expected):
                international_parks_found.add(expected)
            elif not get_park_coordinates(expected):
                unmatched_us_parks.add(expected)
                missing_parks.add(expected)
            else:
                # Add normalized name for US parks
                normalized = normalize_park_name(expected)
                all_normalized_parks.add(normalized)
        
        if model_answer:
            all_original_parks.add(model_answer)
            if is_international_park(model_answer):
                international_parks_found.add(model_answer)
            elif not get_park_coordinates(model_answer):
                unmatched_us_parks.add(model_answer)
                missing_parks.add(model_answer)
            else:
                # Add normalized name for US parks
                normalized = normalize_park_name(model_answer)
                all_normalized_parks.add(normalized)
    
    if international_parks_found:
        print("⚠️  NOTE: International parks found in dataset (entries will be SKIPPED in analysis):")
        print("=" * 65)
        for park in sorted(international_parks_found):
            print(f"  - {park}")
        print("These entries will be excluded from distance calculation.\n")
    
    if unmatched_us_parks:
        print("🔍 DEBUG: Unmatched US park names (possible spelling variants):")
        print("=" * 55)
        for park in sorted(unmatched_us_parks):
            normalized = normalize_park_name(park)
            print(f"  - Original: '{park}'")
            print(f"    Normalized: '{normalized}'")
            print(f"    In coordinates: {normalized in PARK_COORDINATES}")
            print()
    
    if missing_parks:
        print("ERROR: The following national parks are not in the coordinates database:")
        print("=" * 60)
        for park in sorted(missing_parks):
            print(f"  - {park}")
        print(f"\nTotal missing parks: {len(missing_parks)}")
        print(f"Total unique parks in dataset: {len(all_original_parks)}")
        print("\nPlease add coordinates for these parks to PARK_COORDINATES dictionary.")
        exit(1)
    
    print(f"✓ All {len(all_normalized_parks)} unique US National Parks in dataset have coordinates available.")
    print(f"  (Found {len(all_original_parks) - len(international_parks_found)} original park name variants)")
    
    # Debug: Show all US parks found
    us_parks_in_data = set()
    for park in all_original_parks:
        if not is_international_park(park):
            us_parks_in_data.add(park)
    
    print(f"\n🔍 DEBUG: All {len(us_parks_in_data)} US park name variants found:")
    normalized_to_originals = {}
    for park in sorted(us_parks_in_data):
        normalized = normalize_park_name(park)
        if normalized not in normalized_to_originals:
            normalized_to_originals[normalized] = []
        normalized_to_originals[normalized].append(park)
    
    for normalized_name in sorted(normalized_to_originals.keys()):
        variants = normalized_to_originals[normalized_name]
        if len(variants) > 1:
            print(f"  ✓ '{normalized_name}' ← {len(variants)} variants:")
            for variant in variants:
                print(f"    - '{variant}'")
        else:
            print(f"  ✓ '{normalized_name}'")

    print(f"\nSummary: {len(all_normalized_parks)} unique National Parks with {len(us_parks_in_data)} total name variants")

def calculate_difference_scores(data: List[Dict]) -> List[float]:
    """
    Calculate difference scores for all entries.
    Skip entries that involve international parks.
    Returns a list of difference scores.
    """
    scores = []
    skipped_entries = []
    international_skipped = []
    
    for i, entry in enumerate(data):
        query = entry.get('query', '')
        expected = entry.get('answer', '')
        model_answer = entry.get('model_answer', '')
        
        # Extract queried park from the question
        queried_park = extract_queried_park_from_query(query)
        
        if not queried_park:
            skipped_entries.append(i + 1)
            continue
        
        # Skip entries involving international parks
        if (is_international_park(queried_park) or 
            is_international_park(expected) or 
            is_international_park(model_answer)):
            international_skipped.append({
                'entry': i + 1,
                'query': queried_park,
                'expected': expected,
                'model_answer': model_answer
            })
            continue
        
        # Get coordinates for all three parks
        queried_coords = get_park_coordinates(queried_park)
        expected_coords = get_park_coordinates(expected)
        model_coords = get_park_coordinates(model_answer)
        
        if not queried_coords or not expected_coords or not model_coords:
            skipped_entries.append(i + 1)
            continue
        
        # Calculate distances
        d1 = calc_park_to_park_distance(queried_coords[0], queried_coords[1], 
                                       model_coords[0], model_coords[1])
        d2 = calc_park_to_park_distance(queried_coords[0], queried_coords[1],
                                       expected_coords[0], expected_coords[1])
        
        # Calculate difference score (absolute difference)
        difference_score = abs(d1 - d2)
        scores.append(difference_score)
    
    return scores

def calculate_statistics(scores: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of scores."""
    if not scores:
        return {"mean": 0, "sum": 0, "min": 0, "max": 0, "std": 0}
    
    return {
        "mean": statistics.mean(scores),
        "sum": sum(scores),
        "min": min(scores),
        "max": max(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0
    }

def main():
    """Main analysis function."""
    input_files = generate_input_file("distance")
    
    print("=" * 80)
    print("DISTANCE DIFFERENCE SCORE ANALYSIS RESULTS")
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
        
        print(f"Loading data from: {file_path}")
        data = load_jsonl(file_path)
        
        if not data:
            print("❌ No data loaded for this model")
            print()
            continue
        
        # Calculate difference scores
        scores = calculate_difference_scores(data)
        
        if not scores:
            print("❌ No valid scores calculated for this model")
            print()
            continue
        
        # Calculate and display statistics
        stats = calculate_statistics(scores)
        
        print(f"📈 Distance Difference Statistics (km):")
        print(f"   • Mean:     {stats['mean']:.4f}")
        print(f"   • Sum:      {stats['sum']:.2f}")
        print(f"   • Min:      {stats['min']:.4f}")
        print(f"   • Max:      {stats['max']:.4f}")
        print(f"   • Std Dev:  {stats['std']:.4f}")
        print(f"   • Count:    {len(scores)}")
        print()
        
        # Create 13 equal-width buckets for distance distribution
        if stats['max'] > stats['min']:  # Avoid division by zero
            bucket_width = (stats['max'] - stats['min']) / 13
            bucket_counts = [0] * 13
            
            for score in scores:
                # Calculate which bucket this score falls into
                bucket_index = int((score - stats['min']) / bucket_width)
                # Handle edge case where score equals max
                if bucket_index >= 13:
                    bucket_index = 12
                bucket_counts[bucket_index] += 1
            
            print(f"📊 Distance Difference Distribution (13 buckets, width: {bucket_width:.2f} km):")
            for i in range(13):
                bucket_start = stats['min'] + i * bucket_width
                bucket_end = stats['min'] + (i + 1) * bucket_width
                count = bucket_counts[i]
                percentage = (count / len(scores)) * 100 if scores else 0
                print(f"   • Bucket {i+1:2d} [{bucket_start:6.2f} - {bucket_end:6.2f}): {count:3d} ({percentage:5.1f}%)")
        else:
            print(f"📊 Distance Distribution: All scores are the same ({stats['min']:.2f} km)")
        
        print("=" * 50)
        print()
    
    print("✅ Analysis completed!")

if __name__ == "__main__":
    main()
