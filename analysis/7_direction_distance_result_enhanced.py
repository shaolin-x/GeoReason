import json
import argparse
import re
import statistics
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, deque, Counter
import os
import math
import pandas as pd
import geopandas as gpd
import difflib
from thefuzz import process, fuzz

import math

def calculate_bearing(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    d_lon = math.radians(lon2 - lon1)

    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)

    bearing_rad = math.atan2(x, y)
    bearing_deg = (math.degrees(bearing_rad) + 360) % 360
    return bearing_deg

def bearing_to_direction(bearing):
    """
    Convert bearing in degrees to 8-point compass direction.
    """
    directions = [
        'North', 'Northeast', 'East', 'Southeast',
        'South', 'Southwest', 'West', 'Northwest'
    ]
    idx = int(((bearing + 22.5) % 360) / 45)
    return directions[idx]

# input: row containing park1 and park2 names
# output: direction from park1 to park2
def calc_direction(park1, park2):
    
    row = {'park1': park1, 'park2': park2}
    lat1 = parks_df.loc[parks_df['name'] == row['park1'], 'latitude'].iat[0]
    lon1 = parks_df.loc[parks_df['name'] == row['park1'], 'longitude'].iat[0]
    lat2 = parks_df.loc[parks_df['name'] == row['park2'], 'latitude'].iat[0]
    lon2 = parks_df.loc[parks_df['name'] == row['park2'], 'longitude'].iat[0]
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
    return bearing_to_direction(bearing)

def calc_park_to_park_distance(park1, park2):
    
    try:
        park1_lat = parks_df.loc[parks_df['name'] == park1, 'latitude'].iat[0]
        park1_lon = parks_df.loc[parks_df['name'] == park1, 'longitude'].iat[0]
        park2_lat = parks_df.loc[parks_df['name'] == park2, 'latitude'].iat[0]
        park2_lon = parks_df.loc[parks_df['name'] == park2, 'longitude'].iat[0]
    except:
        print(park1, "|", park2)
    R = 6371.0  # Radius of Earth in kilometers

    # Convert degrees to radians
    lat1_rad = math.radians(park1_lat)
    lon1_rad = math.radians(park1_lon)
    lat2_rad = math.radians(park2_lat)
    lon2_rad = math.radians(park2_lon)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

parks_df = None
time_zones_df = None
states_df = None

def init_shaolin_code():
    global parks_df, time_zones_df, states_df
    df = pd.read_csv('./dataset/parks_coordinates.csv')

    # Convert 'latitude' and 'longitude' columns to numeric, forcing errors to NaN if any bad data
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Drop rows with missing or invalid coordinates, if needed
    df = df.dropna(subset=['latitude', 'longitude'])

    # Convert to GeoDataFrame
    parks_df = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs='EPSG:4326'  # Assuming coordinates are in WGS84
    )
    STATES_PATH = './dataset/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
    TIMEZONES_PATH = './dataset/NTAD_Time_Zones_467650596632424595/Time_Zones.shp'

    # Exclusions
    EXCLUDED_ZONES = {"Chamorro", "Samoa", "Atlantic"}
    EXCLUDED_STATES = {
        "American Samoa", "Guam", "Commonwealth of the Northern Mariana Islands",
        "Puerto Rico", "United States Virgin Islands"}

    # Load data
    states = gpd.read_file(STATES_PATH)
    time_zones = gpd.read_file(TIMEZONES_PATH)

    # Ensure both layers use the same CRS
    states = states.to_crs(time_zones.crs)

    # Apply exclusions
    time_zones_df = time_zones[~time_zones["zone"].isin(EXCLUDED_ZONES)]
    states_df = states[~states["name"].isin(EXCLUDED_STATES)]
    
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



file_name_1 = ["deepseekchat.jsonl", "deepseekreasoner.jsonl", "gpt-4.1.jsonl", "o3-mini.jsonl"]
file_name_2 = ["gemini-2.5-pro-preview-05-06","gemini-2.5-flash-preview-05-20","claude-sonnet-4-20250514"]
file_name_3 = ["Llama-3.3-70B-Instruct-Turbo-Free"]


def direction_distance(direction1, direction2):
    dirction_list = {
        "north":0,
        "northeast":1,
        "east":2,
        "southeast":3,
        "south":4,
        "southwest":5,
        "west":6,
        "northwest":7,
    }
    try:
        direction1_pos = dirction_list[direction1.strip().lower()]
        direction2_pos = dirction_list[direction2.strip().lower()]
        
    except:
        pass
        # print(standard,"|", model)
    
    ans = 1e10
    

    ans1 = abs(direction1_pos-direction2_pos)
    ans2 = 8-ans1
    return min(ans1, ans2)
    

def generate_input_file(cur_type):
    input_files = []
    # cur_type = "direction"
    for i in file_name_1:
        input_files.append("./tier2_results/"+cur_type+"_merged_results_"+i)
    for i in file_name_2:
        input_files.append("./tier2_results/"+i + "/" + cur_type+"_queries_output.jsonl")
    for i in file_name_3:
        input_files.append("./tier2_results/"+i + "/" + cur_type+"_queries_output.jsonl")
    return input_files

def find_park(name):
    if not name or name.lower() in ['none', 'null', '']:
        return None
    if " is " in name:
        return None
    name = name.strip()
    if ("Hawaii" in name or "Hawai'i" in name or "Hawaiʻi" or "Hawai'i" in name) and "Volcanoes" in name:
        name =  "Hawai'i Volcanoes National Park"
    elif ("Haleakala" in name or "Haleakalā" in name) and "National Park" in name:
         name = "Haleakala National Park"
    
    # Handle American Samoa variations
    elif "American Samoa" in name and "National Park" in name:
        name = "National Park of American Samoa"
    
    # Handle Wrangell-St. Elias variations (with different dash types and preserve suffix)
    elif "Wrangell" in name and "St. Elias" in name and "National Park" in name:
        name = "Wrangell–St. Elias National Park"  # Use the em dash version
    
    # Handle Redwood variations
    elif "Redwood" in name and ("National and State Parks" in name or "National Park" in name):
        name = "Redwood National Park"
    
    # Remove "and Preserve" suffix for parks that have both designations
    else:
        if "National Park and Preserve" in name:
            name = name.replace("and Preserve", "").strip()
    
    # Add "National Park" if not present (and not a preserve)
        if not name.endswith("National Park") and "National Park" not in name:
            name += " National Park"
    
    result_df = parks_df.loc[parks_df['name'] == name, 'name']
    # park_names_list = parks_df['name'].unique()
    if result_df.empty:
        # closest_matches = difflib.get_close_matches(name, park_names_list, n=1, cutoff=0.6)
        # print(name,"|", closest_matches)
        return None
    
    
    # print(result_df.iat[0])
    return result_df.iat[0]

direction_lists = ["northwest", "northeast", "southeast" ,"southwest","north","east","south","west"]

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

def main():
    
    init_shaolin_code()
    input_files =  generate_input_file("direction_and_distance")
    # print(parks_df)
    
    print("=" * 80)
    print("DIRECTION AND DISTANCE ANALYSIS RESULTS")
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
        
        distances = []
        directions_distance = []
        
        
        for item in data:
            park1 = find_park(item["answer"])
            
            park2 =  find_park(item["model_answer"])
            
            if park1 == None or park2 == None:
                continue
            
            ground_truth = None
            for i in direction_lists:
                if i in item["query"].lower():
                    ground_truth = i
                    break
            
            # print(park1, " | ", park2)
            distances.append(calc_park_to_park_distance(park1, park2))
            model_direction = calc_direction(park1, park2)
            directions_distance.append(direction_distance(ground_truth, model_direction))
            # print(ground_truth, model_direction)
            
        if not directions_distance:
            print("❌ No valid data found for this model")
            print()
            continue
            
        # Calculate statistics
        measured = directions_distance
        list_sum = sum(measured)
        list_min = min(measured)
        list_max = max(measured)
        list_mean = statistics.mean(measured)
        list_std = statistics.stdev(measured) if len(measured) > 1 else 0
        
        # Calculate score distribution (0-4)
        score_counts = Counter(measured)
        
        # Display results
        print(f"📈 Direction Distance Statistics:")
        print(f"   • Mean:     {list_mean:.4f}")
        print(f"   • Sum:      {list_sum}")
        print(f"   • Min:      {list_min}")
        print(f"   • Max:      {list_max}")
        print(f"   • Std Dev:  {list_std:.4f}")
        print(f"   • Count:    {len(measured)}")
        print()
        
        print(f"📊 Score Distribution (0=Perfect, 4=Worst):")
        for score in range(5):  # 0-4
            count = score_counts.get(score, 0)
            percentage = (count / len(measured)) * 100 if measured else 0
            print(f"   • Score {score}: {count:3d} ({percentage:5.1f}%)")
        print()
        
        # Also show distance statistics
        if distances:
            dist_mean = statistics.mean(distances)
            dist_std = statistics.stdev(distances) if len(distances) > 1 else 0
            dist_min = min(distances)
            dist_max = max(distances)
            
            print(f"🗺️  Park Distance Statistics (km):")
            print(f"   • Mean:     {dist_mean:.2f}")
            print(f"   • Min:      {dist_min:.2f}")
            print(f"   • Max:      {dist_max:.2f}")
            print(f"   • Std Dev:  {dist_std:.2f}")
            
            # Create 13 equal-width buckets for distance distribution
            if dist_max > dist_min:  # Avoid division by zero
                bucket_width = (dist_max - dist_min) / 13
                bucket_counts = [0] * 13
                
                for distance in distances:
                    # Calculate which bucket this distance falls into
                    bucket_index = int((distance - dist_min) / bucket_width)
                    # Handle edge case where distance equals dist_max
                    if bucket_index >= 13:
                        bucket_index = 12
                    bucket_counts[bucket_index] += 1
                
                print(f"📊 Distance Distribution (13 buckets, width: {bucket_width:.2f} km):")
                for i in range(13):
                    bucket_start = dist_min + i * bucket_width
                    bucket_end = dist_min + (i + 1) * bucket_width
                    count = bucket_counts[i]
                    percentage = (count / len(distances)) * 100 if distances else 0
                    print(f"   • Bucket {i+1:2d} [{bucket_start:6.2f} - {bucket_end:6.2f}): {count:3d} ({percentage:5.1f}%)")
            else:
                print(f"📊 Distance Distribution: All distances are the same ({dist_min:.2f} km)")
        
        print("=" * 50)
        print()
        
    print("✅ Analysis completed!")
    # print(input_files)

if __name__ == "__main__":
    main()