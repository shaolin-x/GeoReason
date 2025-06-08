import json
import argparse
import re
import statistics
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, deque
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

def calc_topology(state_name, timezone_name, direction):
    state = states_df[states_df['name'] == state_name].iloc[0]
    timezone = time_zones_df[time_zones_df['zone'] == timezone_name].iloc[0]
    upper = 0.98
    lower = 0.02
    geom_state = state.geometry
    geom_other = timezone.geometry
    inter = geom_state.intersection(geom_other)
    ia = inter.area
    area_state = geom_state.area
    upper_area = upper * area_state
    lower_area = lower * area_state

    rels = []
    if ia == 0:
        if state['name'] == "Utah" and timezone.zone == "Pacific" :
            rels.append("touch")
        elif state['name'] == "Montana" and timezone.zone == "Central":
            rels.append("touch")
        else: rels.append("disjoint")
    if 0 < ia <= lower_area:
        rels.append("touch")
    if lower_area <ia <= upper_area:
        rels.append("overlaps")
    if ia >= upper_area:
        special = {"Iowa", "Missouri", "Arkansas", "West Virginia"}
        if state['name'] in special:
            if(direction == "state"):
                rels.append("within, covered by")
            else:   
                rels.append("contains, covers")
        elif state['name'] == "Alaska":
            if(direction == "zone"):
                rels.append("within, covered by")
            else:   
                rels.append("contains, covers")
        else:
            if(direction == "state"):
                rels.append("covered by")
            else:   
                rels.append("covers")
    
    return sorted(set(rels))

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
topology_list = ["touch", "disjoint", "overlaps", "within", "covered by", "contains", "covers"]
def main():
    
    init_shaolin_code()
    input_files =  generate_input_file("topology_and_direction")
    # print(parks_df)
    
    for file_path in input_files:
        try:
            assert (os.path.isfile(file_path))
        except:
            print(file_path)
        data = load_jsonl(file_path)
        
        topology_correct_p  = []
        directions_correction_p = []
        
        
        
        for item in data:
            # park_answer = []
            # if item["answer"] not in ["None"]:
            #     for i in item["answer"]:
            #         park_answer.append(find_park(i))
            # print(park1)
            park_model = []
            if item["model_answer"] not in ["None"] and item["model_answer"] is not None:
                for i in item["model_answer"].split(","):
                    park = find_park(i)
                    if park is not None:
                        park_model.append(park)
            
            # print(item["model_answer"])
            # print(park_model)
            if park_model == None or len(park_model) == 0 or (len(park_model) ==1 and park_model[0] == None):
                continue
            # print(item["model_answer"])
            # print(park_model)
            
            given_time_zone = None
            given_park = None
            given_topology = None
            given_direction = None
            
            for i in parks_df['name']:
                if i in item['query']:
                    given_park = i
                    break
            for i in time_zones_df['zone']:
                if i in item['query']:
                    given_time_zone = i
                    break
            for i in topology_list:
                if i in item['query']:
                    given_topology = i
                    break
            for i in direction_lists:
                if i in item['query'].lower():
                    given_direction = i
                    break
            # print(given_topology)
            assert(given_time_zone is not None)
            assert(given_park is not None)
            assert(given_topology is not None)
            assert(given_direction is not None)
            # continue
            
            topology_correct = 0
            direction_correct = 0
            # print(states_df['name'])
            for i in park_model:
                try:
                    park_data = parks_df.loc[parks_df['name'] == i, 'states'].iat[0]
                except:
                    print(i)
                park_state = park_data.split(", ")
                for state in park_state:
                    # print(given_topology, calc_topology(state, given_time_zone, "state"))
                    try:
                        if given_topology in calc_topology(state, given_time_zone, "state"):
                            topology_correct +=1
                            break
                    except:
                        # print(state)
                        pass
                        
                str1 = calc_direction(given_park, i).lower().strip()
                str2 = given_direction.lower().strip()
                if str1==str2:
                    direction_correct +=1
            topology_correct_p.append(topology_correct/len(park_model))
            directions_correction_p.append(direction_correct/len(park_model))
            # print(topology_correct, len(park_model))
            # print(direction_correct, len(park_model))
            
                # print(park_state)
            # print(ground_truth, model_direction)
            
        #     # break
        # # print(data)
        measured = directions_correction_p
        list_sum = sum(measured)
        list_min = min(measured)
        list_max = max(measured)
        list_mean = statistics.mean(measured)
        list_std = statistics.stdev(measured)
        
        print(f"{list_mean}, {list_sum}, {list_min}, {list_max}, {list_std}, {len(measured)}")
        # break
        
    # print(input_files)
if __name__ == "__main__":
    main()