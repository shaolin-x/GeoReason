import json
import argparse
import re
import statistics
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, deque, Counter
import os
import sys
import csv

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

def direction_distance(standard, model, raw_data):
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
        model = model.strip()
        model_pos = []
        if " " in model:
            # print(raw_data['model_name'])
            model = model.split(" ")
            for i in model:
                 model_pos.append(dirction_list[i.strip().lower()])
        else:
            model_pos.append(dirction_list[model.strip().lower()])
        standard_pos = dirction_list[standard.strip().lower()]
        
    except:
        print(standard,"|", model)
        return 1e10  # Return large number for invalid cases
    
    ans = 1e10
    
    for i in model_pos:
        ans1 = abs(i-standard_pos)
        ans2 = 8-ans1
        tmp = min(ans1, ans2)
        ans = min(ans, tmp)
    # print(standard, model, standard_pos, model_pos, min(ans1,ans2))
    return ans
    

def generate_input_file(cur_type):
    input_files = []
    # cur_type = "direction"
    for i in file_name_1:
        input_files.append("./tier1_results/"+cur_type+"_merged_results_"+i)
    for i in file_name_2:
        input_files.append("./tier1_results/"+"model_output/"+i + "/" + cur_type+"_queries_output.jsonl")
    for i in file_name_3:
        input_files.append("./tier1_results/"+i + "/" + cur_type+"_queries_output.jsonl")
    return input_files

def main():
    input_files = generate_input_file("direction")
    
    output_txt = "direction_analysis_results.txt"
    output_csv = "low_score_summary.csv"
    
    with open(output_txt, "w", encoding="utf-8") as f_txt, open(output_csv, "w", newline='', encoding="utf-8") as f_csv:
        # Redirect stdout to the text file
        original_stdout = sys.stdout
        sys.stdout = f_txt

        # Prepare CSV writer
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["Model", "Low Score Count (<2)", "Low Score Percentage (%)"])
        
        print("=" * 80)
        print("DIRECTION ANALYSIS RESULTS")
        print("=" * 80)
        print()
        
        for file_path in input_files:
            if not os.path.isfile(file_path):
                print(f"Warning: File not found - {file_path}")
                continue

            model_name = extract_model_name(file_path)
            print(f"📊 MODEL: {model_name}")
            print("-" * 50)

            data = load_jsonl(file_path)
            distances = []

            for item in data:
                dist = direction_distance(item["answer"], item["model_answer"], item)
                if dist != 1e10:
                    distances.append(dist)

            if not distances:
                print("❌ No valid data found for this model")
                print()
                continue

            score_counts = Counter(distances)

            list_sum = sum(distances)
            list_min = min(distances)
            list_max = max(distances)
            list_mean = statistics.mean(distances)
            list_std = statistics.stdev(distances) if len(distances) > 1 else 0
            count_total = len(distances)
            low_score_count = sum(score_counts.get(s, 0) for s in range(2))
            low_score_percentage = (low_score_count / count_total) * 100 if count_total else 0

            # Write to CSV
            csv_writer.writerow([model_name, low_score_count, f"{low_score_percentage:.2f}"])

            print(f"📈 Direction Distance Statistics:")
            print(f"   • Mean:     {list_mean:.4f}")
            print(f"   • Sum:      {list_sum}")
            print(f"   • Min:      {list_min}")
            print(f"   • Max:      {list_max}")
            print(f"   • Std Dev:  {list_std:.4f}")
            print(f"   • Count:    {count_total}")
            print(f"   • Scores <2: {low_score_count} ({low_score_percentage:.1f}%)")
            print()

            print(f"📊 Direction Score Distribution (0=Perfect, 4=Worst):")
            for score in range(5):
                count = score_counts.get(score, 0)
                percentage = (count / count_total) * 100 if count_total else 0
                print(f"   • Score {score}: {count:3d} ({percentage:5.1f}%)")
            print()
            print("=" * 50)
            print()

        # Restore stdout
        sys.stdout = original_stdout

    print(f"✅ Text output saved to: {output_txt}")
    print(f"✅ CSV summary saved to: {output_csv}")

if __name__ == "__main__":
    main()