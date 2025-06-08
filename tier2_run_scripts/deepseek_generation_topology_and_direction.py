import os
import glob

from openai import OpenAI
# Two API key configurations
#        
#         
client_gpt_1 = OpenAI(   "  ", base_url="https://api.deepseek.com")
client_gpt_2 = OpenAI(   "   ", base_url="https://api.deepseek.com")
# Two model names
model_name_1 = "deepseek-chat"
model_name_2 = "deepseek-reasoner"

# Input and output directory configuration
input_dir = "prompts_topology_and_direction/"
output_dir_1 = "answers_topology_and_direction_deepseekchat"
output_dir_2 = "answers_topology_and_direction_deepseekreasoner"

# Create output directories
for output_dir in [output_dir_1, output_dir_2]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Get all txt files
txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
txt_files.sort()  # Sort to ensure consistent processing order

print(f"Found {len(txt_files)} txt files to process")

# Process each file
for txt_file in txt_files:
    # Get file name (without path and extension)
    file_name = os.path.basename(txt_file)
    base_name = os.path.splitext(file_name)[0]
    
    print(f"Processing: {file_name}")
    
    # Read the question
    with open(txt_file, "r") as file:
        question = file.read()
    
    messages = [{"role": "user", "content": question}]
    
    # Process with first model
    try:
        response_1 = client_gpt_2.chat.completions.create(
            model=model_name_1,
            messages=messages
        )
        action_sequence_1 = response_1.choices[0].message.content
        
        # Save first model's result
        output_file_1 = os.path.join(output_dir_1, f"{base_name.replace('query', 'answer')}.txt")
        with open(output_file_1, "w") as file:
            file.write(action_sequence_1)
        
        print(f"  {model_name_1} completed")
    except Exception as e:
        print(f"  {model_name_1} failed: {e}")
    
    # Process with second model
    try:
        response_2 = client_gpt_2.chat.completions.create(
            model=model_name_2,
            messages=messages
        )
        action_sequence_2 = response_2.choices[0].message.content
        
        # Save second model's result
        output_file_2 = os.path.join(output_dir_2, f"{base_name.replace('query', 'answer')}.txt")
        with open(output_file_2, "w") as file:
            file.write(action_sequence_2)
        
        print(f"  {model_name_2} completed")
    except Exception as e:
        print(f"  {model_name_2} failed: {e}")

print("All files processed!")

