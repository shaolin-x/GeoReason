# from google import genai
# client = genai.Client(   " ")
# response = client.models.generate_content(
#     model="models/gemini-2.5-flash-preview-05-20",
#     contents=["How does AI work?"]
# )
# print(response.text)
import asyncio
import json
import os
import sys # For printing errors to stderr
from google import genai
from tqdm.asyncio import tqdm

# --- Configuration ---
   = " " # From your request
# USER_MODEL_NAME = "models/gemini-2.5-flash-preview-05-20"      # From your request
USER_MODEL_NAME = "models/gemini-2.5-pro-preview-05-06"
OUTPUT_FILENAME = "gemini_responses.txt" # Plain text output
CONCURRENCY_LIMIT = 15
QUERY_TYPE = 0

Filenames = ["../queries/direction_queries.json", "../queries/distance_farthest_queries.json", "../queries/distance_nearest_queries.json", "../queries/topology_queries_new.json"]
# Filenames = [ "../queries/topology_queries.json"]

Prompt = [
'''Please respond in the following JSON format: {
"answer": “<answer>”, // "answer" is one of: North, East, West, South, Southwest, Southeast, Northwest, Northeast "explanation": "<reason>"
}
Example Prompt: Query: What's the closest 8-point compass direction from Mexico to Canada?
Example Response: { "answer": North, "explanation": "Mexico is south of Canada" }
Provide your answer in this format.
''', 
'''Please respond in the following JSON format: {
"answer": “<answer>”, // "answer" is full name name of an officially designated national park "explanation": "<reason>"
}
Example Prompt: Query: Which officially designated national park is in Maine?
Example Response: { "answer": “Acadia National Park”, "explanation": "Acadia National Park is in Maine" }
Provide your answer in this format.
''',
'''
Please respond in the following JSON format: {
"answer": “<answer>”, // "answer" is full name name of an officially designated national park "explanation": "<reason>"
}
Example Prompt: Query: Which officially designated national park is in Maine?
Example Response: { "answer": “Acadia National Park”, "explanation": "Acadia National Park is in Maine" }
Provide your answer in this format.
''',

# '''Please respond in the following JSON format:
# {
# "answer": "<answer>", //"answer" is all the valid relationships from: disjoint, touch, overlap, within, contains, covered by, covers, equals.
#             //explanations to the terms are:
#             // disjoint – two regions are disjoint if they share no common points;
#             // touch – two regions touch if they touch at their boundaries, with no interior overlaps;
#             // overlap – two regions overlap if they share some interior area but neither is entirely contained within the other;
#             // within – region A is within region B if it lies completely inside region B without touching region B's boundary;
#             // contains – region A contains region B if it region B lies completely inside region A without touching region A's boundary;
#             // covered by – region A is covered by region B if it lies entirely within region B while touching region B's boundary;
#             // covers – region A covers region B if region B lies entirely inside region A while touching region A's boundary
#             // equals – the two regions equals if they coincide exactly.
# "explanation": "<reason>"
# }
# Example Prompt: Query: How is Canada spatially related to Calgary?
# Example Response: { "answer": "covers, contains", "explanation": "Calgary is inside Canada" }
# Provide your answer in this format.
# '''
'''Please respond in the following JSON format:
{
"answer": "<answer>", //"answer" is all the valid relationships from: disjoint, touch, overlap, within, contains, covered by, covers, equals.
"explanation": "<reason>"
}
Example Prompt 1: Query: How is Canada spatially related to Calgary?
Example Response 1: { "answer": "covers, contains", "explanation": "Calgary is inside Canada" }
Example Prompt 2: Query: How is Chattanooga city in Tennessee State related to Eastern Time Zone ?
Example Response 2: { "answer": “overlap”, “explanation": "Most of the Chattanooga city is in the Eastern Time Zone, however, some surrounding areas (e.g., northwestern suburbs) may follow Central Time” }
Provide your answer in this format.
'''
]
from pydantic import BaseModel
class Answer(BaseModel):
    answer: str
    explanation: str
async def get_gemini_response(client, query_text, prompt, semaphore):
    async with semaphore:
        # As per your earlier instruction, queries are formatted with "Query: "
        formatted_query_for_api = f"Query: {query_text['query']}\nPrompt:\n"+prompt
        # print(formatted_query_for_api)
        try:
            response = await client.aio.models.generate_content(model=USER_MODEL_NAME,contents=formatted_query_for_api,config={"response_mime_type": "application/json","response_schema": Answer})
            # response = await client.generate_content(contents=[formatted_query_for_api])
            output_raw = response.text
            output_raw = output_raw.removesuffix("```")
            output_raw = output_raw.removeprefix("```json")
            
            output_json = json.loads(output_raw)
            query_text['model_answer'] = output_json['answer']
            query_text['model_explanation'] = output_json['explanation']
            query_text['model_name'] = USER_MODEL_NAME
            return query_text
        except Exception as e:
            print(f"API Error for query '{query_text['query'][:30]}...': {e}", file=sys.stderr)
            return "" # Return an empty string for errors

async def main():
    for QUERY_TYPE in [3]:
        client = genai.Client(     )
        queries = []
        INPUT_FILENAME = Filenames[QUERY_TYPE]
        PROMPT = Prompt[QUERY_TYPE]
        try:
            with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
                lines = ""
                for i, line in enumerate(f):
                    line_content = line.strip()
                    lines += line_content
                    if line_content!='}':
                        continue
                    # print(lines)
                    data = json.loads(lines)
                    if "query" in data:
                        queries.append(data)
                    lines = ""
            print(f"Loaded {len(queries)} queries from '{INPUT_FILENAME}'.")
        except Exception as e:
            print(f"Error reading input file '{INPUT_FILENAME}': {e}", file=sys.stderr)
            return
        
        # queries = queries[0:3]
        
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        tasks = [get_gemini_response(client, q_text, PROMPT, semaphore) for q_text in queries]
        print(f"Processing {len(tasks)} queries using model '{USER_MODEL_NAME}' (Concurrency: {CONCURRENCY_LIMIT})...")
        api_responses = await tqdm.gather(*tasks, desc="Calling Gemini API")
            
        model_name = USER_MODEL_NAME.split("/")[-1]
        OUTPUT_FILENAME = model_name + "/"+ INPUT_FILENAME.split('/')[-1].split('.')[0] + "_output3.jsonl"
        print(OUTPUT_FILENAME)

        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            for response_text in api_responses:
                f.write(json.dumps(response_text)+"\n")
    #         f.write(response_text + '\n') # Write each response on a new line
    # print(f"Successfully saved {len(api_responses)} responses to '{OUTPUT_FILENAME}'.")

if __name__ == "__main__":
    asyncio.run(main())