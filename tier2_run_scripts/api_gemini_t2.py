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
import re
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

Filenames = ["../queries/direction_and_distance_queries.json", "../queries/topology_and_direction_queries.json", "../queries/topology_and_distance_queries.json"]
Prompt = [
'''Please respond in the following JSON format: {
"answer": “<answer>”, // "answer" is full name name of an officially designated national park "explanation": "<reason>"
}
Example Prompt: Query: Which officially designated national park is nearest and south of Yellowstone National Park?
Example Response: { "answer": “Grand Teton National Park”, "explanation": "Grand Teton National Park is the national park that is south to Yellowstone national park and is the closest." }
Provide your answer in this format.
''', 
'''Please respond in the following JSON format: {
"answer": “<answer>”, // "answer" is the full name of all the officially designated national parks that are answers to the query, separated by comma "explanation": "<reason>"
}
Example Prompt: Query: Which national parks are south of Rocky Mountain National Park and topologically disjoint from it, within the state of Colorado?
Example Response: { "answer": “Black Canyon of the Gunnison National Park, Great Sand Dunes National Park, Mesa Verde National Park””, "explanation": "These parks are all disjoint from and south of Rocky Mountain National Park" }
Provide your answer in this format.
''',
'''
Please respond in the following JSON format: {
"answer": “<answer>”, // "answer" is the name of the state "explanation": "<reason>"
}
Example Prompt: Query: Which state is north to Illinois and topologically touches?
Example Response: { "answer": “Wisconsin”, "explanation": "Wisconsin is the state that is north of Illinois and state boundaries touch." }
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
            # print(response)
            # response = await client.generate_content(contents=[formatted_query_for_api])
            output_raw = response.text
            output_raw = output_raw.removesuffix("```")
            output_raw = output_raw.removeprefix("```json")
            # print(output_raw)
            output_raw = output_raw.split("\"explanation\"")
            output_raw[1] = re.sub(r'"\s*"', ' ', output_raw[1])
            output_raw = "\"explanation\"".join(output_raw)
            # print(output_raw)
                
            output_json = json.loads(output_raw)
            query_text['model_answer'] = output_json['answer']
            query_text['model_explanation'] = output_json['explanation']
            query_text['model_name'] = USER_MODEL_NAME
            return query_text
        except Exception as e:
            print(f"API Error for query '{query_text['query'][:30]}...': {e}", file=sys.stderr)
            print(response)
            query_text['model_answer'] = output_raw
            query_text['model_explanation'] = ""
            query_text['model_name'] = USER_MODEL_NAME
            return query_text # Return an empty string for errors

async def main():
    for QUERY_TYPE in [1,2]:
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
        
        # queries = queries[0:20]
        # print(queries)
        # return
        
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        tasks = [get_gemini_response(client, q_text, PROMPT, semaphore) for q_text in queries]
        print(f"Processing {len(tasks)} queries using model '{USER_MODEL_NAME}' (Concurrency: {CONCURRENCY_LIMIT})...")
        api_responses = await tqdm.gather(*tasks, desc="Calling Gemini API")
        # print()
            
        model_name = USER_MODEL_NAME.split("/")[-1]
        OUTPUT_FILENAME = model_name + "/"+ INPUT_FILENAME.split('/')[-1].split('.')[0] + "_output.jsonl"
        print(OUTPUT_FILENAME)

        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            for response_text in api_responses:
                f.write(json.dumps(response_text)+"\n")
    #         f.write(response_text + '\n') # Write each response on a new line
    # print(f"Successfully saved {len(api_responses)} responses to '{OUTPUT_FILENAME}'.")

if __name__ == "__main__":
    asyncio.run(main())