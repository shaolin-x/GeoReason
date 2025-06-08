import anthropic
import asyncio
import json
import os
import sys # For printing errors to stderr
from google import genai
from tqdm.asyncio import tqdm
from time import sleep
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

USER_MODEL = "claude-sonnet-4-20250514"
Filenames = ["../queries/direction_queries.json", "../queries/distance_farthest_queries.json", "../queries/distance_nearest_queries.json", "../queries/topology_queries_new.json"]
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
'''Please respond in the following JSON format: {
"answer": “<answer>”, // "answer" is all the valid relationships from: disjoint, touch, within, contains, covered by, covers, equals "
explanation": "<reason>"
}
Example Prompt: Query: How is Canada spatially related to Calgary?
Example Response: { "answer": “covers, contains”, "explanation": "Calgary is inside Canada" }
Provide your answer in this format.
'''
]
async def main():
    query_dict = {}
    requests = []
    for QUERY_TYPE in [3]:
        queries = []
        
        INPUT_FILENAME = Filenames[QUERY_TYPE]
        PROMPT = Prompt[QUERY_TYPE]
        type_name = INPUT_FILENAME.split("/")[-1].split(".")[0]
        query_dict[type_name] = []
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
        client = anthropic.Anthropic()
        for i, query in enumerate(queries):
            query_id = INPUT_FILENAME.split('/')[-1].split('.')[0] + "-"+str(i)
            content = f"Query: {query['query']}\nPrompt:\n"+Prompt[QUERY_TYPE]
            # print(content)
            requests.append(Request(
                custom_id=query_id,
                params=MessageCreateParamsNonStreaming(
                    model=USER_MODEL,
                    max_tokens=8192,
                    messages=[{
                        "role": "user",
                        "content": content,
                    }]
                )
            ))
            query_dict[type_name].append(query)
            # print(requests)
    
    
    result = client.messages.batches.create(
        requests=requests
    )
    
    req_id = result.id
    print(req_id)
    # req_id = "   "
    # 3.7     
    # 4    
    # print()
    while client.messages.batches.retrieve(req_id,).processing_status != "ended":
        sleep(1)
    
    
    ans_dict = {}
    last_id = -1
    # print(query_dict)
    for result in client.messages.batches.results(
        req_id,
    ):
        exp_name = result.custom_id.split('-')[0]
        exp_id = int(result.custom_id.split('-')[-1])
        
        output_raw = result.result.message.content[0].text
        
        output_raw = output_raw.removesuffix("```")
        output_raw = output_raw.removeprefix("```json").strip().split("\n")
        json_raw = ""
        idx = 0
        for i, output in enumerate(output_raw):
            if output == '{':
                idx = i
        for i in range(idx, len(output_raw)):
            output = output_raw[i]
            if output == '{':
                json_raw += output
            elif json_raw != "":
                json_raw+=output
        json_raw = "".join(json_raw)
        # print(json_raw)
        print(json_raw)
        output_json = json.loads(json_raw)
        query_dict[exp_name][exp_id]['model_answer'] = output_json['answer']
        query_dict[exp_name][exp_id]['model_explanation'] = output_json['explanation']
        query_dict[exp_name][exp_id]['model_name'] = USER_MODEL
        
        # print(exp_name, exp_id)
        # print(output_json)
        # print()
    
    for query in query_dict:
        OUTPUT_FILENAME = USER_MODEL + "/"+ query + "_output.jsonl"
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            for response_text in query_dict[query]:
                f.write(json.dumps(response_text)+"\n")
    # model_name = USER_MODEL
    # OUTPUT_FILENAME = model_name + "/"+ INPUT_FILENAME.split('/')[-1].split('.')[0] + "_output.jsonl"
    # print(OUTPUT_FILENAME)

    # with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
    #     for response_text in api_responses:
    #         f.write(json.dumps(response_text)+"\n")
    # print(query_dict)
        
    
        # print(queries)
        
        # semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        # tasks = [get_gemini_response(client, q_text, PROMPT, semaphore) for q_text in queries]
        # print(f"Processing {len(tasks)} queries using model '{USER_MODEL_NAME}' (Concurrency: {CONCURRENCY_LIMIT})...")
        # api_responses = await tqdm.gather(*tasks, desc="Calling Gemini API")
            
        # model_name = USER_MODEL_NAME.split("/")[-1]
        # OUTPUT_FILENAME = model_name + "/"+ INPUT_FILENAME.split('/')[-1].split('.')[0] + "_output.jsonl"
        # print(OUTPUT_FILENAME)

        # with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        #     for response_text in api_responses:
        #         f.write(json.dumps(response_text)+"\n")
    #         f.write(response_text + '\n') # Write each response on a new line
    # print(f"Successfully saved {len(api_responses)} responses to '{OUTPUT_FILENAME}'.")

if __name__ == "__main__":
    asyncio.run(main())
    