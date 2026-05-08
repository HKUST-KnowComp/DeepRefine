import json
import itertools
import os
from tqdm import tqdm
import random
import argparse
import asyncio
import logging
from functools import wraps
from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path
import csv

API_KEY = ""
BASE_URL = ""

MCQ_PROMPT = """
You are an expert in generating multiple-choice questions (MCQs) from scientific texts.
Your task is to generate 5 multiple-choice questions based on the following passage.

Each question should:
- Focus on factual claims, numerical data, definitions, or relational knowledge from the passage.
- Have 4 options (one correct answer and three plausible distractors).
- Clearly indicate the correct answer.

The output should be in JSON format, with each question as a dictionary containing:
- "question": The MCQ question.
- "options": A list of 4 options (e.g., ["A: ..", "B: ..", "C: ..", "D: .."]).
- "answer": The correct answer (e.g., "A").

Output Example:
```
[
  {
    "question": "What is the primary role of a catalyst in a chemical reaction?",
    "options": [
      "A: To make a thermodynamically unfavorable reaction proceed",
      "B: To provide a lower energy pathway between reactants and products",
      "C: To decrease the rate of a chemical reaction",
      "D: To change the overall reaction itself"
    ],
    "answer": "B"
  },
  {
    "question": "By how much can catalysis speed up a chemical reaction compared to its rate without a catalyst?",
    "options": [
      "A: By a factor of several hundred times",
      "B: By a factor of several thousand times",
      "C: By a factor of several million times",
      "D: By a factor of several billion times"
    ],
    "answer": "C"
  }
]
```

Passage:
{passage}

Output:
"""

async def api_model(
    args, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=API_KEY, base_url=BASE_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    response = await openai_async_client.chat.completions.create(
        model=args.model, messages=messages, **kwargs
    )

    return response.choices[0].message.content


async def _run_api(args, queries, max_concurrent=8):
    semaphore = asyncio.Semaphore(max_concurrent)  # 限制最大并发数为5

    async def limited_api_model(query):
        async with semaphore:
            return await api_model(args, query)

    tasks = [limited_api_model(query) for query in queries]
    answers = await tqdm_asyncio.gather(*tasks, total=len(tasks))  # 使用 tqdm_asyncio.gather
    return answers


async def generate(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.logging_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    args.kg_file = args.kg_file.replace("{dataset_name}", args.dataset)
    args.output_file = args.output_file.replace("{dataset_name}", args.dataset)

    folder = Path(args.kg_file)
    for file_path in tqdm(list(folder.rglob('*'))):
        kg_data_list = []
        if file_path.is_file():
            # get kg data
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    if not (isinstance(data['entity_relation_dict'], list)\
                            and isinstance(data['event_entity_relation_dict'], list)\
                            and isinstance(data['event_relation_dict'], list)):
                        continue
                    passage = data['original_text']
                    entity_relation_ori = [x for x in data['entity_relation_dict'] if isinstance(x, dict) and 'Head' in x and 'Relation' in x and 'Tail' in x]
                    entity_relation_list = [f"{triple_dict['Head']}, {triple_dict['Relation']}, {triple_dict['Tail']}" for triple_dict in entity_relation_ori]
                    event_entity_ori = [x for x in data['event_entity_relation_dict'] if isinstance(x, dict) and 'Event' in x]
                    event_entity_list = [f"{event_dict['Event']}" for event_dict in event_entity_ori]
                    event_relation_ori = [x for x in data['event_relation_dict'] if isinstance(x, dict) and 'Head' in x and 'Relation' in x and 'Tail' in x]
                    event_relation_list = [f"{event_dict['Head']}, {event_dict['Relation']}, {event_dict['Tail']}" for event_dict in event_relation_ori]
                    data_dict = {
                        'passage': passage,
                        'entity_relation_list': entity_relation_list,
                        'event_entity_list': event_entity_list,
                        'event_relation_list': event_relation_list
                    }
                    kg_data_list.append(data_dict)
    if len(kg_data_list) > args.sample_size:
        kg_data_list = random.sample(kg_data_list, args.sample_size)
    
    # get queries for MCQ generation
    query_list = []
    for kg_data_dict in kg_data_list:
        passage = kg_data_dict['passage']
        query = MCQ_PROMPT.replace("{passage}", passage)
        query_list.append(query)
    
    # get generate results
    answers = await _run_api(args=args, queries=query_list)
    kg_data_list_with_mcqs = []
    for response, kg_data_dict in zip(answers, kg_data_list):
        try:
            decoded_response = json.loads(response.split('```')[1])
        except:
            print(f"ERROR DECODING: {response}")
            continue
        kg_data_dict.update({"mcq_list": decoded_response})
        kg_data_list_with_mcqs.append(kg_data_dict)

    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for kg_data in kg_data_list_with_mcqs:
            json.dump(kg_data, outfile, ensure_ascii=False)
            outfile.write('\n')


async def main():
    random.seed(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="llama3_cc")
    parser.add_argument('--kg_file', type=str, default="/data/jbai/{dataset_name}", help="Path to the input file.")
    parser.add_argument('--output_file', type=str, default="../data/{dataset_name}/mcq_{dataset_name}_llama3-8b.json", help="Path to the output file.")
    parser.add_argument('--logging_file', type=str, default="./generate_log_for_test.log", help="Path to the logging file.")
    parser.add_argument('-m', '--model', type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help="Model name in huggingface or local path.") # meta-llama/Llama-3-8b-chat-hf
    parser.add_argument('-b', '--batch_size', type=int, default=5, help="Number of sessions processed at the same time.")
    parser.add_argument('-sample', '--sample_size', type=int, default=200, help="Number of samples.")
    args = parser.parse_args()

    await generate(args)


if __name__=="__main__":
    asyncio.run(main())