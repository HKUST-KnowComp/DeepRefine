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
import re

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
- "options": A list of 4 options (e.g., ["Option A", "Option B", "Option C", "Option D"]).
- "answer": The correct answer (e.g., "Option A").

Passage:
{passage}

Now, with the given context, give me the json formatted output directly:
"""

MCQ_ANSWER_PROMPT = """
Given the contexts or evidences:
{contexts}

Here is a multiple-choice question:

Question: {question}

Options:
A. {options_0}
B. {options_1}
C. {options_2}
D. {options_3}

Please select the correct answer by choosing A, B, C, or D. Respond with only the letter of your choice.
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

async def _run_api(args, queries, max_concurrent=4):
    semaphore = asyncio.Semaphore(max_concurrent)
    async def limited_api_model(query):
        async with semaphore:
            return await api_model(args, query)
    tasks = [limited_api_model(query) for query in queries]
    answers = await tqdm_asyncio.gather(*tasks, total=len(tasks))
    return answers

async def generate(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.logging_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    args.kg_file = args.kg_file.replace("{dataset_name}", args.dataset)
    args.output_file = args.output_file.replace("{dataset_name}", args.dataset)

    # get kg data
    kg_data_list = []
    with open(args.kg_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            passage = data['original_text'].split('<|eot_id|>')[1].split('Here is the passage. ')[1]
            entity_relation_list = [f"{triple_dict['Head']}, {triple_dict['Relation']}, {triple_dict['Tail']}" for triple_dict in data['entity_relation_dict']]
            event_entity_list = [f"{event_dict['Event']}" for event_dict in data['event_entity_relation_dict']]
            event_relation_list = [f"{event_dict['Head']}, {event_dict['Relation']}, {event_dict['Tail']}" for event_dict in data['event_relation_dict']]
            data_dict = {
                'passage': passage,
                'entity_relation_list': entity_relation_list,
                'event_entity_list': event_entity_list,
                'event_relation_list': event_relation_list
            }
            kg_data_list.append(data_dict)
    
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
        kg_data_dict.update({"mcq_list": decoded_response})
        kg_data_list_with_mcqs.append(kg_data_dict)

    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for kg_data in kg_data_list_with_mcqs:
            json.dump(kg_data, outfile, ensure_ascii=False)
            outfile.write('\n')

async def evaluate(args):
    args.kg_file = args.kg_file.format(dataset_name=args.dataset)
    args.output_file = args.output_file.format(dataset_name=args.dataset)
    with open(args.kg_file, 'r', encoding='utf-8') as infile:
        kg_data_list_with_mcqs = [json.loads(line) for line in infile]

    total_mcqs = 0
    correct_answers = 0
    all_queries = []
    all_correct_answers = []

    for kg_data in kg_data_list_with_mcqs:
        original_passage = kg_data['passage']
        entity_relation_list = kg_data['entity_relation_list']
        event_entity_list = kg_data['event_entity_list']
        event_relation_list = kg_data['event_relation_list']
        if args.context_type == 'passage':
            contexts = original_passage
        elif args.context_type == 'entity':
            context_list = entity_relation_list
            contexts = "; ".join(context_list)
        elif args.context_type == 'event_only':
            event_list = event_entity_list + event_relation_list
            context_list = event_list
            contexts = "; ".join(context_list)
        elif args.context_type == 'event':
            event_list = event_entity_list + event_relation_list
            context_list = event_list + entity_relation_list
            contexts = "; ".join(context_list)
        elif args.context_type == 'empty':
            contexts = "None"

        for mcq in kg_data['mcq_list']:
            question = mcq['question']
            options = mcq['options']
            correct_answer = mcq['answer']  # e.g., "Option A" -> "A"
            prompt = MCQ_ANSWER_PROMPT.format(
                contexts=contexts,
                question=question, 
                options_0=options[0],
                options_1=options[1],
                options_2=options[2],
                options_3=options[3]
                )
            all_queries.append(prompt)
            all_correct_answers.append(correct_answer)
            total_mcqs += 1

    # Get answers from LLM
    answers = await _run_api(args=args, queries=all_queries, max_concurrent=args.batch_size)

    for i, answer in enumerate(answers):
        match = re.search(r'[A-D]', answer, re.IGNORECASE)
        if match:
            gt_answer_match = re.search(r'[A-D]', all_correct_answers[i], re.IGNORECASE)
            gt_answer = gt_answer_match.group(0).upper()
            user_answer = match.group(0).upper()
            if user_answer == gt_answer:
                correct_answers += 1

    accuracy = correct_answers / total_mcqs if total_mcqs > 0 else 0
    print(f"Total MCQs: {total_mcqs}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.4f}")

async def main():
    random.seed(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="llama3_wiki_full")
    parser.add_argument('--kg_file', type=str, default="../data/{dataset_name}/mcq_{dataset_name}_llama3-8b.json", help="Path to the input file.")
    parser.add_argument('--output_file', type=str, default="../result/mcq_answers_{dataset_name}_llama3-8b.json", help="Path to the output file.")
    parser.add_argument('--logging_file', type=str, default="./generate_log_for_test.log", help="Path to the logging file.")
    parser.add_argument('-m', '--model', type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help="Model name in huggingface or local path.")
    parser.add_argument('-b', '--batch_size', type=int, default=8, help="Number of sessions processed at the same time.")
    parser.add_argument('-sample', '--sample_size', type=int, default=200, help="Number of samples.")
    parser.add_argument('-context_type', '--context_type', type=str, default='event', help="Context type. empty or passage or event or event_only or entity")
    args = parser.parse_args()

    # await generate(args)
    await evaluate(args)

if __name__=="__main__":
    asyncio.run(main())