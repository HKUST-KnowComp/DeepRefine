import random
import argparse
import asyncio
import logging
import csv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

API_KEY = ""
BASE_URL = ""


EVENT_PROMPT = '''I will give you an EVENT. You need to give several phrases containing 1-2 words for the ABSTRACT EVENT of this EVENT.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract event words should fulfill the following requirements.
            1. The ABSTRACT EVENT phrases can well represent the EVENT, and it could be the type of the EVENT or the related concepts of the EVENT.    
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

            EVENT: A man retreats to mountains and forests
            Your answer: retreat, relaxation, escape, nature, solitude
            
            EVENT: A cat chased a prey into its shelter
            Your answer: hunting, escape, predation, hidding, stalking

            EVENT: Sam playing with his dog
            Your answer: relaxing event, petting, playing, bonding, friendship

            EVENT: [EVENT]
            Your answer:
            '''

ENTITY_PROMPT = '''I will give you an ENTITY. You need to give several phrases containing 1-2 words for the ABSTRACT ENTITY of this ENTITY.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract intention words should fulfill the following requirements.
            1. The ABSTRACT ENTITY phrases can well represent the ENTITY, and it could be the type of the ENTITY or the related concepts of the ENTITY.
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

            ENTITY: Soul
            CONTEXT: premiered BFI London Film Festival, became highest-grossing Pixar release
            Your answer: movie, film

            ENTITY: Thinkpad X60
            CONTEXT: Richard Stallman announced he is using Trisquel on a Thinkpad X60
            Your answer: Thinkpad, laptop, machine, device, hardware, computer, brand

            ENTITY: Harry Callahan
            CONTEXT: bluffs another robber, tortures Scorpio
            Your answer: person, Amarican, character, police officer, detective

            ENTITY: Black Mountain College
            CONTEXT: was started by John Andrew Rice, attracted faculty
            Your answer: college, university, school, liberal arts college

            EVENT: 1st April
            CONTEXT: Utkal Dibas celebrates
            Your answer: date, day, time, festival

            ENTITY: [ENTITY]
            CONTEXT: [CONTEXT]
            Your answer:
            '''

RELATION_PROMPT = '''I will give you an RELATION. You need to give several phrases containing 1-2 words for the ABSTRACT RELATION of this RELATION.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract intention words should fulfill the following requirements.
            1. The ABSTRACT RELATION phrases can well represent the RELATION, and it could be the type of the RELATION or the simplest concepts of the RELATION.
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.
            
            RELATION: participated in
            Your answer: become part of, attend, take part in, engage in, involve in

            RELATION: be included in
            Your answer: join, be a part of, be a member of, be a component of

            RELATION: [RELATION]
            Your answer:
            '''


LLaMA3_CHAT_TEMPLATE = {
        "system_start": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>""",
        "prompt_start": """<|start_header_id|>user<|end_header_id|>""",
        "prompt_end": "<|eot_id|>",
        "model_start": "<|start_header_id|>assistant<|end_header_id|>"
    }


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
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_api_model(query):
        async with semaphore:
            return await api_model(args, query)

    tasks = [limited_api_model(query) for query in queries]
    answers = await tqdm_asyncio.gather(*tasks, total=len(tasks))
    return answers


async def generate_entity_concepts(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.logging_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    triple_file = args.triple_file.replace("{dataset_name}", args.dataset)
    type_file = args.type_file.replace("{dataset_name}", args.dataset)
    output_file = args.output_file.replace("{dataset_name}", args.dataset)

    # read data
    entity_set = set()
    relation_set = set()
    predecessors = {}
    successors = {}
    with open(triple_file, encoding='utf-8') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            if args.dataset == "YAGO43kET":
                h = " ".join(h.split("_"))
                t = " ".join(t.split("_"))
            entity_set.add(h)
            entity_set.add(t)
            relation_set.add(r)
            if t not in predecessors.keys():
                predecessors[t] = [" ".join([h, r, t])]
            else:
                predecessors[t].append(" ".join([h, r, t]))
            if h not in predecessors.keys():
                predecessors[h] = []
            if t not in successors.keys():
                successors[t] = []
            if h not in successors.keys():
                successors[h] = [" ".join([h, r, t])]
            else:
                successors[h].append(" ".join([h, r, t]))
                
        # remove unobserved entities and collect types
        gt_entity_concept = {}
        with open(type_file, encoding='utf-8') as r:
            for line in r:
                entity, concept = line.strip().split('\t')
                if args.dataset == "YAGO43kET":
                    entity = " ".join(entity.split("_"))
                    concept = " ".join(concept.split("_")[1:])
                    if entity in entity_set:
                        if entity not in gt_entity_concept.keys():
                            gt_entity_concept[entity] = [concept]
                        else:
                            gt_entity_concept[entity].append(concept)
                elif args.dataset == "FB15kET":
                    concepts = concept.split("/")[-1:]
                    concepts = list(set(concepts))
                    if entity in entity_set:
                        if entity not in gt_entity_concept.keys():
                            gt_entity_concept[entity] = concepts
                        else:
                            gt_entity_concept[entity].extend(concepts)
        if args.dataset == "FB15kET":
            for entity in gt_entity_concept.keys():
                gt_entity_concept[entity] = list(set(gt_entity_concept[entity]))
        gt_concept = gt_entity_concept

    # prompt construction & generation
    pred_entity_concept = {}
    query_list = []
    for entity in gt_concept.keys():
        random_predecessors = random.sample(predecessors[entity], min(3, len(predecessors[entity])))
        random_successors = random.sample(successors[entity], min(3, len(successors[entity])))
        context_list = random_predecessors + random_successors
        context = ", ".join([f"{neighbor}" for neighbor in context_list])
        query = ENTITY_PROMPT.replace("[ENTITY]", entity)
        query = query.replace("[CONTEXT]", context)
        query_list.append(query)
    answers = await _run_api(args=args, queries=query_list)
    for answer, entity in zip(answers, gt_concept.keys()):
        answer = answer.split('\n')[-1]
        pred_entity_concept[entity] = []
        pred_entity_concept[entity].extend([x.strip().lower() for x in answer.split(",")])

    # write the generation resutls
    with open(output_file, "w", newline='') as file:
        # write to csv
        csv_writer = csv.writer(file)
        csv_writer.writerow(["node", "conceptualized_node", "node_type"])
        for entity in pred_entity_concept.keys():
            csv_writer.writerow([entity, pred_entity_concept[entity], "entity"])


async def generate_relation_concepts(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.logging_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    triple_file = args.triple_file.replace("{dataset_name}", args.dataset)
    output_file = args.output_file.replace("{dataset_name}", args.dataset)

    # read data
    gt_relation_concept = {}
    with open(triple_file, encoding='utf-8') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            if "." in r:
                r1 = r.split(".")[0].split("/")[-1]
                concepts_r1 = r.split(".")[0].split("/")[1:-1]
                r2 = r.split(".")[1].split("/")[-1]
                concepts_r2 = r.split(".")[1].split("/")[1:-1]
                r_list = [r1, r2]
                concepts_list = [concepts_r1, concepts_r2]
            else:
                r_list = [r.split("/")[-1]]
                concepts_list = [r.split("/")[1:-1]]
            for r, concepts in zip(r_list, concepts_list):
                if r not in gt_relation_concept.keys():
                    gt_relation_concept[r] = concepts
                else:
                    gt_relation_concept[r].extend(concepts)
    gt_concept = gt_relation_concept

    # prompt construction & generation
    pred_relation_concept = {}
    query_list = []
    for event in gt_concept.keys():
        query = RELATION_PROMPT.replace("[RELATION]", event)
        query_list.append(query)
    answers = await _run_api(args=args, queries=query_list)
    for answer, event in zip(answers, gt_concept.keys()):
        answer = answer.split('\n')[-1]
        pred_relation_concept[event] = []
        pred_relation_concept[event].extend([x.strip().lower() for x in answer.split(",")])

    # write the generation resutls
    with open(output_file, "w", newline='') as file:
        # write to csv
        csv_writer = csv.writer(file)
        csv_writer.writerow(["node", "conceptualized_node", "node_type"])
        for event in pred_relation_concept.keys():
            csv_writer.writerow([event, pred_relation_concept[event], "relation"])


async def generate_event_concepts(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.logging_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    type_file = args.type_file.replace("{dataset_name}", args.dataset)
    output_file = args.output_file.replace("{dataset_name}", args.dataset)

    # read data
    gt_event_concept = {}
    with open(type_file, encoding='utf-8') as r:
        for line in r:
            event, concept = line.strip().split('\t')
            if event not in gt_event_concept.keys():
                gt_event_concept[event] = [concept]
            else:
                gt_event_concept[event].append(concept)
    for event in gt_event_concept.keys():
        gt_event_concept[event] = list(set(gt_event_concept[event]))
    gt_concept = gt_event_concept

    # prompt construction & generation
    pred_event_concept = {}
    query_list = []
    for event in gt_concept.keys():
        query = EVENT_PROMPT.replace("[EVENT]", event)
        query_list.append(query)
    answers = await _run_api(args=args, queries=query_list)
    for answer, event in zip(answers, gt_concept.keys()):
        answer = answer.split('\n')[-1]
        pred_event_concept[event] = []
        pred_event_concept[event].extend([x.strip().lower() for x in answer.split(",")])

    # write the generation resutls
    with open(output_file, "w", newline='') as file:
        # write to csv
        csv_writer = csv.writer(file)
        csv_writer.writerow(["node", "conceptualized_node", "node_type"])
        for event in pred_event_concept.keys():
            csv_writer.writerow([event, pred_event_concept[event], "event"])


async def main():
    random.seed(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="YAGO43kET")
    parser.add_argument('--triple_file', type=str, default="../data/{dataset_name}/test.txt", help="Path to the input file.")
    parser.add_argument('--type_file', type=str, default="../data/{dataset_name}/node_type_test.txt", help="Path to the input file.")
    parser.add_argument('--output_file', type=str, default="../result/node_concept_prediction_{dataset_name}_llama3-80b.csv", help="Path to the output file.")
    parser.add_argument('--logging_file', type=str, default="./generate_log_for_test.log", help="Path to the logging file.")
    parser.add_argument('-m', '--model', type=str, default="meta-llama/Llama-3-8b-chat-hf", help="Model name in huggingface or local path.")
    parser.add_argument('-b', '--batch_size', type=int, default=5, help="Number of sessions processed at the same time.")
    args = parser.parse_args()

    if args.dataset in ["YAGO43kET", "FB15kET"]:
        await generate_entity_concepts(args)
    elif args.dataset in ["wikihow"]:
        await generate_event_concepts(args)
    elif args.dataset in ["FB15kRT"]:
        await generate_relation_concepts(args)


if __name__=="__main__":
    asyncio.run(main())