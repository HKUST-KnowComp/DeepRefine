import json  
import random  
from tenacity import retry,stop_after_attempt,stop_after_delay,wait_fixed
from openai import OpenAI
import re
import csv
import pandas as pd
import os

cost = 0
API_KEY = ""
BASE_URL = ""

def main_for_one_file(input_file):
    #input_file = 'meta-llama_Meta-Llama-3-8B-Instruct_biology_output_20240428225851_1_in_6.json'
    #output_file = 'cleaned_file.json'
    input_file = input_file
    def generate_cleaned_filename(original_path):
        base, ext = os.path.splitext(original_path)
        new_filename = f"{base}_cleaned{ext}"
        return new_filename

    output_file = generate_cleaned_filename(input_file)
    
    if not os.path.exists(output_file):
        clean_file(input_file, output_file)
    
    calculate_file = os.path.splitext(output_file)[0] +'.csv'
    
    with open(calculate_file, 'w', newline='',  encoding='utf-8', errors='ignore') as file:
        writer = csv.writer(file)
        writer.writerow(["i", "precision_1", "recall_1", "f1_1", "precision_2", "recall_2", "f1_2", "precision_3", "recall_3", "f1_3"])
    
    with open(output_file, 'r', encoding='utf-8') as file:  
        data = json.load(file)  
    
    random_numbers = [random.randint(0, len(data)-1) for _ in range(100)]
    
    # 处理每个对象的 original_text 字段  
    print(len(data))
    for i in random_numbers:  
        #对单个对象进行处理
        print(i)
        item = data[i]
        text = item['original_text']  
        entity_relation_dict = item['entity_relation_dict'] 
        event_entity_relation_dict = item['event_entity_relation_dict']
        event_relation_dict = item['event_relation_dict']
        # 删除 "Here is the passage" 以前的内容  
        if "Here is the passage" in text:  
            text = text.split("Here is the passage. ", 1)[1].strip()

        try:
            [len_1,len_2,len_3] = count_origin(entity_relation_dict, event_entity_relation_dict, event_relation_dict)
            [task1_more_answer, task2_more_answer, task3_more_answer] = get_more(text, entity_relation_dict, event_entity_relation_dict, event_relation_dict)
            [len1_more, len2_more, len3_more] = count_more(task1_more_answer, task2_more_answer, task3_more_answer)
            [task1_incorrect_answer, task2_incorrect_answer, task3_incorrect_answer] = get_incorrect(text, entity_relation_dict, event_entity_relation_dict, event_relation_dict)
            [len1_incorrect,len2_incorrect, len3_incorrect] = count_incorrect(task1_incorrect_answer, task2_incorrect_answer, task3_incorrect_answer)
    
            [precision_1, recall_1, f1_1] = calculate(len_1, len1_incorrect, len1_more)
            [precision_2, recall_2, f1_2] = calculate(len_2, len2_incorrect, len2_more)
            [precision_3, recall_3, f1_3] = calculate(len_3, len3_incorrect, len3_more)

            print("result:" + str([i, precision_1, recall_1, f1_1, precision_2, recall_2, f1_2, precision_3, recall_3, f1_3]))
            
            with open(calculate_file, 'a', newline='',  encoding='utf-8', errors='ignore') as file:
                writer = csv.writer(file)
                writer.writerow([i, precision_1, recall_1, f1_1, precision_2, recall_2, f1_2, precision_3, recall_3, f1_3])
    
            print("cost:" + str(cost))
        except Exception as e:
            print(f"出现异常：{e}")
            pass

def calculate(len, len_incorrect, len_more):
    len = len
    len_incorrect = len_incorrect
    len_more = len_more

    if len != 0 and (len>len_incorrect):
        precision = (len - len_incorrect) / len
        precision = round(precision, 4)
        recall = (len- len_incorrect) / (len- len_incorrect + len_more)
        recall = round(recall, 4)
    else: 
        precision = None
        recall = None


    if (precision != None) and (recall != None):
        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = round(f1, 4)
    else:
        f1 = None

    return [precision, recall, f1]


def clean_file(input_file, output_file):
# 输入和输出文件定义  
#input_file = 'meta-llama_Meta-Llama-3-8B-Instruct_biology_output_20240428225851_1_in_6.json'  # 大 JSON 文件路径  
#output_file = 'cleaned_file.json'  # 输出文件路径  
    with open(input_file, 'r') as f, open(output_file, 'w',encoding='utf-8') as out_f:  
        # 创建一个列表以存储清理后的对象  
        cleaned_data = []  
    
        for line in f:  
            try:  
                # 尝试解析每一行  
                item = json.loads(line)  
    
                # 删除不需要的字段  
                item.pop('output_stage_one', None)  
                item.pop('output_stage_two', None)  
                item.pop('output_stage_three', None)  
    
                # 将清理后的对象添加到列表  
                cleaned_data.append(item)  
            except json.JSONDecodeError as e:  
                print(f"错误解析这一行: {line}\n错误信息: {e}")  
    
        # 将清理后的数据写入新的 JSON 文件（作为数组）  
        with open(output_file, 'w',encoding='utf-8') as out_f:  
            json.dump(cleaned_data, out_f, ensure_ascii=False, indent=4)  # 写入 JSON 数据  
    
    print(f"已成功处理并生成新文件: {output_file}")


@retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(10))
def response(content):    
    client = OpenAI(
        api_key=API_KEY, # your api key
        base_url=BASE_URL
        
    )
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3", # model = "deployment_name"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
                ],
    )
    response_content = response.choices[0].message.content
    print(content)
    print(response_content)
    print('-------------------------------------------------')

    pricing = {
            "DeepSeek-V3":
            {
                "input": 2 * 10**-7,
                "output": 8 * 10**-7
            }
        }
    
    amount = pricing["DeepSeek-V3"]["input"] * len(content) + pricing["DeepSeek-V3"]["output"] * len(response_content)
    global cost
    original = cost
    cost = original + amount
    
    return response_content


def count_origin(entity_relation_dict, event_entity_relation_dict, event_relation_dict):
    entity_relation_dict = entity_relation_dict
    event_entity_relation_dict = event_entity_relation_dict
    event_relation_dict = event_relation_dict
    
    len_1 = len(entity_relation_dict)
    len_3 = len(event_relation_dict)
    
    #count the original named-entity:
    entities = [entity for item in event_entity_relation_dict for entity in item['Entity']]
    #unique_entities = set(entities)
    #print(unique_entities)
    len_2 = len(entities)
    
    return [len_1,len_2,len_3]


def get_more(text, entity_relation_dict, event_entity_relation_dict, event_relation_dict):
    text = text
    entity_relation_dict = entity_relation_dict
    event_entity_relation_dict = event_entity_relation_dict
    event_relation_dict = event_relation_dict
    
    #返回未识别关系的提示
    text_prompt_2 = "I need you to help me to find more unrecognized results for information extraction, \
    The text paragraphs are provided as followed:"
    
    task1_prompt_2 = "\nIf the relations are all recognized, only output \"All recognized!\", don't output anything else. \
    Else, output unrecognized triples strictly line by line as the original format like\
    {'Head': '', 'Relation': '', 'Tail': ''}.Output triples as least as possible.No blank lines.No other words.Line breaks are not allowed in a triple. \
    Output triples shouldn't contain any given event relationship, or similar to the given event relationship.\
    The given entity relationship recognition results as follows: "
    
    task2_prompt_2 = "\nIf the entities are all recognized, output \"All recognized!\" only, don't output anything else. \
    Else, output unrecognized triples strictly line by line as the original format like\
    {'Event': 'sentence', 'Entity': ['entity1','entity2','...']}.Line breaks are not allowed in a triple.Output triples as least as possible.\
    Given entities or entities similar to them should be excluded from the entity list.No blank lines.No other words.\
    The given entity recognition results as followed: "
    
    task3_prompt_2 = "\nIf the relations are all recognized, output \"All recognized!\" only, don't output anything else. \
    Else, output unrecognized triples strictly line by line as the original format like\
    {'Head': '', 'Relation': '', 'Tail': ''}.\
    The relationships types are :before, after, at the same time, because, and as a result.\
    No blank lines.No other words.Line breaks are not allowed in a triple.Output triples as least as possible.\
    Output triples shouldn't contain any given event relationship, or similar to the given event relationship.\
    The given event relationship recognition results as followed:"

    
    task1_more_answer = response(text_prompt_2+ str(text) + task1_prompt_2 + str(entity_relation_dict))
    task2_more_answer = response(text_prompt_2+ str(text) + task2_prompt_2 + str(event_entity_relation_dict))
    task3_more_answer = response(text_prompt_2+ str(text) + task3_prompt_2 + str(event_relation_dict))

    return [task1_more_answer, task2_more_answer, task3_more_answer]

def count_more(task1_more_answer, task2_more_answer, task3_more_answer):
    def remove_empty_lines(text):
        lines = text.split("\n")  # Split the text into lines
        non_empty_lines = [line for line in lines if line.strip() != ""]  # Filter out empty lines
        return "\n".join(non_empty_lines)  # Join the non-empty lines back into a single string

    task1_more_answer = remove_empty_lines(task1_more_answer)
    task2_more_answer = remove_empty_lines(task2_more_answer)
    task3_more_answer = remove_empty_lines(task3_more_answer)
    
    if "All recognized" in str(task1_more_answer):
        len1_more = 0
    else:
        len1_more = len(task1_more_answer.split('\n'))
    
    if "All recognized" in str(task3_more_answer):
        len3_more = 0
    else:
        len3_more = len(task3_more_answer.split('\n'))
    
    if "All recognized" in str(task2_more_answer):
        len2_more = 0
        entities = []
    else:
        more_2 = task2_more_answer.split('\n')
        entities = []
        for item in more_2:
            pattern = r'\[(.*?)\]'
            match = re.search(pattern, item)
            if match:
                entity_match = match.group().strip("[]").split("\', \'")
                for i in entity_match:
                    cleaned_text = i.strip("'")
                    entities.append(cleaned_text)
            else:
                print("no entities!")
        len2_more= len(entities)
    
    return [len1_more, len2_more, len3_more]


def get_incorrect(text, entity_relation_dict, event_entity_relation_dict, event_relation_dict):
    text = text
    entity_relation_dict = entity_relation_dict
    event_entity_relation_dict = event_entity_relation_dict
    event_relation_dict = event_relation_dict
    
    text_prompt_1 = "I need you to help me evaluate the results from information extraction, \
    The text paragraphs are provided as followed:"
    
    task1_prompt_1 = "\nEvaluate the entity relationships with an emphasis on general interpretations and broader meanings rather than strict details.\
    Accept general phrases, reasonable expressions, and contextual representations as correct, even if they are not strictly word-for-word accurate. \
    Focus on capturing the essence of the relationships instead of precise details.\
    If any relationships can be reasonably interpreted as correct, consider them so.\
    All relationships should be accepted as correct unless they are explicitly unsupported by the text.\
    If it is all correct, output \"all correct!\" only. \
    If is not all correct, output the incorrect triples strictly line by line as the original format like\
    {'Head': '', 'Relation': '', 'Tail': ''}.Output triples as least as possible.No blank lines.Line breaks are not allowed in a triple.\
    The model entity relationship recognition results as follows: "

    
    task2_prompt_1 = "\nEvaluate the following extracted entities based on the events described. \
    Assume that every entity mentioned has a significant relevance to the events, either directly or indirectly.\
    Recognize that general involvement, titles, and roles are valid entities, even if they appear redundant or lack explicit support. \
    Approach the evaluation with a broad interpretation of connections and consider that all entities are relevant to the historical context. \
    General or repeated entities should be considered correct.\
    Similar or reasonable entities should be considered correct. \
    Entities not central to specific events should be considered correct.\
    Entities lack of explicit support or direct support should be considered correct.\
    Titles and roles should be considered as correct entities, even with name redundancy.\
    All entities should be accepted as correct unless they are explicitly unsupported by the event.\
    If it is all correct, output \"all correct!\" only. \
    If it is not all correct, output the incorrect triples strictly line by line as the original format like\
    {'Event': 'sentence', 'Entity': ['entity1','entity2', '...']}.Line breaks are not allowed in a triple.No blank lines.Output triples as least as possible.\
    Correct entity should be excluded from the entity list.\
    The model named entity recognition results as followed: "
    
    task3_prompt_1 = "\nEvaluate the relationships between the events derived from the provided text.\
    The relationships types are :before, after, at the same time, because, and as a result.\
    Focus on capturing the essence of the relationships instead of precise details.\
    Consider them all correct if they logically align with the information provided, even if they are not exact.\
    If any relationships can be reasonably interpreted or inferred as correct, consider them so.\
    All relationships should be accepted as correct unless they are explicitly unsupported by the text.\
    If it is all correct, output \"all correct!\" only. \
    If is not all correct, output the incorrect triples strictly line by line as the original format like\
    {'Head': '', 'Relation': '', 'Tail': ''}.Line breaks are not allowed in a triple.No blank lines.Output triples as least as possible.\
    Similar or reasonable expression should be considered correct.Don't need to be same with original text word by word.\
    The model event relationship recognition results as followed:"
    
    task1_incorrect_answer = response(text_prompt_1+ str(text) + task1_prompt_1 + str(entity_relation_dict))
    task2_incorrect_answer = response(text_prompt_1+ str(text) + task2_prompt_1 + str(event_entity_relation_dict))
    task3_incorrect_answer = response(text_prompt_1+ str(text) + task3_prompt_1 + str(event_relation_dict))

    return [task1_incorrect_answer, task2_incorrect_answer, task3_incorrect_answer]


def count_incorrect(task1_incorrect_answer, task2_incorrect_answer, task3_incorrect_answer):
    def remove_empty_lines(text):
        lines = text.split("\n")  # Split the text into lines
        non_empty_lines = [line for line in lines if line.strip() != ""]  # Filter out empty lines
        return "\n".join(non_empty_lines)  # Join the non-empty lines back into a single string

    task1_incorrect_answer = remove_empty_lines(task1_incorrect_answer)
    task2_incorrect_answer = remove_empty_lines(task2_incorrect_answer)
    task3_incorrect_answer = remove_empty_lines(task3_incorrect_answer)
    
    if "all correct" in str(task1_incorrect_answer):
        len1_incorrect = 0
    else:
        len1_incorrect = len(task1_incorrect_answer.split('\n'))
    
    if "all correct" in str(task3_incorrect_answer):
        len3_incorrect = 0
    else:
        len3_incorrect = len(task3_incorrect_answer.split('\n'))
    
    if "all correct" in str(task2_incorrect_answer):
        len2_incorrect = 0
        entities = []
    else:
        incorrect_2 = task2_incorrect_answer.split('\n')
        entities = []
        for item in incorrect_2:
            pattern = r'\[(.*?)\]'
            match = re.search(pattern, item)
            if match:
                entity_match = match.group().strip("[]").split("\', \'")
                for i in entity_match:
                    cleaned_text = i.strip("'")
                    entities.append(cleaned_text)
            else:
                print("no entities!")
        len2_incorrect = len(entities)
    return [len1_incorrect,len2_incorrect, len3_incorrect]


if __name__ == "__main__":
    main_for_one_file('path_to_your_file')
    # calculate average
    df = pd.read_csv('path_to_your_result_file')
    mean_values = df.iloc[:, 1:].mean()
    print(mean_values)