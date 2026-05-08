import ast
import argparse
import csv
from utils import get_tokens, get_bert_score, get_bert_global_recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="YAGO43kET", type=str)
    parser.add_argument("--pred_file", default="../result/node_concept_prediction_{dataset_name}_llama3-8b.csv", type=str)
    parser.add_argument('--triple_file', type=str, default="../data/{dataset_name}/test.txt", help="Path to the input file.")
    parser.add_argument('--type_file', type=str, default="../data/{dataset_name}/node_type_test.txt", help="Path to the input file.")

    args = parser.parse_args()

    pred_file = args.pred_file.replace("{dataset_name}", args.dataset)
    triple_file = args.triple_file.replace("{dataset_name}", args.dataset)
    type_file = args.type_file.replace("{dataset_name}", args.dataset)

    # read data
    if args.dataset in ["YAGO43kET", "FB15kET", "FB15kRT"]:
        # collect triple set
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
        if args.dataset in ["YAGO43kET", "FB15kET"]:    # entity typing
            gt_entity_concept = {}
            with open(type_file, encoding='utf-8') as r:
                for line in r:
                    entity, concept = line.strip().split('\t')
                    if args.dataset == "YAGO43kET":
                        entity = " ".join(entity.split("_"))
                        # concept = " ".join(concept.split("_")[1:])
                        concept = concept.split("_")[-1]
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
            gt_concept = []
            for item in gt_entity_concept.keys():
                gt_concept.append(list(set(gt_entity_concept[item])))
        elif args.dataset in ["FB15kRT"]:   # relation typing
            gt_relation_concept = {}
            concept_list = []
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
                        concept_list.extend(concepts_r1)
                        concept_list.extend(concepts_r2)
                    else:
                        r_list = [r.split("/")[-1]]
                        concepts_list = [r.split("/")[1:-1]]
                        concept_list.extend(concepts_list[0])
                    for r, concepts in zip(r_list, concepts_list):
                        if r not in gt_relation_concept.keys():
                            gt_relation_concept[r] = concepts
                        else:
                            gt_relation_concept[r].extend(concepts)
            gt_concept = []
            for item in gt_relation_concept.keys():
                gt_concept.append(list(set(gt_relation_concept[item])))
            concept_list = list(set(concept_list))
            print(f"Concepts Size: {len(concept_list)}")
    elif args.dataset in ["wikihow"]:
        gt_event_concept = {}
        concept_list = []
        with open(type_file, encoding='utf-8') as r:
            for line in r:
                event, concept = line.strip().split('\t')
                concept_list.append(concept)
                if event not in gt_event_concept.keys():
                    gt_event_concept[event] = [concept]
                else:
                    gt_event_concept[event].append(concept)
        gt_concept = []
        for item in gt_event_concept.keys():
            gt_concept.append(list(set(gt_event_concept[item])))
        concept_list = list(set(concept_list))
        print(f"Concepts Size: {len(concept_list)}")
            
    # conceputalized results
    pred_node_concept = []
    with open(pred_file, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            try:
                node = row[0]
                concept_list = ast.literal_eval(row[1].strip())
                pred_node_concept.append(concept_list)
            except:
                pred_node_concept.append(["None"])
    filtered_pred_node_concept = []
    for item in pred_node_concept:
        filtered_pred_item = [pred for pred in item if pred.strip()]
        filtered_pred_node_concept.append(filtered_pred_item)
    pred_node_concept = filtered_pred_node_concept

    assert len(gt_concept) == len(pred_node_concept)

	# Evaluate
    gold_tokens, pred_tokens = get_tokens(gt_concept, pred_node_concept)

    print("Get bert global recall...")
    bert_global_recall = get_bert_global_recall(gt_concept, pred_node_concept)
    print(f"BertScore Global Recall Score: {bert_global_recall}")
    
    print("Get bert score...")
    precisions_BS, recalls_BS, f1s_BS = get_bert_score(gt_concept, pred_node_concept)
    print(f'BertScore Precision Score: {precisions_BS.sum() / len(gt_concept):.4f}')
    print(f'BertScore Recall Score: {recalls_BS.sum() / len(gt_concept):.4f}')
    print(f'BertScore F1 Score: {f1s_BS.sum() / len(gt_concept):.4f}\n')