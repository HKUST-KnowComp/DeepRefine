import numpy as np
from itertools import chain
from rouge_score import rouge_scorer
from bert_score import score as score_bert
from scipy.optimize import linear_sum_assignment
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import re
from tqdm import tqdm


def get_tokens(gold_edges, pred_edges):
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab, infix_finditer=re.compile(r'''[;]''').finditer)

    gold_tokens = []
    pred_tokens = []

    for i in range(len(gold_edges)):
        gold_tokens_edges = []
        pred_tokens_edges = []

        for sample in tokenizer.pipe(gold_edges[i]):
            gold_tokens_edges.append([j.text for j in sample])
        for sample in tokenizer.pipe(pred_edges[i]):
            pred_tokens_edges.append([j.text for j in sample])
        gold_tokens.append(gold_tokens_edges)
        pred_tokens.append(pred_tokens_edges)

    return gold_tokens, pred_tokens


def process_concepts(concepts_list):
    processed_concepts = []
    for concepts in concepts_list:
        processed_concepts.append([",".join(str(concept)).lower().strip() for concept in concepts])
    return processed_concepts


def get_bert_global_recall(all_gold_edges, all_pred_edges):
    """
    Calculate Coverage based on BERT-Score between ground truth and prediction edges.
    """
    gold_set = set(chain.from_iterable(all_gold_edges))
    pred_set = set(chain.from_iterable(all_pred_edges))

    if not gold_set:
        return 1.0

    if not pred_set:
        return 0.0

    gold_list = list(gold_set)
    pred_list = list(pred_set)

    ref_cand_index = {}
    references = []
    candidates = []

    for i, gold_edge in enumerate(gold_list):
        for j, pred_edge in enumerate(pred_list):
            references.append(gold_edge)
            candidates.append(pred_edge)
            ref_cand_index[(i, j)] = len(references) - 1

    _, _, bs_F1 = score_bert(
        cands=candidates,
        refs=references,
        model_type="roberta-large",
        lang='en',
        idf=False,
        device="cuda:4"
    )
    print("Computed BERT scores for all pairs")

    score_matrix = np.zeros((len(gold_list), len(pred_list)))
    for i in tqdm(range(len(gold_list))):
        for j in range(len(pred_list)):
            if (i, j) in ref_cand_index:
                score_matrix[i][j] = bs_F1[ref_cand_index[(i, j)]]

    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

    matched_scores = score_matrix[row_ind, col_ind]
    recall = matched_scores.sum() / len(gold_list)

    return recall


def get_bert_score(all_gold_edges, all_pred_edges):
    references = []
    candidates = []

    ref_cand_index = {}
    for i in tqdm(range(len(all_gold_edges))):              # for each sample in the dataset
        gold_edges = all_gold_edges[i]
        pred_edges = all_pred_edges[i]
        for (i, gold_edge) in enumerate(gold_edges):        # for each item in the ground truth sample
            for (j, pred_edge) in enumerate(pred_edges):    # for each item in the prediction sample
                references.append(gold_edge)
                candidates.append(pred_edge)
                ref_cand_index[(gold_edge, pred_edge)] = len(references) - 1    # similarity matrix

    _, _, bs_F1 = score_bert(cands=candidates, refs=references, model_type="roberta-large", lang='en', idf=False, device="cuda:4")
    print("Computed bert scores for all pairs")

    precisions, recalls, f1s = [], [], []
    for i in tqdm(range(len(all_gold_edges))):
        gold_edges = all_gold_edges[i]
        pred_edges = all_pred_edges[i]
        score_matrix = np.zeros((len(gold_edges), len(pred_edges)))
        for (i, gold_edge) in enumerate(gold_edges):
            for (j, pred_edge) in enumerate(pred_edges):
                score_matrix[i][j] = bs_F1[ref_cand_index[(gold_edge, pred_edge)]]

        row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)   # find matching

        sample_precision = score_matrix[row_ind, col_ind].sum() / len(pred_edges)
        sample_recall = score_matrix[row_ind, col_ind].sum() / len(gold_edges)

        precisions.append(sample_precision)
        recalls.append(sample_recall)
        f1s.append(2 * sample_precision * sample_recall / (sample_precision + sample_recall))

    return np.array(precisions), np.array(recalls), np.array(f1s)
