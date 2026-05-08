from atlas_rag.retriever.base import BasePassageRetriever
import random
class UpperBoundRetriever(BasePassageRetriever):
    def __init__(self):
        return
    def retrieve(self, query, topN=5, **kwargs):
        # get full list of docs from kwargs
        sorted_passages_contents = kwargs.get("sorted_passages_contents", [])
        sorted_passage_ids = kwargs.get("sorted_passage_ids", [])
        full_list_passages_contents = kwargs.get("full_list_passages_contents", set())
        full_list_passages_ids = kwargs.get("full_list_passages_ids", set())
        if len(sorted_passages_contents) < topN:
            # add random passage from full_list_passages_contents until top 5
            gold_doc_set = set(sorted_passages_contents)
            # random shuffle full_list_passages_contents
            full_list_passages_contents = list(full_list_passages_contents)
            random.shuffle(full_list_passages_contents)
            for passage in full_list_passages_contents:
                if passage not in gold_doc_set:
                    sorted_passages_contents.append(passage)
                    sorted_passage_ids.append("random_added")
                if len(sorted_passages_contents) >= topN:
                    break
        return sorted_passages_contents, sorted_passage_ids