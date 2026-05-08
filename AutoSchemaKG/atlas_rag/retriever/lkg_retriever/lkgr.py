from difflib import get_close_matches
from logging import Logger
import faiss
from neo4j import GraphDatabase
import time
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from graphdatascience import GraphDataScience
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
import string
from atlas_rag.retriever.lkg_retriever.base import BaseLargeKGRetriever



class LargeKGRetriever(BaseLargeKGRetriever):
    def __init__(self, keyword:str, neo4j_driver: GraphDatabase, 
                llm_generator:LLMGenerator, sentence_encoder:BaseEmbeddingModel, 
                node_index:faiss.Index, passage_index:faiss.Index, 
                topN: int   = 5,
                number_of_source_nodes_per_ner: int  = 10,
                sampling_area : int = 250,logger:Logger = None, **kwargs): 
        # istantiate one kg resources
        self.keyword = keyword
        self.neo4j_driver = neo4j_driver
        self.gds_driver = GraphDataScience(self.neo4j_driver)
        self.topN = topN
        self.number_of_source_nodes_per_ner = number_of_source_nodes_per_ner
        self.sampling_area = sampling_area
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.node_faiss_index = node_index
        # self.edge_faiss_index = self.edge_indexes[keyword]
        self.passage_faiss_index = passage_index

        self.verbose = False if logger is None else True
        self.logger = logger
        
        self.ppr_weight_threshold = 0.00005
        self.simple_ner = kwargs.get('simple_ner', False)
   
    def set_model(self, model):
        if self.llm_generator.inference_type == 'openai':
            self.llm_generator.model_name = model
        else:
            raise ValueError("Model can only be set for OpenAI inference type.")
    
    def ner(self, text):
        return self.llm_generator.large_kg_ner(text, simple_ner=self.simple_ner)
    
    def convert_numeric_id_to_name(self, numeric_id):
        if numeric_id.isdigit():
            return self.gds_driver.util.asNode(self.gds_driver.find_node_id(["Node"], {"numeric_id": numeric_id})).get('name')
        else:
            return numeric_id

    def has_intersection(self, word_set, input_string):
        cleaned_string = input_string.translate(str.maketrans('', '', string.punctuation)).lower()
        if self.keyword == 'cc_en':
            # Check if any phrase in word_set is a substring of cleaned_string
            for phrase in word_set:
                if phrase in cleaned_string:
                    return True
            return False
        else:
            # Check if any word in word_set is present in the cleaned_string's words
            words_in_string = set(cleaned_string.split())
            return not word_set.isdisjoint(words_in_string)

    def pagerank(self, personalization_dict, topN=5, sampling_area=200):
        graph = self.gds_driver.graph.get('largekgrag_graph')
        node_count = graph.node_count()
        sampling_ratio = sampling_area / node_count
        aggregation_node_dict = []
        ppr_weight_threshold = self.ppr_weight_threshold
        start_time = time.time()
        
        # Pre-filter nodes based on ppr_weight threshold
        # Precompute word sets based on keyword
        if self.keyword == 'cc_en':
            filtered_personalization = {
                node_id: ppr_weight 
                for node_id, ppr_weight in personalization_dict.items() 
                if ppr_weight >= ppr_weight_threshold
            }
            stop_words = set(stopwords.words('english'))
            word_set_phrases = set()
            word_set_words = set()
            for node_id, ppr_weight in filtered_personalization.items():
                name = self.gds_driver.util.asNode(node_id)['name']
                if name:
                    cleaned_phrase = name.translate(str.maketrans('', '', string.punctuation)).lower().strip()
                    if cleaned_phrase:
                        # Process for 'cc_en': remove stop words and add cleaned phrase
                        filtered_words = [word for word in cleaned_phrase.split() if word not in stop_words]
                        if filtered_words:
                            cleaned_phrase_filtered = ' '.join(filtered_words)
                            word_set_phrases.add(cleaned_phrase_filtered)
        
            word_set = word_set_phrases if self.keyword == 'cc_en' else word_set_words
            if self.verbose:
                self.logger.info(f"Optimized word set: {word_set}")
        else:
            filtered_personalization = personalization_dict
        if self.verbose:
            self.logger.info(f"largekgRAG : Personalization dict: {filtered_personalization}")
            self.logger.info(f"largekgRAG : Sampling ratio: {sampling_ratio}")
            self.logger.info(f"largekgRAG : PPR weight threshold: {ppr_weight_threshold}")
        # Process each node in the filtered personalization dict
        for node_id, ppr_weight in filtered_personalization.items():
            try:
                self.gds_driver.graph.drop('rwr_sample')
                start_time = time.time()
                G_sample, _ = self.gds_driver.graph.sample.rwr("rwr_sample", graph, concurrency=4, samplingRatio = sampling_ratio, startNodes = [node_id],
                                                               restartProbability = 0.4, logProgress = False)
                if self.verbose:
                    self.logger.info(f"largekgRAG : Sampled graph for node {node_id} in {time.time() - start_time:.2f} seconds")
                start_time = time.time()
                result = self.gds_driver.pageRank.stream(
                    G_sample, maxIterations=30, sourceNodes=[node_id], logProgress=False
                ).sort_values("score", ascending=False)
                
                if self.verbose:
                    self.logger.info(f"pagerank type: {type(result)}")
                    self.logger.info(f"pagerank result: {result}")
                    self.logger.info(f"largekgRAG : PageRank calculated for node {node_id} in {time.time() - start_time:.2f} seconds")
                start_time = time.time()
                # if self.keyword == 'cc_en':
                if self.keyword != 'cc_en':
                    result = result[result['score'] > 0.0].nlargest(50, 'score').to_dict('records')
                else:
                    result = result.to_dict('records')
                if self.verbose:
                    self.logger.info(f"largekgRAG :result: {result}")
                for entry in result:
                    if self.keyword == 'cc_en':
                        node_name = self.gds_driver.util.asNode(entry['nodeId'])['name']
                        if not self.has_intersection(word_set, node_name):
                            continue
                    
                    numeric_id = self.gds_driver.util.asNode(entry['nodeId'])['numeric_id']
                    aggregation_node_dict.append({
                        'nodeId': numeric_id,
                        'score': entry['score'] * ppr_weight
                    })

            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Error processing node {node_id}: {e}")
                    self.logger.error(f"Node is filtered out: {self.gds_driver.util.asNode(node_id)['name']}")
                else:
                    continue

    
        aggregation_node_dict = sorted(aggregation_node_dict, key=lambda x: x['score'], reverse=True)[:25]
        if self.verbose:
            self.logger.info(f"Aggregation node dict: {aggregation_node_dict}")
        if self.verbose:
            self.logger.info(f"Time taken to sample and calculate PageRank: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        with self.neo4j_driver.session() as session:
            intermediate_time = time.time()
            # Step 1: Distribute entity scores to connected text nodes and find the top 5
            query_scores = """
            UNWIND $entries AS entry
            MATCH (n:Node {numeric_id: entry.nodeId})-[:Source]->(t:Text)
            WITH t.numeric_id AS textId, SUM(entry.score) AS total_score
            ORDER BY total_score DESC
            LIMIT $topN
            RETURN textId, total_score
            """
            # Execute query to aggregate scores
            result_scores = session.run(query_scores, entries=aggregation_node_dict, topN=topN)
            top_numeric_ids = []
            top_scores = []

            # Extract the top text node IDs and scores
            for record in result_scores:
                top_numeric_ids.append(record["textId"])
                top_scores.append(record["total_score"])

            # Step 2: Use top numeric IDs to retrieve the original text
            if self.verbose:
                self.logger.info(f"Time taken to prepare query 1 : {time.time() - intermediate_time:.2f} seconds")
            intermediate_time = time.time()
            query_text = """
            UNWIND $textIds AS textId
            MATCH (t:Text {numeric_id: textId})
            RETURN t.original_text AS text, t.numeric_id AS textId
            """
            result_texts = session.run(query_text, textIds=top_numeric_ids)
            topN_passages = []
            score_dict = dict(zip(top_numeric_ids, top_scores))

            # Combine original text with scores
            for record in result_texts:
                original_text = record["text"]
                text_id = record["textId"]
                score = score_dict.get(text_id, 0)
                topN_passages.append((original_text, score))
            if self.verbose:
                self.logger.info(f"Time taken to prepare query 2 : {time.time() - intermediate_time:.2f} seconds")
        # Sort passages by score
        topN_passages = sorted(topN_passages, key=lambda x: x[1], reverse=True)
        top_texts = [item[0] for item in topN_passages][:topN]
        top_scores = [item[1] for item in topN_passages][:topN]
        if self.verbose:
            self.logger.info(f"Total passages retrieved: {len(top_texts)}")
            self.logger.info(f"Top passages: {top_texts}")
            self.logger.info(f"Top scores: {top_scores}")
        if self.verbose:
            self.logger.info(f"Neo4j Query Time: {time.time() - start_time:.2f} seconds")
        return top_texts, top_scores
    
    def retrieve_topk_nodes(self, query, top_k_nodes = 2):
        # extract entities from the query
        entities = self.ner(query)
        if self.verbose:
            self.logger.info(f"largekgRAG : LLM Extracted entities: {entities}")
        if len(entities) == 0:
            entities = [query]
        num_entities = len(entities)
        initial_nodes = []
        for entity in entities:
            entity_embedding = self.sentence_encoder.encode([entity])
            D, I = self.node_faiss_index.search(entity_embedding, top_k_nodes)
            if self.verbose:
                self.logger.info(f"largekgRAG : Search results - Distances: {D}, Indices: {I}")
            initial_nodes += [str(i)for i in I[0]]
        if self.verbose:
            self.logger.info(f"largekgRAG : Initial nodes: {initial_nodes}")
        name_id_map = {}
        for node_id in initial_nodes:
            name = self.convert_numeric_id_to_name(node_id)
            name_id_map[name] = node_id  
            
        topk_nodes = list(set(initial_nodes))
        # convert the numeric id to string and filter again then return numeric id
        keywords_before_filter = [self.convert_numeric_id_to_name(n) for n in initial_nodes]
        filtered_keywords = self.llm_generator.large_kg_filter_keywords_with_entity(query, keywords_before_filter)
    
        # Second pass: Add filtered keywords
        filtered_top_k_nodes = []
        filter_log_dict = {}
        match_threshold = 0.8
        if self.verbose:
            self.logger.info(f"largekgRAG : Filtered Before Match Keywords Candidate: {filtered_keywords}")
        for keyword in filtered_keywords:
            # Check for an exact match first
            if keyword in name_id_map:
                filtered_top_k_nodes.append(name_id_map[keyword])
                filter_log_dict[keyword] = name_id_map[keyword]
            else:
                # Look for close matches using difflib's get_close_matches
                close_matches = get_close_matches(keyword, name_id_map.keys(), n=1, cutoff=match_threshold)
                
                if close_matches:
                    # If a close match is found, add the corresponding node
                    filtered_top_k_nodes.append(name_id_map[close_matches[0]])
                
                filter_log_dict[keyword] = name_id_map[close_matches[0]] if close_matches else None
        if self.verbose:
            self.logger.info(f"largekgRAG : Filtered After Match Keywords Candidate: {filter_log_dict}")
        
        topk_nodes = list(set(filtered_top_k_nodes))
        if len(topk_nodes) > 2 * num_entities:
            topk_nodes = topk_nodes[:2 * num_entities]
        return topk_nodes
    
    def _process_text(self, text):
        """Normalize text for containment checks (lowercase, alphanumeric+spaces)"""
        text = text.lower()
        text = ''.join([c for c in text if c.isalnum() or c.isspace()])
        return set(text.split())
    
    def retrieve_personalization_dict(self, query, number_of_source_nodes_per_ner=5):
        topk_nodes = self.retrieve_topk_nodes(query, number_of_source_nodes_per_ner)
        if topk_nodes == []:
            if self.verbose:
                self.logger.info(f"largekgRAG : No nodes found for query: {query}")
            return {}
        if self.verbose:
            self.logger.info(f"largekgRAG : Topk nodes: {[self.convert_numeric_id_to_name(node_id) for node_id in topk_nodes]}")
        freq_dict_for_nodes = {}
        query = """
            UNWIND $nodes AS node
            MATCH (n1:Node {numeric_id: node})-[r:Source]-(n2:Text)
            RETURN n1.numeric_id as numeric_id, COUNT(DISTINCT n2.text_id) AS fileCount
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, nodes=topk_nodes)
            for record in result:
                freq_dict_for_nodes[record["numeric_id"]] = record["fileCount"]
        # Create the personalization dictionary
        personalization_dict = {self.gds_driver.find_node_id(["Node"],{"numeric_id": numeric_id}): 1 / file_count for numeric_id, file_count in freq_dict_for_nodes.items()}
        if self.verbose:
            self.logger.info(f"largekgRAG : Personalization dict's number of node: {len(personalization_dict)}")
        return personalization_dict
       
    def retrieve_passages(self, query):
        if self.verbose:
            self.logger.info(f"largekgRAG : Retrieving passages for query: {query}")
        topN = self.topN
        number_of_source_nodes_per_ner = self.number_of_source_nodes_per_ner
        sampling_area = self.sampling_area
        personalization_dict = self.retrieve_personalization_dict(query, number_of_source_nodes_per_ner)
        if personalization_dict == {}:
            return [], [0]
        topN_passages, topN_scores = self.pagerank(personalization_dict, topN, sampling_area = sampling_area)
        return topN_passages, topN_scores


    