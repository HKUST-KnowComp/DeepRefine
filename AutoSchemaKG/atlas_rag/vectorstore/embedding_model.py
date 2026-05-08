from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import torch.nn.functional as F
from abc import ABC, abstractmethod
import csv
from openai import OpenAI
import torch

class BaseEmbeddingModel(ABC):
    def __init__(self, sentence_encoder):
        self.sentence_encoder = sentence_encoder

    @abstractmethod
    def encode(self, query, **kwargs):
        """Abstract method to encode queries."""
        pass
    
    def compute_kg_embedding(self, node_csv_without_emb, node_csv_file, edge_csv_without_emb, edge_csv_file, text_node_csv_without_emb, text_node_csv, **kwargs):
        with open(node_csv_without_emb, "r") as csvfile_node:
            with open(node_csv_file, "w", newline='') as csvfile_node_emb:
                reader_node = csv.reader(csvfile_node)

                # the reader has [name:ID,type,concepts,synsets,:LABEL]
                writer_node = csv.writer(csvfile_node_emb)
                writer_node.writerow(["name:ID", "type", "file_id", "concepts", "synsets", "embedding:STRING", ":LABEL"])

                # the encoding will be processed in batch of 2048
                batch_size = kwargs.get('batch_size', 2048)
                batch_nodes = []
                batch_rows = []
                for row in reader_node:
                    if row[0] == "name:ID":
                        continue
                    batch_nodes.append(row[0])
                    batch_rows.append(row)
                
                    if len(batch_nodes) == batch_size:
                        node_embeddings = self.encode(batch_nodes, batch_size=batch_size, show_progress_bar=False)
                        node_embedding_dict = dict(zip(batch_nodes, node_embeddings))
                        for row in batch_rows:
                        
                            new_row = [row[0], row[1], "", row[2], row[3], node_embedding_dict[row[0]].tolist(), row[4]]
                            writer_node.writerow(new_row)
                            
                        
                        
                        batch_nodes = []
                        batch_rows = []

                if len(batch_nodes) > 0:
                    node_embeddings = self.encode(batch_nodes, batch_size=batch_size, show_progress_bar=False)
                    node_embedding_dict = dict(zip(batch_nodes, node_embeddings))
                    for row in batch_rows:
                        new_row = [row[0], row[1], "", row[2], row[3], node_embedding_dict[row[0]].tolist(), row[4]]
                        writer_node.writerow(new_row)
                    batch_nodes = []
                    batch_rows = []
        

        with open(edge_csv_without_emb, "r") as csvfile_edge:
            with open(edge_csv_file, "w", newline='') as csvfile_edge_emb:
                reader_edge = csv.reader(csvfile_edge)
                # [":START_ID",":END_ID","relation","concepts","synsets",":TYPE"]
                writer_edge = csv.writer(csvfile_edge_emb)
                writer_edge.writerow([":START_ID", ":END_ID", "relation", "file_id", "concepts", "synsets", "embedding:STRING", ":TYPE"])

                # the encoding will be processed in batch of 4096
                batch_size = 2048
                batch_edges = []
                batch_rows = []
                for row in reader_edge:
                    if row[0] == ":START_ID":
                        continue
                    batch_edges.append(" ".join([row[0], row[2], row[1]]))
                    batch_rows.append(row)
                
                    if len(batch_edges) == batch_size:
                        edge_embeddings = self.encode(batch_edges, batch_size=batch_size, show_progress_bar=False)
                        edge_embedding_dict = dict(zip(batch_edges, edge_embeddings))
                        for row in batch_rows:
                            new_row = [row[0], row[1], row[2], "", row[3], row[4], edge_embedding_dict[" ".join([row[0], row[2], row[1]])].tolist(), row[5]]
                            writer_edge.writerow(new_row)
                        batch_edges = []
                        batch_rows = []

                if len(batch_edges) > 0:
                    edge_embeddings = self.encode(batch_edges, batch_size=batch_size, show_progress_bar=False)
                    edge_embedding_dict = dict(zip(batch_edges, edge_embeddings))
                    for row in batch_rows:
                        new_row = [row[0], row[1], row[2], "", row[3], row[4], edge_embedding_dict[" ".join([row[0], row[2], row[1]])].tolist(), row[5]]    
                        writer_edge.writerow(new_row)
                    batch_edges = []
                    batch_rows = []
        

        with open(text_node_csv_without_emb, "r") as csvfile_text_node:
            with open(text_node_csv, "w", newline='') as csvfile_text_node_emb:
                reader_text_node = csv.reader(csvfile_text_node)
                # [text_id:ID,original_text,:LABEL]
                writer_text_node = csv.writer(csvfile_text_node_emb)

                writer_text_node.writerow(["text_id:ID", "original_text", ":LABEL", "embedding:STRING"])

                # the encoding will be processed in batch of 2048
                batch_size = 2048
                batch_text_nodes = []
                batch_rows = []
                for row in reader_text_node:
                    if row[0] == "text_id:ID":
                        continue
                    
                    batch_text_nodes.append(row[1])
                    batch_rows.append(row)
                    if len(batch_text_nodes) == batch_size:
                        text_node_embeddings = self.encode(batch_text_nodes, batch_size=batch_size, show_progress_bar=False)
                        text_node_embedding_dict = dict(zip(batch_text_nodes, text_node_embeddings))
                        for row in batch_rows:
                            embedding  = text_node_embedding_dict[row[1]].tolist()
                            new_row = [row[0], row[1], row[2], embedding]
                            writer_text_node.writerow(new_row)

                        batch_text_nodes = []
                        batch_rows = []

                if len(batch_text_nodes) > 0:
                    text_node_embeddings = self.encode(batch_text_nodes, batch_size=batch_size, show_progress_bar=False)
                    text_node_embedding_dict = dict(zip(batch_text_nodes, text_node_embeddings))
                    for row in batch_rows:
                        embedding  = text_node_embedding_dict[row[1]].tolist()
                        new_row = [row[0], row[1], row[2], embedding]
                        
                        writer_text_node.writerow(new_row)
                    batch_text_nodes = []
                    batch_rows = []


class EmbeddingAPI(BaseEmbeddingModel):
    def __init__(self, emb_client: OpenAI, model_name="text-embedding-3-small"):
        """
        Initializes the embedding API with an OpenAI client.

        Args:
            emb_client: An OpenAI client instance.
            model_name: Name of the OpenAI embedding model to use.
        """
        self.emb_client = emb_client
        self.model_name = model_name

    def embed(self, input_texts: list) -> torch.Tensor:
        """
        Embeds the input texts using the OpenAI embedding client.

        Args:
            input_texts (list): A list of strings to embed.

        Returns:
            torch.Tensor: A tensor containing the embeddings for the input texts.
        """
        results = self.emb_client.embeddings.create(input=input_texts, model=self.model_name)
        embeddings = torch.tensor([d.embedding for d in results.data])
        return embeddings

    def compute_similarity(self, queries: list, documents: list) -> torch.Tensor:
        """
        Computes similarity scores between queries and documents.

        Args:
            queries (list): A list of query strings.
            documents (list): A list of document strings.

        Returns:
            torch.Tensor: A tensor containing similarity scores between queries and documents.
        """
        query_embeddings = self.embed(queries)
        document_embeddings = self.embed(documents)
        scores = query_embeddings @ document_embeddings.T
        return scores

    def encode(self, query, query_type=None, query_prefixes=None, **kwargs):
        """
        Encodes the input query or list of queries into vector representations.

        Args:
            query (str or list): The input query or queries to encode.
            query_type (str, optional): Used to select a prefix from query_prefixes.
            query_prefixes (dict, optional): Mapping from query_type to prefix string.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            np.ndarray: The encoded vector(s) as a NumPy array.
        """
        if query_prefixes is None:
            query_prefixes = kwargs.get('query_prefixes', {})

        # Determine prefix
        query_prefix = None
        if query_type is not None and query_prefixes and query_type in query_prefixes:
            query_prefix = query_prefixes[query_type]

        # Prepare input texts
        if query_prefix:
            if isinstance(query, list):
                input_texts = [query_prefix + q for q in query]
            else:
                input_texts = [query_prefix + query]
        else:
            if isinstance(query, list):
                input_texts = query
            else:
                input_texts = [query]

        embeddings = self.embed(input_texts)
        # Normalize if requested
        normalize_embeddings = kwargs.get('normalize_embeddings', True)
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()

class NvEmbed(BaseEmbeddingModel):
    def __init__(self, sentence_encoder: SentenceTransformer | AutoModel):
        self.sentence_encoder = sentence_encoder

    def add_eos(self, input_examples):
        """Add EOS token to input examples."""
        if self.sentence_encoder.tokenizer.eos_token is not None:
            return [input_example + self.sentence_encoder.tokenizer.eos_token for input_example in input_examples]
        else:
            return input_examples

    def encode(self, query, query_type=None, **kwargs):
        """
        Encode the query into embeddings.
        
        Args:
            query: Input text or list of texts.
            query_type: Type of query (e.g., 'passage', 'entity', 'edge', 'fill_in_edge', 'search').
            **kwargs: Additional arguments (e.g., normalize_embeddings).
        
        Returns:
            Embeddings as a NumPy array.
        """
        normalize_embeddings = kwargs.get('normalize_embeddings', True)

        # Define prompt prefixes based on query type
        prompt_prefixes = {
            'passage': 'Given a question, retrieve relevant documents that best answer the question.',
            'entity': 'Given a question, retrieve relevant phrases that are mentioned in this question.',
            'edge': 'Given a question, retrieve relevant triplet facts that matches this question.',
            'fill_in_edge': 'Given a triples with only head and relation, retrieve relevant triplet facts that best fill the atomic query.'
        }

        if query_type in prompt_prefixes:
            prompt_prefix = prompt_prefixes[query_type]
            query_prefix = f"Instruct: {prompt_prefix}\nQuery: "
        else:
            query_prefix = None

        # Encode the query
        if issubclass(type(self.sentence_encoder), SentenceTransformer):
            if query_prefix:
                query_embeddings = self.sentence_encoder.encode(self.add_eos(query), prompt=query_prefix, **kwargs)
            else:
                query_embeddings = self.sentence_encoder.encode(self.add_eos(query), **kwargs)
        elif issubclass(type(self.sentence_encoder), AutoModel):
            if query_prefix:
                query_embeddings = self.sentence_encoder.encode(query, instruction=query_prefix, max_length = 32768, **kwargs)
            else:
                query_embeddings = self.sentence_encoder.encode(query, max_length = 32768, **kwargs)

            # Normalize embeddings if required
            if normalize_embeddings:
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1).detach().cpu().numpy()

        # Move to CPU and convert to NumPy
        return query_embeddings

class SentenceEmbedding(BaseEmbeddingModel):
    def __init__(self, sentence_encoder: SentenceTransformer):
        self.sentence_encoder = sentence_encoder

    def encode(self, query, **kwargs):
        """
        Encode the query into embeddings.
        
        Args:
            query: Input text or list of texts.
            **kwargs: Additional arguments (filters out unsupported kwargs).
        
        Returns:
            Embeddings as a NumPy array.
        """

        # Check if the model accepts additional kwargs by inspecting get_model_kwargs
        if hasattr(self.sentence_encoder, 'get_model_kwargs'):
            try:
                # Get the list of accepted kwargs from the model
                model_kwargs = self.sentence_encoder.get_model_kwargs()
                if model_kwargs is not None:
                    # Only pass kwargs that are in model_kwargs
                    kwargs = {k: v for k, v in kwargs.items() if k in model_kwargs}
            except Exception:
                # If we can't determine supported kwargs, use filtered_kwargs as is
                available_kwargs_key = ['normalize_embeddings', 'show_progress_bar', 'batch_size']
                kwargs = {k: v for k, v in kwargs.items() if k in available_kwargs_key}
        
        return self.sentence_encoder.encode(query, **kwargs)
    

class Qwen3Emb(BaseEmbeddingModel):
    def __init__(self, emb_client: OpenAI, model_name="Qwen/Qwen3-Embedding-0.6B"):
        """
        Initializes the Reranker with an async embedding client.

        Args:
            emb_client: An async embedding client instance (e.g., AsyncOpenAI client).
            model_name: Name of the embedding model to use.
        """
        self.emb_client = emb_client
        self.model_name = model_name

    def embed(self, input_texts: list) -> torch.Tensor:
        """
        Embeds the input texts using the async embedding client, with automatic batching.
        
        Args:
            input_texts (list): A list of strings to embed.
        
        Returns:
            torch.Tensor: A tensor containing the embeddings for the input texts.
        """
        # Set a safe batch size (API limit is 1024, using 512 to be conservative)
        batch_size = 512
        all_embeddings = []
        
        # Process input texts in batches to avoid API limits
        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i:i+batch_size]
            try:
                results = self.emb_client.embeddings.create(input=batch, model=self.model_name)
                batch_embeddings = [d.embedding for d in results.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error embedding batch {i//batch_size}: {e}")
                # Re-raise the exception to maintain original behavior
                raise
        
        # Convert results to tensor
        return torch.tensor(all_embeddings)

    def compute_similarity(self, queries: list, documents: list) -> torch.Tensor:
        """
        Computes similarity scores between queries and documents.

        Args:
            queries (list): A list of query strings.
            documents (list): A list of document strings.

        Returns:
            torch.Tensor: A tensor containing similarity scores between queries and documents.
        """
        input_texts = queries + documents
        embeddings = self.embed(input_texts)
        query_embeddings = embeddings[:len(queries)]
        document_embeddings = embeddings[len(queries):]
        scores = query_embeddings @ document_embeddings.T
        return scores

    def encode(self, query, query_type=None, **kwargs):
        """
        Encodes the input query into a vector representation.

        Args:
            query (str): The input query to encode.
            query_type (str, optional): The type of query (e.g., "passage", "entity", "edge").
            **kwargs: Additional keyword arguments to pass to the embedding model.

        Returns:
            torch.Tensor: The encoded vector representation of the query.
        """
        normalize_embeddings = kwargs.get('normalize_embeddings', True)

        # Define prompt prefixes based on query type
        prompt_prefixes = {
            'passage': 'Given a question, retrieve relevant documents that best answer the question.',
            'entity': 'Given a question, retrieve relevant phrases that are mentioned in this question.',
            'edge': 'Given a question, retrieve relevant triplet facts that matches this question.',
            'fill_in_edge': 'Given a triples with only head and relation, retrieve relevant triplet facts that best fill the atomic query.'
        }

        if query_type in prompt_prefixes:
            prompt_prefix = prompt_prefixes[query_type]
            query_prefix = f"Instruct: {prompt_prefix}\nQuery: "
        else:
            query_prefix = None

        # Encode the query
        if isinstance(query, list):
            input_texts = [query_prefix + q if query_prefix else q for q in query]
        else:
            input_texts = [query_prefix + query] if query_prefix else [query]
        query_embeddings = self.embed(input_texts)
        if normalize_embeddings:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        # Move to CPU and convert to NumPy
        return query_embeddings.detach().cpu().numpy()
