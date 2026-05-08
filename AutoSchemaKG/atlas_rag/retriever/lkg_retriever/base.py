from abc import ABC, abstractmethod


class BaseLargeKGRetriever(ABC):
    def __init__():
        raise NotImplementedError("This is a base class and cannot be instantiated directly.")
    @abstractmethod
    def retrieve_passages(self, query, retriever_config:dict):
        """
        Retrieve passages based on the query.
        
        Args:
            query (str): The input query.
            topN (int): Number of top passages to retrieve.
            number_of_source_nodes_per_ner (int): Number of source nodes per named entity recognition.
            sampling_area (int): Area for sampling in the graph.
        
        Returns:
            List of retrieved passages and their scores.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class BaseLargeKGEdgeRetriever(ABC):
    def __init__():
        raise NotImplementedError("This is a base class and cannot be instantiated directly.")
    @abstractmethod
    def retrieve_passages(self, query, retriever_config:dict):
        """
        Retrieve Edges / Paths based on the query.
        
        Args:
            query (str): The input query.
            topN (int): Number of top passages to retrieve.
            number_of_source_nodes_per_ner (int): Number of source nodes per named entity recognition.
            sampling_area (int): Area for sampling in the graph.
        
        Returns:
            List of retrieved passages and their scores.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")