from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseRetriever(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    @abstractmethod
    def retrieve(self, query, topk=5, **kwargs) -> Tuple[List[str], List[str]]:
        raise NotImplementedError("This method should be overridden by subclasses.") 

class BaseEdgeRetriever(BaseRetriever):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    @abstractmethod
    def retrieve(self, query, topk=5, **kwargs) -> Tuple[List[str], List[str]]:
        raise NotImplementedError("This method should be overridden by subclasses.")

class BasePassageRetriever(BaseRetriever):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    @abstractmethod
    def retrieve(self, query, topk=5, **kwargs) -> Tuple[List[str], List[str]]:
        raise NotImplementedError("This method should be overridden by subclasses.")

