import pytest
from atlas_rag.retrieval.retriever.base import (
    InferenceConfig,
    BaseRetriever,
    BaseEdgeRetriever,
    BasePassageRetriever
)

def test_inference_config_default_values():
    config = InferenceConfig()
    assert config.keyword == "musique"
    assert config.topk == 5
    assert config.Dmax == 4
    assert config.weight_adjust == 1.0
    assert config.topk_edges == 50
    assert config.topk_nodes == 10
    assert config.ppr_alpha == 0.99
    assert config.ppr_max_iter == 2000
    assert config.ppr_tol == 1e-7

def test_inference_config_custom_values():
    config = InferenceConfig(
        keyword="test",
        topk=10,
        Dmax=5,
        weight_adjust=0.5,
        topk_edges=100,
        topk_nodes=20,
        ppr_alpha=0.95,
        ppr_max_iter=1000,
        ppr_tol=1e-6
    )
    assert config.keyword == "test"
    assert config.topk == 10
    assert config.Dmax == 5
    assert config.weight_adjust == 0.5
    assert config.topk_edges == 100
    assert config.topk_nodes == 20
    assert config.ppr_alpha == 0.95
    assert config.ppr_max_iter == 1000
    assert config.ppr_tol == 1e-6

def test_base_retriever_initialization():
    class TestRetriever(BaseRetriever):
        def retrieve(self, query, topk=5, **kwargs):
            return [], []
    
    retriever = TestRetriever(test_param="value")
    assert hasattr(retriever, "test_param")
    assert retriever.test_param == "value"

def test_base_edge_retriever_initialization():
    class TestEdgeRetriever(BaseEdgeRetriever):
        def retrieve(self, query, topk=5, **kwargs):
            return [], []
    
    retriever = TestEdgeRetriever(test_param="value")
    assert hasattr(retriever, "test_param")
    assert retriever.test_param == "value"

def test_base_passage_retriever_initialization():
    class TestPassageRetriever(BasePassageRetriever):
        def retrieve(self, query, topk=5, **kwargs):
            return [], []
    
    retriever = TestPassageRetriever(test_param="value")
    assert hasattr(retriever, "test_param")
    assert retriever.test_param == "value"