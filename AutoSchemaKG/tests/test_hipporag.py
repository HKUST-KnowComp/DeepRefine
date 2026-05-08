import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch
from atlas_rag.retrieval.retriever.hipporag import HippoRAGRetriever
from atlas_rag.retrieval.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from atlas_rag.retrieval.retriever.base import InferenceConfig

@pytest.fixture
def mock_sentence_encoder():
    encoder = Mock(spec=BaseEmbeddingModel)
    encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    return encoder

@pytest.fixture
def mock_llm_generator():
    generator = Mock(spec=LLMGenerator)
    generator.ner.return_value = "entity1, entity2"
    return generator

@pytest.fixture
def sample_graph_data():
    # Create a sample graph
    G = nx.DiGraph()
    
    # Add nodes with different types
    G.add_node("node1", id="Entity 1", type="entity", file_id="file1")
    G.add_node("node2", id="Entity 2", type="entity", file_id="file1")
    G.add_node("node3", id="Passage 1", type="passage", file_id="file1")
    G.add_node("node4", id="Passage 2", type="passage", file_id="file2")
    
    # Add edges
    G.add_edge("node1", "node2", relation="related_to")
    G.add_edge("node2", "node3", relation="described_in")
    G.add_edge("node3", "node4", relation="references")
    
    # Create mock embeddings
    mock_embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4]
    ])
    
    return {
        "KG": G,
        "node_list": ["node1", "node2", "node3", "node4"],
        "node_embeddings": mock_embeddings,
        "text_dict": {
            "node3": "This is passage 1",
            "node4": "This is passage 2"
        }
    }

@pytest.fixture
def inference_config():
    return InferenceConfig(
        topk_nodes=2,
        ppr_alpha=0.85,
        ppr_max_iter=100,
        ppr_tol=1e-6
    )

def test_hipporag_retriever_initialization(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = HippoRAGRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    assert retriever.passage_dict == sample_graph_data["text_dict"]
    assert retriever.llm_generator == mock_llm_generator
    assert retriever.sentence_encoder == mock_sentence_encoder
    assert np.array_equal(retriever.node_embeddings, sample_graph_data["node_embeddings"])
    assert retriever.node_list == sample_graph_data["node_list"]
    assert retriever.inference_config == inference_config
    assert not retriever.logging

def test_retrieve_personalization_dict(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = HippoRAGRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    topN = 2
    
    personalization_dict = retriever.retrieve_personalization_dict(query, topN=topN)
    
    # Verify that NER was called
    mock_llm_generator.ner.assert_called_once_with(query)
    
    # Verify that encode was called for each entity
    assert mock_sentence_encoder.encode.call_count >= 1
    
    # Verify the personalization dictionary
    assert isinstance(personalization_dict, dict)
    assert all(isinstance(node, str) for node in personalization_dict.keys())
    assert all(isinstance(weight, float) for weight in personalization_dict.values())

def test_retrieve(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = HippoRAGRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    topN = 2
    
    passages, passage_ids = retriever.retrieve(query, topN=topN)
    
    # Verify the output format
    assert len(passages) == topN
    assert len(passage_ids) == topN
    assert all(passage in sample_graph_data["text_dict"].values() for passage in passages)
    assert all(passage_id in sample_graph_data["text_dict"].keys() for passage_id in passage_ids)

def test_retrieve_with_logger(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    mock_logger = Mock()
    retriever = HippoRAGRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config,
        logger=mock_logger
    )
    
    assert retriever.logging
    assert retriever.logger == mock_logger
    
    query = "test query"
    topN = 2
    
    retriever.retrieve_personalization_dict(query, topN=topN)
    
    # Verify that logger was used
    assert mock_logger.info.call_count >= 1 