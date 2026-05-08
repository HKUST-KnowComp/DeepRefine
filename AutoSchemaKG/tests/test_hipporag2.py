import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch
from atlas_rag.retrieval.retriever.hipporag2 import HippoRAG2Retriever, min_max_normalize
from atlas_rag.retrieval.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from atlas_rag.retrieval.retriever.base import InferenceConfig

@pytest.fixture
def mock_sentence_encoder():
    encoder = Mock(spec=BaseEmbeddingModel)
    # Return an embedding that will give good scores for both passages
    encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    return encoder

@pytest.fixture
def mock_llm_generator():
    generator = Mock(spec=LLMGenerator)
    generator.ner.return_value = "entity1, entity2"
    generator.filter_triples_with_entity_event.return_value = [
        ["Entity 1", "relation", "Entity 2"]
    ]
    return generator

@pytest.fixture
def sample_graph_data():
    # Create a sample graph
    G = nx.DiGraph()
    
    # Add nodes with different types
    G.add_node("node1", id="Entity 1", type="entity", file_id="file1")
    G.add_node("node2", id="Entity 2", type="entity", file_id="file2")
    G.add_node("node3", id="Passage 1", type="passage", file_id="file1")
    G.add_node("node4", id="Passage 2", type="passage", file_id="file2")
    
    # Add edges
    G.add_edge("node1", "node2", relation="related_to")
    G.add_edge("node2", "node3", relation="described_in")
    G.add_edge("node3", "node4", relation="references")
    
    # Create mock embeddings
    mock_node_embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4]
    ])
    
    mock_edge_embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    
    # Create text embeddings that will give good scores for both passages
    mock_text_embeddings = np.array([
        [0.1, 0.2, 0.3],  # First passage embedding
        [0.1, 0.2, 0.3]   # Second passage embedding - same as first to ensure both get good scores
    ])
    
    # Create mock FAISS index that returns two results
    mock_edge_index = Mock()
    mock_edge_index.search.return_value = (
        np.array([[0.9, 0.8]]),  # Two similarity scores
        np.array([[0, 1]])       # Two indices
    )
    
    return {
        "KG": G,
        "node_list": ["node1", "node2", "node3", "node4"],
        "edge_list": [("node1", "node2"), ("node2", "node3"), ("node3", "node4")],
        "node_embeddings": mock_node_embeddings,
        "edge_embeddings": mock_edge_embeddings,
        "text_embeddings": mock_text_embeddings,
        "edge_faiss_index": mock_edge_index,
        "text_dict": {
            "node3": "This is passage 1",
            "node4": "This is passage 2"
        }
    }

@pytest.fixture
def inference_config():
    return InferenceConfig(
        topk_edges=2,
        weight_adjust=0.05,
        ppr_alpha=0.85,
        ppr_max_iter=100,
        ppr_tol=1e-6
    )

def test_min_max_normalize():
    # Test with normal case
    x = np.array([1, 2, 3, 4, 5])
    normalized = min_max_normalize(x)
    assert np.min(normalized) == 0
    assert np.max(normalized) == 1
    
    # Test with all same values
    x = np.array([1, 1, 1, 1])
    normalized = min_max_normalize(x)
    assert np.all(normalized == 1)

def test_hipporag2_retriever_initialization(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = HippoRAG2Retriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    assert retriever.llm_generator == mock_llm_generator
    assert retriever.sentence_encoder == mock_sentence_encoder
    assert np.array_equal(retriever.node_embeddings, sample_graph_data["node_embeddings"])
    assert retriever.node_list == sample_graph_data["node_list"]
    assert retriever.edge_list == sample_graph_data["edge_list"]
    assert np.array_equal(retriever.edge_embeddings, sample_graph_data["edge_embeddings"])
    assert np.array_equal(retriever.text_embeddings, sample_graph_data["text_embeddings"])
    assert retriever.edge_faiss_index == sample_graph_data["edge_faiss_index"]
    assert retriever.passage_dict == sample_graph_data["text_dict"]
    assert retriever.inference_config == inference_config
    assert not retriever.logging

def test_ner2node(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = HippoRAG2Retriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    topN = 2
    
    node_score_dict = retriever.ner2node(query, topN=topN)
    
    # Verify that NER was called
    mock_llm_generator.ner.assert_called_once_with(query)
    
    # Verify that encode was called
    assert mock_sentence_encoder.encode.call_count >= 1
    
    # Verify the output format
    assert isinstance(node_score_dict, dict)
    assert all(isinstance(node, str) for node in node_score_dict.keys())
    assert all(isinstance(score, float) for score in node_score_dict.values())

def test_query2node(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = HippoRAG2Retriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    topN = 2
    
    node_score_dict = retriever.query2node(query, topN=topN)
    
    # Verify that encode was called
    mock_sentence_encoder.encode.assert_called_once_with([query], query_type="entity")
    
    # Verify the output format
    assert isinstance(node_score_dict, dict)
    assert len(node_score_dict) <= topN
    assert all(isinstance(node, str) for node in node_score_dict.keys())
    assert all(isinstance(score, float) for score in node_score_dict.values())

def test_query2edge(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = HippoRAG2Retriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    topN = 2
    
    node_score_dict = retriever.query2edge(query, topN=topN)
    
    # Verify that encode was called
    assert mock_sentence_encoder.encode.call_count >= 1
    
    # Verify that filter_triples_with_entity_event was called
    mock_llm_generator.filter_triples_with_entity_event.assert_called_once()
    
    # Verify the output format
    assert isinstance(node_score_dict, dict)
    assert all(isinstance(node, str) for node in node_score_dict.keys())
    assert all(isinstance(score, float) for score in node_score_dict.values())

def test_query2passage(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = HippoRAG2Retriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    weight_adjust = 0.05
    
    passage_scores = retriever.query2passage(query, weight_adjust=weight_adjust)
    
    # Verify that encode was called
    mock_sentence_encoder.encode.assert_called_once_with([query], query_type="passage")
    
    # Verify the output format
    assert isinstance(passage_scores, dict)
    assert all(isinstance(passage_id, str) for passage_id in passage_scores.keys())
    assert all(isinstance(score, float) for score in passage_scores.values())
    assert all(0 <= score <= weight_adjust for score in passage_scores.values())

def test_retrieve(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = HippoRAG2Retriever(
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
    assert all(passage_id in ["Passage 1", "Passage 2"] for passage_id in passage_ids) 