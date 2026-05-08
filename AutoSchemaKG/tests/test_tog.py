import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch
from atlas_rag.retrieval.retriever.tog import TogRetriever
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
    
    def mock_generate_response(messages):
        # Check the content of the messages to determine what type of response to return
        content = messages[1]['content']  # Get the user's message content
        
        if "rating" in content.lower():
            # For path rating requests
            return "4"
        elif "reasoning" in content.lower():
            # For reasoning requests
            return "yes"
        else:
            # For NER requests
            return "entity1, entity2"
    
    # Set up the mock to use our custom function
    generator._generate_response.side_effect = mock_generate_response
    return generator

@pytest.fixture
def sample_graph_data():
    # Create a sample graph
    G = nx.DiGraph()
    
    # Add nodes with different types
    G.add_node("node1", id="Entity 1", type="entity")
    G.add_node("node2", id="Entity 2", type="entity")
    G.add_node("node3", id="Entity 3", type="entity")
    
    # Add edges
    G.add_edge("node1", "node2", relation="related_to")
    G.add_edge("node2", "node3", relation="connected_to")
    
    # Create mock embeddings
    mock_node_embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    
    mock_edge_embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ])
    
    return {
        "KG": G,
        "node_embeddings": mock_node_embeddings,
        "edge_embeddings": mock_edge_embeddings
    }

@pytest.fixture
def inference_config():
    return InferenceConfig(
        Dmax=2,
        topk=3
    )

def test_tog_retriever_initialization(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = TogRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    assert retriever.KG == sample_graph_data["KG"]
    assert retriever.llm_generator == mock_llm_generator
    assert retriever.sentence_encoder == mock_sentence_encoder
    assert np.array_equal(retriever.node_embeddings, sample_graph_data["node_embeddings"])
    assert np.array_equal(retriever.edge_embeddings, sample_graph_data["edge_embeddings"])
    assert retriever.inference_config == inference_config
    assert len(retriever.edge_list_with_relation) == len(sample_graph_data["KG"].edges)
    assert len(retriever.edge_list_string) == len(sample_graph_data["KG"].edges)

def test_ner(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = TogRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    text = "test query"
    entities = retriever.ner(text)
    
    # Verify that _generate_response was called with correct messages
    mock_llm_generator._generate_response.assert_called()
    assert entities == "entity1, entity2"

def test_retrieve_topk_nodes(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = TogRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    topN = 2
    
    topk_nodes = retriever.retrieve_topk_nodes(query, topN=topN)
    
    # Verify that NER was called
    mock_llm_generator._generate_response.assert_called()
    
    # Verify that encode was called
    mock_sentence_encoder.encode.assert_called()
    
    # Verify the output format
    assert isinstance(topk_nodes, list)
    assert len(topk_nodes) <= 2 * topN
    assert all(node in sample_graph_data["KG"].nodes for node in topk_nodes)

def test_search(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = TogRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    initial_paths = [["node1"]]
    
    new_paths = retriever.search(query, initial_paths)
    
    # Verify the output format
    assert isinstance(new_paths, list)
    assert all(isinstance(path, list) for path in new_paths)
    assert all(len(path) >= 1 for path in new_paths)
    assert all(all(node in sample_graph_data["KG"].nodes for node in path[::2]) for path in new_paths)
    assert all(all(rel in ["related_to", "connected_to"] for rel in path[1::2]) for path in new_paths)

def test_prune(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = TogRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    paths = [
        ["node1", "related_to", "node2"],
        ["node2", "connected_to", "node3"]
    ]
    topN = 2
    
    pruned_paths = retriever.prune(query, paths, topN=topN)
    
    # Verify that _generate_response was called for each path
    assert mock_llm_generator._generate_response.call_count >= len(paths)
    
    # Verify the output format
    assert isinstance(pruned_paths, list)
    assert len(pruned_paths) <= topN
    assert all(isinstance(path, list) for path in pruned_paths)

def test_reasoning(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = TogRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    paths = [
        ["node1", "related_to", "node2"],
        ["node2", "connected_to", "node3"]
    ]
    
    result = retriever.reasoning(query, paths)
    
    # Verify that _generate_response was called
    mock_llm_generator._generate_response.assert_called()
    
    # Verify the output format
    assert isinstance(result, bool)

def test_generate(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = TogRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    paths = [
        ["node1", "related_to", "node2"],
        ["node2", "connected_to", "node3"]
    ]
    
    triples, scores = retriever.generate(query, paths)
    
    # Verify the output format
    assert isinstance(triples, list)
    assert isinstance(scores, list)
    assert len(triples) == len(scores)
    assert all(isinstance(triple, str) for triple in triples)
    assert all(score == "N/A" for score in scores)

def test_retrieve(mock_sentence_encoder, mock_llm_generator, sample_graph_data, inference_config):
    retriever = TogRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data,
        inference_config=inference_config
    )
    
    query = "test query"
    topN = 2
    
    triples, scores = retriever.retrieve(query, topN=topN)
    
    # Verify the output format
    assert isinstance(triples, list)
    assert isinstance(scores, list)
    assert len(triples) == len(scores)
    assert all(isinstance(triple, str) for triple in triples)
    assert all(score == "N/A" for score in scores) 