import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch
from atlas_rag.retrieval.retriever.simple_retriever import SimpleGraphRetriever, SimpleTextRetriever
from atlas_rag.retrieval.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import LLMGenerator

@pytest.fixture
def mock_sentence_encoder():
    encoder = Mock(spec=BaseEmbeddingModel)
    encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    return encoder

@pytest.fixture
def mock_llm_generator():
    return Mock(spec=LLMGenerator)

@pytest.fixture
def sample_graph_data():
    # Create a sample graph
    G = nx.DiGraph()
    G.add_node("node1", id="Entity 1", type="entity")
    G.add_node("node2", id="Entity 2", type="entity")
    G.add_edge("node1", "node2", relation="related_to")
    
    # Create mock FAISS indices
    mock_node_index = Mock()
    mock_edge_index = Mock()
    mock_node_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
    mock_edge_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
    
    return {
        "KG": G,
        "node_list": ["node1", "node2"],
        "edge_list": [("node1", "node2")],
        "node_faiss_index": mock_node_index,
        "edge_faiss_index": mock_edge_index
    }

@pytest.fixture
def sample_text_data():
    passage_dict = {
        "doc1": "This is the first passage",
        "doc2": "This is the second passage",
        "doc3": "This is the third passage"
    }
    
    # Create mock embeddings
    mock_embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    
    return {
        "passage_dict": passage_dict,
        "text_embeddings": mock_embeddings
    }

def test_simple_graph_retriever_initialization(mock_sentence_encoder, mock_llm_generator, sample_graph_data):
    retriever = SimpleGraphRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data
    )
    
    assert retriever.KG == sample_graph_data["KG"]
    assert retriever.node_list == sample_graph_data["node_list"]
    assert retriever.edge_list == sample_graph_data["edge_list"]
    assert retriever.llm_generator == mock_llm_generator
    assert retriever.sentence_encoder == mock_sentence_encoder
    assert retriever.node_faiss_index == sample_graph_data["node_faiss_index"]
    assert retriever.edge_faiss_index == sample_graph_data["edge_faiss_index"]

def test_simple_graph_retriever_retrieve(mock_sentence_encoder, mock_llm_generator, sample_graph_data):
    retriever = SimpleGraphRetriever(
        llm_generator=mock_llm_generator,
        sentence_encoder=mock_sentence_encoder,
        data=sample_graph_data
    )
    
    query = "test query"
    topk = 2
    
    edges, scores = retriever.retrieve(query, topk=topk)
    
    # Verify that encode was called with correct parameters
    mock_sentence_encoder.encode.assert_called_once_with([query], query_type='edge')
    
    # Verify that search was called on the edge index
    sample_graph_data["edge_faiss_index"].search.assert_called_once()
    
    # Verify the output format
    assert len(edges) == 1  # We only have one edge in our sample data
    assert len(scores) == 1
    assert all(score == "N/A" for score in scores)

def test_simple_text_retriever_initialization(mock_sentence_encoder, sample_text_data):
    retriever = SimpleTextRetriever(
        passage_dict=sample_text_data["passage_dict"],
        sentence_encoder=mock_sentence_encoder,
        data={"text_embeddings": sample_text_data["text_embeddings"]}
    )
    
    assert retriever.sentence_encoder == mock_sentence_encoder
    assert retriever.passage_dict == sample_text_data["passage_dict"]
    assert retriever.passage_list == list(sample_text_data["passage_dict"].values())
    assert retriever.passage_keys == list(sample_text_data["passage_dict"].keys())
    assert np.array_equal(retriever.text_embeddings, sample_text_data["text_embeddings"])

def test_simple_text_retriever_retrieve(mock_sentence_encoder, sample_text_data):
    retriever = SimpleTextRetriever(
        passage_dict=sample_text_data["passage_dict"],
        sentence_encoder=mock_sentence_encoder,
        data={"text_embeddings": sample_text_data["text_embeddings"]}
    )
    
    query = "test query"
    topk = 2
    
    passages, passage_ids = retriever.retrieve(query, topk=topk)
    
    # Verify that encode was called with correct parameters
    mock_sentence_encoder.encode.assert_called_once_with([query], query_type="passage")
    
    # Verify the output format
    assert len(passages) == topk
    assert len(passage_ids) == topk
    assert all(passage in sample_text_data["passage_dict"].values() for passage in passages)
    assert all(passage_id in sample_text_data["passage_dict"].keys() for passage_id in passage_ids) 