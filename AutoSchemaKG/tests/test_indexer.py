import pytest
import numpy as np
import networkx as nx
import os
import tempfile
import shutil
import faiss
from atlas_rag.retrieval.indexer import (
    compute_graph_embeddings,
    build_faiss_index,
    compute_text_embeddings,
    create_embeddings_and_index
)
from atlas_rag.retrieval.embedding_model import NvEmbed
from sentence_transformers import SentenceTransformer

@pytest.fixture
def sentence_encoder():
    return NvEmbed(SentenceTransformer('all-MiniLM-L6-v2'))

@pytest.fixture
def sample_graph():
    G = nx.DiGraph()
    # Add nodes with different types
    # Add multiple entities
    G.add_node("node1", type="entity", id="Entity 1")
    G.add_node("node2", type="entity", id="Entity 2")
    G.add_node("node3", type="entity", id="Entity 3")
    
    # Add multiple events
    G.add_node("node4", type="event", id="Event 1")
    G.add_node("node5", type="event", id="Event 2")
    
    # Add multiple passages
    G.add_node("node6", type="passage", id="Passage 1")
    G.add_node("node7", type="passage", id="Passage 2")
    
    # Add multiple concepts
    G.add_node("node8", type="concept", id="Concept 1")
    G.add_node("node9", type="concept", id="Concept 2")
    
    # Add edges connecting different types of nodes
    # Entity to Entity edges
    G.add_edge("node1", "node2", relation="related_to")
    G.add_edge("node2", "node3", relation="connected_to")
    
    # Entity to Event edges
    G.add_edge("node1", "node4", relation="causes")
    G.add_edge("node2", "node5", relation="triggers")
    
    # Event to Passage edges
    G.add_edge("node4", "node6", relation="described_in")
    G.add_edge("node5", "node7", relation="documented_in")
    
    # Passage to Concept edges
    G.add_edge("node6", "node8", relation="about")
    G.add_edge("node7", "node9", relation="related_to")
    
    # Add some bidirectional edges
    G.add_edge("node3", "node1", relation="references")
    G.add_edge("node9", "node6", relation="appears_in")
    
    return G

@pytest.fixture
def temp_working_dir():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)

def test_compute_graph_embeddings(sentence_encoder):
    node_list = ["Node 1", "Node 2", "Node 3"]
    edge_list = ["Node 1 relation Node 2", "Node 2 relation Node 3"]
    
    node_embeddings, edge_embeddings = compute_graph_embeddings(
        node_list, edge_list, sentence_encoder, batch_size=2
    )
    
    assert len(node_embeddings) == len(node_list)
    assert len(edge_embeddings) == len(edge_list)
    assert all(isinstance(emb, np.ndarray) for emb in node_embeddings)
    assert all(isinstance(emb, np.ndarray) for emb in edge_embeddings)

def test_build_faiss_index():
    # Create sample embeddings
    embeddings = [np.random.rand(384).astype('float32') for _ in range(5)]
    
    index = build_faiss_index(embeddings)
    assert isinstance(index, faiss.Index)  # FAISS index type

def test_compute_text_embeddings(sentence_encoder):
    text_list = ["Text 1", "Text 2", "Text 3"]
    
    embeddings = compute_text_embeddings(text_list, sentence_encoder, batch_size=2)
    
    assert len(embeddings) == len(text_list)
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)

def test_create_embeddings_and_index(sentence_encoder, sample_graph, temp_working_dir):
    # Create necessary directory structure
    os.makedirs(os.path.join(temp_working_dir, "kg_graphml"), exist_ok=True)
    os.makedirs(os.path.join(temp_working_dir, "precompute"), exist_ok=True)
    
    # Save the sample graph
    graph_path = os.path.join(temp_working_dir, "kg_graphml", "test_graph.graphml")
    nx.write_graphml(sample_graph, graph_path)
    
    # Test with different combinations of include_events and include_concept
    test_cases = [
        (False, False),  # Only entities
        (True, False),   # Events and entities
        (True, True),    # Events, concepts, and entities
    ]
    
    for include_events, include_concept in test_cases:
        result = create_embeddings_and_index(
            sentence_encoder,
            model_name="test_model",
            working_directory=temp_working_dir,
            keyword="test",
            include_events=include_events,
            include_concept=include_concept,
            normalize_embeddings=True,
            text_batch_size=2,
            node_and_edge_batch_size=2
        )
        
        # Check if all required components are present
        assert "KG" in result
        assert "node_faiss_index" in result
        assert "edge_faiss_index" in result
        assert "text_faiss_index" in result
        assert "node_embeddings" in result
        assert "edge_embeddings" in result
        assert "text_embeddings" in result
        assert "node_list" in result
        assert "edge_list" in result
        assert "text_dict" in result
        
        # Verify that we have the expected number of nodes based on the configuration
        if not include_events and not include_concept:
            # Should only have entity nodes
            assert all(sample_graph.nodes[node]["type"] == "entity" for node in result["node_list"])
        elif include_events and not include_concept:
            # Should have entity and event nodes
            assert all(sample_graph.nodes[node]["type"] in ["entity", "event"] for node in result["node_list"])
        else:
            # Should have entity, event, and concept nodes
            assert all(sample_graph.nodes[node]["type"] in ["entity", "event", "concept"] for node in result["node_list"])
        
        # Verify that we have edges and edge embeddings
        assert len(result["edge_list"]) > 0, "Edge list should not be empty"
        assert len(result["edge_embeddings"]) > 0, "Edge embeddings should not be empty"
        assert len(result["edge_list"]) == len(result["edge_embeddings"]), "Number of edges should match number of edge embeddings"
        
        # Verify edge structure
        for edge in result["edge_list"]:
            source, target = edge
            assert source in result["node_list"], f"Source node {source} should be in node_list"
            assert target in result["node_list"], f"Target node {target} should be in node_list"
            assert sample_graph.has_edge(source, target), f"Edge {source}->{target} should exist in original graph"

def test_create_embeddings_and_index_invalid_combination(sentence_encoder, sample_graph, temp_working_dir):
    # Create necessary directory structure
    os.makedirs(os.path.join(temp_working_dir, "kg_graphml"), exist_ok=True)
    os.makedirs(os.path.join(temp_working_dir, "precompute"), exist_ok=True)
    
    # Save the sample graph
    graph_path = os.path.join(temp_working_dir, "kg_graphml", "test_graph.graphml")
    nx.write_graphml(sample_graph, graph_path)
    
    # Test invalid combination (False, True)
    with pytest.raises(ValueError):
        create_embeddings_and_index(
            sentence_encoder,
            model_name="test_model",
            working_directory=temp_working_dir,
            keyword="test",
            include_events=False,
            include_concept=True,
            normalize_embeddings=True
        ) 