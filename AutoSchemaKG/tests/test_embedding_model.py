import pytest
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import numpy as np
from atlas_rag.retrieval.embedding_model import NvEmbed, SentenceEmbedding

@pytest.fixture
def sentence_transformer():
    # Use a small model for testing
    return SentenceTransformer('all-MiniLM-L6-v2')

@pytest.fixture
def auto_model():
    # Use a small model for testing
    return AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def test_nv_embed_init(sentence_transformer, auto_model):
    # Test initialization with SentenceTransformer
    nv_embed_st = NvEmbed(sentence_transformer)
    assert nv_embed_st.sentence_encoder == sentence_transformer

    # Test initialization with AutoModel
    nv_embed_am = NvEmbed(auto_model)
    assert nv_embed_am.sentence_encoder == auto_model

def test_nv_embed_add_eos(sentence_transformer):
    nv_embed = NvEmbed(sentence_transformer)
    test_inputs = ["test query 1", "test query 2"]
    result = nv_embed.add_eos(test_inputs)
    
    # Check that EOS token is added to each input
    assert len(result) == len(test_inputs)
    if sentence_transformer.tokenizer.eos_token is not None:
        assert all(result[i].endswith(sentence_transformer.tokenizer.eos_token) for i in range(len(result)))

def test_nv_embed_encode(sentence_transformer):
    nv_embed = NvEmbed(sentence_transformer)
    test_query = "test query"
    
    # Test encoding with different query types
    query_types = ['passage', 'entity', 'edge', 'fill_in_edge', None]
    for query_type in query_types:
        embeddings = nv_embed.encode(test_query, query_type=query_type)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 1  # Single query should return 1D array

def test_sentence_embedding_init(sentence_transformer):
    sent_embed = SentenceEmbedding(sentence_transformer)
    assert sent_embed.sentence_encoder == sentence_transformer

def test_sentence_embedding_encode(sentence_transformer):
    sent_embed = SentenceEmbedding(sentence_transformer)
    test_query = "test query"
    
    # Test encoding with normalization
    embeddings = sent_embed.encode(test_query, normalize_embeddings=True)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 1
    
    # Test encoding without normalization
    embeddings = sent_embed.encode(test_query, normalize_embeddings=False)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 1 