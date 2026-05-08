import pytest
from atlas_rag.retrieval.prompt_template import (
    one_shot_rag_qa_docs,
    one_shot_ircot_demo,
    rag_qa_system,
    one_shot_rag_qa_input,
    one_shot_rag_qa_output,
    prompt_template
)

def test_one_shot_rag_qa_docs():
    # Test that the docs contain expected content
    assert "The Last Horse" in one_shot_rag_qa_docs
    assert "Southampton" in one_shot_rag_qa_docs
    assert "Stanton Township" in one_shot_rag_qa_docs
    assert "Neville A. Stanton" in one_shot_rag_qa_docs
    assert "Finding Nemo" in one_shot_rag_qa_docs

def test_one_shot_ircot_demo():
    # Test that the demo contains both docs and question
    assert one_shot_rag_qa_docs in one_shot_ircot_demo
    assert "When was Neville A. Stanton's employer founded?" in one_shot_ircot_demo
    assert "Thought:" in one_shot_ircot_demo
    assert "1862" in one_shot_ircot_demo

def test_rag_qa_system():
    # Test that the system prompt contains key instructions
    assert "reading comprehension assistant" in rag_qa_system
    assert "Thought:" in rag_qa_system
    assert "Answer:" in rag_qa_system

def test_one_shot_rag_qa_input():
    # Test that the input contains both docs and question
    assert one_shot_rag_qa_docs in one_shot_rag_qa_input
    assert "When was Neville A. Stanton's employer founded?" in one_shot_rag_qa_input
    assert "Thought:" in one_shot_rag_qa_input

def test_one_shot_rag_qa_output():
    # Test that the output contains both thought process and answer
    assert "University of Southampton" in one_shot_rag_qa_output
    assert "1862" in one_shot_rag_qa_output
    assert "Answer:" in one_shot_rag_qa_output

def test_prompt_template():
    # Test the structure of the prompt template
    assert len(prompt_template) == 3
    
    # Test system message
    assert prompt_template[0]["role"] == "system"
    assert rag_qa_system in prompt_template[0]["content"]
    
    # Test user message
    assert prompt_template[1]["role"] == "user"
    assert one_shot_rag_qa_input in prompt_template[1]["content"]
    
    # Test assistant message
    assert prompt_template[2]["role"] == "assistant"
    assert one_shot_rag_qa_output in prompt_template[2]["content"] 