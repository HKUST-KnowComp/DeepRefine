import pytest
import json
import jsonschema
from atlas_rag.retrieval.filter_template import (
    flatten_and_filter_triplets,
    validate_filter_output,
    messages
)

def test_flatten_and_filter_triplets():
    # Test with simple nested structure
    input_data = {
        "fact": [
            ["subject1", "predicate1", "object1"],
            [["subject2", "predicate2", "object2"]],
            ["subject3", "predicate3", "object3"]
        ]
    }
    expected_output = {
        "fact": [
            ["subject1", "predicate1", "object1"],
            ["subject2", "predicate2", "object2"],
            ["subject3", "predicate3", "object3"]
        ]
    }
    assert flatten_and_filter_triplets(input_data) == expected_output

    # Test with invalid triplets
    input_data = {
        "fact": [
            ["subject1", "predicate1"],  # Invalid: only 2 elements
            ["subject2", "predicate2", "object2", "extra"],  # Invalid: 4 elements
            ["subject3", "predicate3", "object3"]  # Valid
        ]
    }
    expected_output = {
        "fact": [
            ["subject3", "predicate3", "object3"]
        ]
    }
    assert flatten_and_filter_triplets(input_data) == expected_output

def test_validate_filter_output():
    # Test valid output
    valid_output = json.dumps({
        "fact": [
            ["subject1", "predicate1", "object1"],
            ["subject2", "predicate2", "object2"]
        ]
    })
    result = validate_filter_output(valid_output)
    assert isinstance(result, dict)
    assert "fact" in result
    assert len(result["fact"]) == 2
    assert all(len(triplet) == 3 for triplet in result["fact"])

    # Test invalid output (wrong type)
    invalid_output = json.dumps({
        "fact": "not an array"
    })
    valid_output = validate_filter_output(invalid_output)
    assert "fact" in valid_output

    # Test invalid output (missing fact key)
    invalid_output = json.dumps({
        "other_key": []
    })
    valid_output = validate_filter_output(invalid_output)
    assert "fact" in valid_output

    # Test invalid output (wrong array length)
    invalid_output = json.dumps({
        "fact": [
            ["subject1", "predicate1"]  # Only 2 items instead of 3
        ]
    })
    valid_output = validate_filter_output(invalid_output)
    assert "fact" in valid_output

    # Test invalid output (wrong item type)
    invalid_output = json.dumps({
        "fact": [
            ["subject1", 123, "object1"]  # Number instead of string
        ]
    })
    valid_output = validate_filter_output(invalid_output)
    assert "fact" in valid_output

def test_messages_structure():
    # Test that messages list has the correct structure
    assert len(messages) >= 3  # At least system message and one example
    
    # Test system message
    assert messages[0]["role"] == "system"
    assert "question-answering system" in messages[0]["content"]
    
    # Test example messages
    for i in range(1, len(messages), 2):
        if i + 1 < len(messages):
            # Check user message
            assert messages[i]["role"] == "user"
            assert "[[ ## question ## ]]" in messages[i]["content"]
            assert "[[ ## fact_before_filter ## ]]" in messages[i]["content"]
            
            # Check assistant message
            assert messages[i + 1]["role"] == "assistant"
            assert isinstance(json.loads(messages[i + 1]["content"]), dict)
            assert "fact" in json.loads(messages[i + 1]["content"]) 