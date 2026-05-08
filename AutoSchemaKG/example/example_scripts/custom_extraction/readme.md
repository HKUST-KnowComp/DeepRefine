# Custom Knowledge Graph Extraction

This directory demonstrates how to use custom prompts and schemas for knowledge graph extraction, allowing you to define your own triple extraction format and structure.

## Files

- **`custom_kg_extraction.py`**: Main script for knowledge graph extraction using custom prompts and schemas, supports configurable models and datasets via command-line arguments.

- **`benchmarking.py`**: Evaluation script for testing the quality of extracted knowledge graphs on multi-hop QA benchmarks (e.g., 2WikiMultiHopQA).

- **`custom_prompt/custom_prompt.json`**: Custom prompt template that defines the instructions for triple extraction in different languages (currently supports English).

- **`custom_prompt/custom_schema.json`**: JSON schema that validates the structure of extracted triples (subject, relation, object format).

## Quick Start

### Basic Extraction

Run knowledge graph extraction with default settings:

```bash
python custom_kg_extraction.py
```

### Custom Configuration

Specify your own model and dataset:

```bash
python custom_kg_extraction.py \
    --keyword musique \
    --model Qwen/Qwen2.5-7B-Instruct
```

### Benchmarking

Evaluate the extracted knowledge graph:

```bash
python benchmarking.py \
    --keyword 2wikimultihopqa 
```

## Customization

### Custom Prompts

Edit `custom_prompt/custom_prompt.json` to modify:
- System instructions
- Triple extraction guidelines
- Output format requirements
- Language-specific variations

Example structure:
```json
{
  "en": {
    "system": "You are a helpful assistant",
    "triple_extraction": "Your extraction instructions..."
  }
}
```

### Custom Schemas

Edit `custom_prompt/custom_schema.json` to define your triple structure:
```json
{
    "triple_extraction": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "subject": { "type": "string" },
                "relation": { "type": "string" },
                "object": { "type": "string" }
            }
        }
    }
}
```

You can extend this to include additional fields like:
- `confidence`: Confidence score for the triple
- `source`: Source document or sentence
- `type`: Type of relationship
- `attributes`: Additional metadata

## Configuration

Key parameters in `custom_kg_extraction.py`:

```python
ProcessingConfig(
    triple_extraction_prompt_path='custom_prompt/custom_prompt.json',  # Custom prompt
    triple_extraction_schema_path='custom_prompt/custom_schema.json',  # Custom schema
    batch_size_triple=16,          # Batch size for extraction
    max_new_tokens=8192,           # Max tokens per generation
    include_concept=False,         # Include concept nodes
    remove_doc_spaces=True,        # Clean document text
    record=True                    # Save results to JSON
)
```

## Output

The script generates:
- **JSON files**: Raw extracted triples with metadata
- **CSV files**: Node and edge lists for Neo4j import
- **GraphML files**: Graph structure for NetworkX/visualization

## Use Cases

1. **Domain-Specific Extraction**: Customize prompts for medical, legal, or scientific texts
2. **Multi-Language Support**: Add language-specific prompts in `custom_prompt.json`
3. **Custom Schemas**: Define specialized triple formats (e.g., temporal, hierarchical)
4. **Benchmark Testing**: Evaluate different extraction strategies on standard datasets

## Related Documentation

- [AutoSchemaKG Main README](../../../README.md)
- [Processing Config Reference](../../../atlas_rag/kg_construction/triple_config.py)
