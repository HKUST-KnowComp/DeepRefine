# AutoSchemaKG Examples

This directory contains comprehensive examples, scripts, and data for using AutoSchemaKG to construct knowledge graphs from various data sources and formats.

## Directory Structure

```
example/
├── atlas_billion_kg_usage.ipynb      # Using ATLAS billion-scale knowledge graphs
├── atlas_full_pipeline.ipynb         # Complete KG construction pipeline
├── atlas_multihopqa.ipynb            # Multi-hop QA evaluation
├── example_data/                     # Sample data in various formats
├── example_scripts/                  # Reusable scripts for different use cases
├── generated/                        # Example output directory for generated KGs
├── hotpotqa_corpus_kg_input/         # Benchmark extraction results
└── pdf_md_conversion/                # PDF/Markdown conversion tools
```

## Quick Start

### 1. Jupyter Notebooks

Interactive notebooks demonstrating key workflows:

- **`atlas_billion_kg_usage.ipynb`**: Learn how to host and query ATLAS billion-scale KGs (ATLAS-Wiki, ATLAS-Pes2o, ATLAS-CC) with RAG
- **`atlas_full_pipeline.ipynb`**: Complete end-to-end pipeline from raw text to knowledge graph construction and RAG
- **`atlas_multihopqa.ipynb`**: Benchmark your KGs on multi-hop QA datasets (MuSiQue, HotpotQA, 2WikiMultiHopQA)

### 2. Example Data

Example datasets in different formats:

```
example_data/
├── Dulce.json                        # English text corpus
├── Dulce_test.json                   # Test dataset
├── md_data/                          # Markdown files
│   ├── Apple_Environmental_Progress_Report_2024.md
│   └── CICGPC_Glazing_ver1.0a.md
├── multilingual_data/                # Multi-language datasets
│   ├── RomanceOfTheThreeKingdom-zh-CN.json  # Simplified Chinese
│   └── RomanceOfTheThreeKingdom-zh-HK.json  # Traditional Chinese
└── pdf_data/                         # PDF documents and converted JSON
    ├── Apple_Environmental_Progress_Report_2024.pdf
    └── CICGPC_Glazing_ver1.0a.pdf
```

### 3. Example Scripts

Scripts for various scenarios:

- **[benchmark_extraction_example/](example_scripts/benchmark_extraction_example/readme.md)**: Time cost benchmarking for KG extraction and concept generation
- **[custom_extraction/](example_scripts/custom_extraction/readme.md)**: Using custom prompts and schemas for domain-specific extraction
- **[neo4j_kg/](example_scripts/neo4j_kg/readme.md)**: Hosting knowledge graphs as Neo4j-compatible API servers
- **[parallel_generation/](example_scripts/parallel_generation/readme.md)**: Large-scale parallel KG construction

## Multi-Language Knowledge Graph Construction

AutoSchemaKG provides comprehensive support for constructing knowledge graphs in multiple languages, including English, Chinese (Simplified and Traditional), Japanese, Korean, and many others.

For detailed information on multi-language processing, see the **[Multi-Language Processing Guide](multilingual_processing.md)**.

## Common Workflows

### Workflow 1: Basic KG Construction

```python
# See: atlas_full_pipeline.ipynb
# 1. Prepare data → 2. Extract triples → 3. Generate concepts → 4. Export to GraphML
```

### Workflow 2: Large-Scale Parallel Processing

```bash
# See: example_scripts/parallel_generation/
# Use shell scripts for multi-shard parallel extraction
bash run_full_pipeline.sh
```

### Workflow 3: Custom Domain Extraction

```python
# See: example_scripts/custom_extraction/
# Define custom prompts and schemas for your domain
```

### Workflow 4: Time Benchmarking

```python
# See: example_scripts/benchmark_extraction_example/
# Measure extraction and concept generation performance
```

### Workflow 5: PDF Document Processing

```bash
# See: pdf_md_conversion/readme.md
# 1. PDF → Markdown → 2. Markdown → JSON → 3. JSON → KG
```

## Output Formats

AutoSchemaKG supports multiple output formats:

- **JSON**: Raw extracted triples with metadata
- **CSV**: Node and edge lists for Neo4j import
- **GraphML**: Graph structure for NetworkX and visualization tools
- **Vector Indices**: Precomputed embeddings for retrieval
- **Neo4j**: Neo4j GraphDB for storing and managing the knowledge graph efficiently

