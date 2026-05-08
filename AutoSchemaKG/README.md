# AutoSchemaKG: A Knowledge Graph Construction Framework with Schema Generation and Knowledge Graph Completion

This repository contains the implementation of AutoSchemaKG, a novel framework for automatic knowledge graph construction that combines schema generation via conceptualization. The framework is designed to address the challenges of constructing high-quality knowledge graphs from unstructured text.

This project uses the following paper and data:

*   **Paper:** [Read the paper](https://arxiv.org/abs/2505.23628)
*   **Full Data:** [Download the dataset](https://hkustconnect-my.sharepoint.com/:f:/g/personal/httsangaj_connect_ust_hk/EgpZiquW1dNCniDY_3O7WbIBMrHkXT1dFis7rhfeHZ_AOQ?e=zXF3Yg) (onedrive)
*   **Neo4j CSV Dumps:** [Download the dataset](https://huggingface.co/datasets/AlexFanWei/AutoSchemaKG) (huggingface dataset)

### Update
- (05/07) Update with batch generation and refactor the codebase. Add comprehensive documentation for examples including PDF/Markdown conversion, multi-language processing, parallel generation, and custom extraction.
- (24/06) Add: ToG, Chinese KG construction (refer to example/multilingual_processing.md for KG construction with different languages). Separate NV-embed-v2 transformers dependency.

## AutoSchemaKG Overview

AutoSchemaKG introduces a two-stage approach:
1. **Knowledge Graph Triple Extraction**: Extract triples comprising entities and events from text by using LLMs 
2. **Schema Induction**: Automatically generate schema for the knowledge graph by using conceptualization and create semantic bridges between seemingly disparate information to enable zero-shot inferencing across domains


The framework achieves state-of-the-art performance on multiple benchmarks and demonstrates strong generalization capabilities across different domains.

## ATLAS Knowledge Graphs

ATLAS (Automated Triple Linking And Schema induction) is a family of knowledge graphs created through the AutoSchemaKG framework, which enables fully autonomous knowledge graph construction without predefined schemas. Here's a summary of what ATLAS is and how it works:

### Key Features of ATLAS Knowledge Graphs

- **Scale**: Consists of 900+ million nodes connected by 5.9 billion edges
- **Autonomous Construction**: Built without predefined schemas or manual intervention
- **Three Variants**: ATLAS-Wiki (from Wikipedia), ATLAS-Pes2o (from academic papers), and ATLAS-CC (from Common Crawl)




## Project Structure

```
AutoSchemaKG/
├── atlas_rag/                # Main package directory
│   ├── kg_construction/      # Knowledge graph construction modules
│   ├── llm_generator/        # Components for large language model generation
│   ├── retriever/            # Retrieval components for RAG
│   ├── utils/                # Utility functions for various tasks
│   └── vectorstore/          # Components for managing vector storage and embeddings
├── example/                  # Comprehensive examples and tutorials
│   ├── atlas_billion_kg_usage.ipynb      # Using ATLAS billion-scale KGs
│   ├── atlas_full_pipeline.ipynb         # Complete KG construction pipeline
│   ├── atlas_multihopqa.ipynb            # Multi-hop QA evaluation
│   ├── example_data/                     # Sample datasets (JSON, Markdown, PDF)
│   ├── example_scripts/                  # Production-ready scripts
│   │   ├── benchmark_extraction_example/ # Time cost benchmarking
│   │   ├── custom_extraction/            # Custom prompts and schemas
│   │   ├── neo4j_kg/                     # Neo4j API hosting
│   │   └── parallel_generation/          # Large-scale parallel processing
│   ├── generated/                        # Output directory for generated KGs
│   ├── hotpotqa_corpus_kg_input/         # Benchmark extraction results
│   ├── pdf_md_conversion/                # PDF/Markdown conversion tools
│   ├── multilingual_processing.md        # Multi-language KG construction guide
│   └── readme.md                         # Example directory documentation
├── EvaluateKGC/              # Knowledge graph quality evaluation
├── EvaluateFactuality/       # Factual consistency evaluation (FELM)
├── EvaluateGeneralTask/      # General performance evaluation (MMLU)
├── neo4j_scripts/            # Scripts for managing Neo4j databases
├── tests/                    # Unit tests for the project
└── README.md                 # Main documentation for the project
```

The project is organized into several key components:
- **`atlas_rag/`**: Core package with KG construction, LLM generation, retrieval, and vector storage
- **`example/`**: Complete tutorials, scripts, and sample data for various use cases
- **Evaluation directories**: Comprehensive metrics for KG quality, factuality, and general performance
- **`neo4j_scripts/`**: Database management and hosting utilities
- **`tests/`**: Unit tests ensuring code reliability

## Install atlas-rag through pip
```bash
pip install atlas-rag
```
To support NV-embed-v2, install the transformers package with the version constraint >=4.42.4,<=4.47.1 by running:
```bash
pip install atlas-rag[nvembed]
```

### KG Construction with ATLAS

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI
from transformers import pipeline
# client = OpenAI(api_key='<your_api_key>',base_url="<your_api_base_url>") 
# model_name = "meta-llama/llama-3.1-8b-instruct"

model_name = "meta-llama/Llama-3.1-8B-Instruct"
client = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto",
)
keyword = 'Dulce'
output_directory = f'import/{keyword}'
triple_generator = LLMGenerator(client, model_name=model_name)
kg_extraction_config = ProcessingConfig(
      model_path=model_name,
      data_directory="example_data",
      filename_pattern=filename_pattern,
      batch_size_triple=3, # batch size for triple extraction
      batch_size_concept=16, # batch size for concept generation
      output_directory=f"{output_directory}",
      max_new_tokens=2048,
      max_workers=3,
      remove_doc_spaces=True, # For removing duplicated spaces in the document text
)
kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)

# Construct entity&event graph
kg_extractor.run_extraction() # Involved LLM Generation
# Convert Triples Json to CSV
kg_extractor.convert_json_to_csv()
# Concept Generation
kg_extractor.generate_concept_csv(batch_size=64) # Involved LLM Generation
# Create Concept CSV
kg_extractor.create_concept_csv()
# Convert csv to graphml for networkx
kg_extractor.convert_to_graphml()
```

## Large Knowledge Graph Hosting and Retrieval Augmented Generation

This repository provides support for hosting and implementing Retrieval Augmented Generation (RAG) over our constructed knowledge graphs: `ATLAS-wiki`, `ATLAS-pes2o`, and `ATLAS-cc`. For detailed instructions on hosting and running these knowledge graphs, please refer to the `atlas_billion_kg_usage.ipynb` notebook. 

## Building New Knowledge Graphs and Implementing RAG

The `atlas_full_pipeline.ipynb` notebook demonstrates how to:
- Build new knowledge graphs using AutoschemaKG
- Implement Retrieval Augmented Generation on your custom knowledge graphs

## Examples and Documentation

The [`example/`](example/) directory provides comprehensive tutorials, scripts, and documentation for various use cases:

- **[Example Directory Overview](example/readme.md)**: Complete guide to all examples and workflows
- **[Multi-Language Processing](example/multilingual_processing.md)**: Build KGs in Chinese, Japanese, Korean, and more
- **[PDF/Markdown Conversion](example/pdf_md_conversion/readme.md)**: Convert PDF documents for KG construction
- **[Parallel Generation](example/example_scripts/parallel_generation/readme.md)**: Large-scale parallel KG processing
- **[Custom Extraction](example/example_scripts/custom_extraction/readme.md)**: Define custom prompts and schemas
- **[Benchmark Extraction](example/example_scripts/benchmark_extraction_example/readme.md)**: Time cost benchmarking
- **[Neo4j KG Hosting](example/example_scripts/neo4j_kg/readme.md)**: Host KGs as Neo4j-compatible APIs

### Quick Start Examples

**Jupyter Notebooks:**
- `example/atlas_billion_kg_usage.ipynb` - Using ATLAS billion-scale knowledge graphs
- `example/atlas_full_pipeline.ipynb` - Complete KG construction pipeline
- `example/atlas_multihopqa.ipynb` - Multi-hop QA evaluation

**Sample Datasets:**
- English corpus: `example/example_data/Dulce.json`
- Chinese text: `example/example_data/multilingual_data/RomanceOfTheThreeKingdom-zh-CN.json`
- PDF documents: `example/example_data/pdf_data/`
- Markdown files: `example/example_data/md_data/`


## Multi-hop Question Answering Evaluation

To replicate our multi-hop question answering evaluation results on benchmark datasets:
- `MuSiQue`
- `HotpotQA` 
- `2WikiMultiHopQA`

Please follow the instructions in the `atlas_multihopqa.ipynb` notebook, which contains all necessary code and configuration details.

## General Evaluation


The framework includes comprehensive evaluation metrics across three dimensions:
- Knowledge Graph Quality  (`EvaluateKGC`)
- Factual Consistency on FELM (`EvaluateFactuality`)
- General Performance on MMLU (`EValuateGeneralTask`)

Detailed evaluation procedures can be found in the respective evaluation directories.

## PDF and Markdown Support

AutoSchemaKG supports processing PDF documents for knowledge graph construction. For detailed instructions on converting PDFs to Markdown and then to JSON format suitable for KG construction, please refer to the [PDF/Markdown Conversion Guide](example/pdf_md_conversion/readme.md).

**Quick Overview:**
1. Convert PDF to Markdown using the [pdf_process](https://github.com/Swgj/pdf_process) tool
2. Convert Markdown to JSON for AutoSchemaKG processing
3. Use the JSON files in your KG construction pipeline

See the [complete documentation](example/pdf_md_conversion/readme.md) for setup instructions, configuration options, and usage examples.

## Citation

If you use this code in your research, please cite our paper:

```
@misc{bai2025autoschemakgautonomousknowledgegraph,
      title={AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora}, 
      author={Jiaxin Bai and Wei Fan and Qi Hu and Qing Zong and Chunyang Li and Hong Ting Tsang and Hongyu Luo and Yauwai Yim and Haoyu Huang and Xiao Zhou and Feng Qin and Tianshi Zheng and Xi Peng and Xin Yao and Huiwen Yang and Leijie Wu and Yi Ji and Gong Zhang and Renhai Chen and Yangqiu Song},
      year={2025},
      eprint={2505.23628},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.23628}, 
}
```



## Contact

Jiaxin Bai: jbai@connect.ust.hk 

Dennis Hong Ting TSANG : httsangaj@connect.ust.hk

Haoyu Huang: haoyuhuang@link.cuhk.edu.hk

