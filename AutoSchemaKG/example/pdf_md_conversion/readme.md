# PDF to Markdown Conversion for Knowledge Graph Construction

This directory provides tools for converting PDF documents to Markdown format, which can then be processed into JSON format suitable for knowledge graph construction with AutoSchemaKG.

**Creator:** [swgj](https://github.com/Swgj)

## Overview

The PDF-to-Markdown transformation pipeline enables you to:
1. Convert PDF files to clean Markdown format
2. Extract images and generate descriptions (optional)
3. Convert Markdown to JSON format for KG construction
4. Process single files or entire directories

## Prerequisites

Due to the version requirements of `marker-pdf`, we recommend creating a separate conda environment for PDF-to-Markdown transformation.

## Installation

### 1. Clone the PDF Transform Repository

```bash
git clone https://github.com/Swgj/pdf_process
cd pdf_process
```

### 2. Create Conda Environment

```bash
conda create --name pdf-marker pip python=3.10
conda activate pdf-marker
```

### 3. Install Dependencies

```bash
pip install 'marker-pdf[full]'
pip install google-genai
```

## Configuration

### Edit the `config.yaml` File

The configuration file controls all aspects of the PDF processing pipeline:

```yaml
processing_config:
  llm_service: "marker.services.azure_openai.AzureOpenAIService" # Azure OpenAI Service
  # To use default Gemini server, comment out the line above
  
  other_config:
    use_llm: true
    extract_images: false  # false: use LLM for text description; true: extract images without descriptions
    page_range: null  # null: process all pages, or use List[int] format like [9, 10, 11, 12]
    max_concurrency: 2 # maximum number of concurrent processes
    
    # Azure OpenAI API configuration
    azure_endpoint: <your endpoint>
    azure_api_version: "2024-10-21"
    deployment_name: "gpt-4o"

# API configuration
api:
  # api_key_env: "GEMINI_API_KEY"  # Uncomment for Gemini API key
  api_key_env: "AZURE_API_KEY"      # Use for Azure OpenAI

# Input path configuration - can be a file or folder path
input:
  # Supports relative and absolute paths
  path: "test_data"  # Can be a single file path or folder path
  # path: "data/Apple_Environmental_Progress_Report_2024.pdf"  # Example of a single file
  
  # If it's a folder, you can set file filtering conditions
  file_filters:
    extensions: [".pdf"]  # Only process PDF files
    recursive: true       # Whether to recursively process subfolders
    exclude_patterns:     # Exclude files that match these patterns
      - "*temp*"
      - "*~*"

# Output configuration
output:
  base_dir: "md_output"     # Output directory
  create_subdirs: true      # Whether to create a subdirectory for each input file
  format: "md"              # Output format (md, txt)
  
# Logging configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  show_progress: true
```

### Key Configuration Options

#### LLM Service Options
- **Azure OpenAI**: Set `llm_service: "marker.services.azure_openai.AzureOpenAIService"`
- **Gemini**: Comment out the `llm_service` line to use default Gemini

#### Processing Options
- `use_llm`: Enable LLM-based text extraction and enhancement
- `extract_images`: 
  - `false`: Use LLM to generate text descriptions of images
  - `true`: Extract images as files without descriptions
- `page_range`: Specify pages to process (e.g., `[9, 10, 11, 12]`) or `null` for all pages
- `max_concurrency`: Number of concurrent processes (adjust based on your system)

#### Input Options
- `path`: Single file or directory path
- `file_filters.extensions`: File types to process (default: `[".pdf"]`)
- `file_filters.recursive`: Process subdirectories
- `file_filters.exclude_patterns`: Patterns to exclude from processing

#### Output Options
- `base_dir`: Output directory for Markdown files
- `create_subdirs`: Create separate subdirectories for each input file
- `format`: Output format (`md` or `txt`)

## Usage Workflow

### Step 1: PDF to Markdown Conversion

1. **Place your PDF files** in the input directory specified in `config.yaml`

2. **Run the conversion script:**
   ```bash
   bash run.sh
   ```

3. **Output:** You'll find Markdown files in the `md_output` directory (or your specified `base_dir`)

### Step 2: Markdown to JSON Conversion

After obtaining Markdown files, convert them to JSON format for AutoSchemaKG:

```bash
# Return to the AutoSchemaKG parent directory
cd /path/to/AutoSchemaKG

# Convert Markdown to JSON
python -m atlas_rag.kg_construction.utils.md_processing.markdown_to_json \
    --input example_data/md_data \
    --output example_data
```

**Parameters:**
- `--input`: Path to the directory containing Markdown files
- `--output`: Path where JSON files will be saved

### Step 3: Knowledge Graph Construction

Use the generated JSON files with AutoSchemaKG for knowledge graph construction:

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator

# Your KG construction code here
# See main README for full example
```

## Example: Complete Workflow

```bash
# 1. Set up environment
conda activate pdf-marker
cd pdf_process

# 2. Configure config.yaml with your settings
# Edit: input path, Azure endpoint, API keys, etc.

# 3. Convert PDF to Markdown
bash run.sh

# 4. Return to AutoSchemaKG directory
cd /path/to/AutoSchemaKG

# 5. Convert Markdown to JSON
python -m atlas_rag.kg_construction.utils.md_processing.markdown_to_json \
    --input pdf_process/md_output \
    --output example/example_data

# 6. Run KG construction
# Use the JSON files in your KG construction pipeline
```


## Credits

PDF-to-Markdown conversion tool developed by [swgj](https://github.com/Swgj).
