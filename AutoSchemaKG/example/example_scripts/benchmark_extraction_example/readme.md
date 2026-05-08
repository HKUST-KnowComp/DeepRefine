# Knowledge Graph Extraction Time Benchmarking

This directory provides scripts for benchmarking the time cost of knowledge graph extraction and concept generation processes. Unlike the `parallel_generation` directory which focuses on parallel processing, this directory is designed specifically for measuring and analyzing extraction performance.

## Files

- **`1_slice_kg_extraction.py`**: Benchmarks the time cost of entity-event triple extraction from text documents with detailed timing metrics.

- **`2_concept_generation.py`**: Benchmarks the time cost of concept node generation and graph construction from extracted triples.

## Purpose

This benchmark suite helps you:
- **Measure extraction speed** for different LLM models
- **Compare performance** across different hardware configurations
- **Optimize batch sizes** for maximum throughput
- **Estimate processing time** for large-scale datasets
- **Profile bottlenecks** in the extraction pipeline

## Quick Start

### 1. Triple Extraction Benchmark

Run entity-event extraction timing:

```bash
python 1_slice_kg_extraction.py \
    --shard 0 \
    --total_shards 1 \
    --port 8135
```

**Key Parameters:**
- `--shard`: Which data shard to process (default: 0)
- `--total_shards`: Total number of data shards (default: 1)
- `--port`: vLLM/SGLang server port (default: 8135)

### 2. Concept Generation Benchmark

Run concept generation timing:

```bash
python 2_concept_generation.py \
    --shard 0 \
    --total_shards 1 \
    --port 8135
```

## Timing Metrics

### Entity-Event Extraction Time

The total extraction time is recorded in the **last object** of the output JSON file:

**Location:** `output_dir/kg_extraction/xxx_1_in_1.json`

**Key:** `total_extraction_time_seconds`

Example:
```json
{
  "id": "doc_12345",
  "text": "...",
  "triples": [...],
  "total_extraction_time_seconds": 245.67
}
```

### Concept Generation Time

The concept generation time is recorded in the **last line** of the logging file:

**Location:** `output_dir/concepts/logging.txt`

**Format:** `Total concept generation time: xxx seconds`

Example:
```
Processing concepts...
Creating CSV files...
Total time: 89.34 seconds
```

## Configuration

### Benchmark Settings

Both scripts use `benchmark=True` and `record=True` in `ProcessingConfig`:

```python
kg_extraction_config = ProcessingConfig(
    model_path=model_name,
    data_directory="/data/AutoSchema/processed_data/cc_en_head",
    filename_pattern=keyword,
    batch_size_triple=16,        # Extraction batch size
    batch_size_concept=64,       # Concept generation batch size
    output_directory=f'/data/AutoSchema/processed_data/cc_en_head/{model_name}',
    current_shard_triple=args.shard,
    total_shards_triple=args.total_shards,
    record=True,                 # Save detailed results
    max_new_tokens=8192,         # Max tokens (extraction: 8192, concept: 512)
    benchmark=True               # Enable timing metrics
)
```

## Benchmarking Workflow

### Step-by-Step Process

1. **Start LLM Server**
   ```bash
   # Example: vLLM server
   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8135
   ```

2. **Run Triple Extraction Benchmark**
   ```bash
   python 1_slice_kg_extraction.py --port 8135
   ```

3. **Check Extraction Time**
   ```bash
   # View last object in JSON output
   tail -n 20 output_dir/kg_extraction/xxx_1_in_1.json | grep total_extraction_time_seconds
   ```

4. **Run Concept Generation Benchmark**
   ```bash
   python 2_concept_generation.py --port 8135
   ```

5. **Check Concept Time**
   ```bash
   # View last line of logging file
   tail -n 1 output_dir/concepts/logging.txt
   ```

## Performance Analysis

### Factors Affecting Speed

1. **Model Size**: Larger models (70B) are slower but more accurate than smaller models (7B)
2. **Batch Size**: Larger batches improve throughput but require more memory
3. **Max Tokens**: Higher token limits allow more complex extractions but increase latency
4. **Hardware**: GPU memory and compute capability directly impact speed
5. **Concurrency**: `max_workers` parameter controls parallel API calls

## Output Structure

After benchmarking, you'll find:

```
output_dir/
├── kg_extraction/
│   └── xxx_1_in_1.json          # Contains total_extraction_time_seconds
├── concepts/
│   ├── logging.txt               # Contains total concept generation time
│   ├── concept_nodes.csv
│   └── concept_edges.csv
└── graphml/
    └── knowledge_graph.graphml
```

