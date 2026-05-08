# Parallel Knowledge Graph Generation

This directory demonstrates how to parallelize knowledge graph extraction and concept generation using multiple LLM instances running on different ports. By distributing the workload across multiple LLM servers, you can significantly speed up the extraction process for large datasets.

## Overview

The parallel generation approach divides your dataset into **shards** and processes each shard with a separate LLM instance. Each instance runs on a different port, allowing multiple extractions to happen simultaneously.

This directory includes two parallel workflows:
1. **Triple Extraction**: Extract knowledge graph triples from raw text
2. **Concept Generation**: Generate concepts and mappings for extracted entities

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Original Dataset                          │
└─────────────────────────────────────────────────────────────┘
                         │
                         ├─────────────────────────┐
                         │                         │
                         ▼                         ▼
              ┌─────────────────┐      ┌─────────────────┐
              │   Shard 0       │      │   Shard 1       │
              │   Port 8135     │      │   Port 8136     │
              └─────────────────┘      └─────────────────┘
                         │                         │
                         ▼                         ▼
              ┌─────────────────┐      ┌─────────────────┐
              │ LLM Instance 0  │      │ LLM Instance 1  │
              │ (localhost:8135)│      │ (localhost:8136)│
              └─────────────────┘      └─────────────────┘
                         │                         │
                         └─────────┬───────────────┘
                                   ▼
                        ┌─────────────────────┐
                        │   Merged Results    │
                        └─────────────────────┘
```

## Prerequisites

### Starting Multiple vLLM Instances

Start multiple LLM server instances on different ports. For 3 parallel instances:

```bash
# Terminal 1: Port 8135
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8135 --max-model-len 8192 --gpu-memory-utilization 0.9

# Terminal 2: Port 8136
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8136 --max-model-len 8192 --gpu-memory-utilization 0.9

# Terminal 3: Port 8137
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8137 --max-model-len 8192 --gpu-memory-utilization 0.9
```

**Tips:**
- Use `tmux` or `screen` to manage multiple terminals
- Adjust `--gpu-memory-utilization` based on available GPU memory
- Use `CUDA_VISIBLE_DEVICES` to assign different GPUs to each instance

## Files

- **`1_slice_kg_extraction.py`**: Triple extraction script for a single shard
- **`1_slice_kg_extraction.sh`**: Launches parallel triple extraction across all shards
- **`2_concept_generation.py`**: Concept generation script for a single shard
- **`2_concept_generation.sh`**: Launches parallel concept generation across all shards
- **`run_full_pipeline.sh`**: Master script that runs both stages sequentially

## Configuration

Edit the variables at the top of the shell scripts:

```bash
TOTAL_SHARDS=3        # Number of parallel processes (match number of LLM instances)
BASE_PORT=8135        # Starting port number
LOG_DIR="/path/to/log"
SCRIPT_DIR="/path/to/parallel_generation"
```

### Port Assignment

Ports are automatically assigned: `BASE_PORT + shard_number`
- Shard 0 → Port 8135
- Shard 1 → Port 8136
- Shard 2 → Port 8137

## Usage

### Quick Start (Recommended)

```bash
cd /home/httsangaj/projects/AutoSchemaKG/example/example_scripts/parallel_generation

# 1. Start LLM instances on ports 8135, 8136, 8137 (see Prerequisites)
# 2. Configure TOTAL_SHARDS and BASE_PORT in the scripts
# 3. Run the complete pipeline
./run_full_pipeline.sh
```

### Manual Workflow

If you prefer to run each stage separately:

#### Stage 1: Triple Extraction

```bash
chmod +x slice_kg_extraction.sh
./slice_kg_extraction.sh

# Monitor progress
tail -f /home/httsangaj/projects/AutoSchemaKG/log/shard_*.log
```

#### Stage 2: Concept Generation

After triple extraction completes:

```bash
chmod +x concept_generation.sh
./concept_generation.sh

# Monitor progress
tail -f /home/httsangaj/projects/AutoSchemaKG/log/concept_shard_*.log
```
