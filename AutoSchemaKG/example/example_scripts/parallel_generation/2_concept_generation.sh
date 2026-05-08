#!/bin/bash

# Configuration
TOTAL_SHARDS=3
LOG_DIR="/home/httsangaj/projects/AutoSchemaKG/log"
SCRIPT_DIR="/home/httsangaj/projects/AutoSchemaKG/example/example_scripts/parallel_generation"

# Base port number - each shard will use a different port
BASE_PORT=8135

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Function to run a single shard with its own LLM instance port
run_shard() {
    local shard_num=$1
    local port=$((BASE_PORT + shard_num))
    
    echo "Starting concept generation for shard ${shard_num}/${TOTAL_SHARDS} on port ${port}"
    python "${SCRIPT_DIR}/2_concept_generation.py" \
        --shard "${shard_num}" \
        --total_shards "${TOTAL_SHARDS}" \
        --port "${port}" \
        > "${LOG_DIR}/concept_shard_${shard_num}_port_${port}.log" 2>&1 &
    
    echo "  Process ID: $!"
}

echo "=========================================="
echo "Parallel Concept Generation"
echo "=========================================="
echo "Total shards: ${TOTAL_SHARDS}"
echo "Base port: ${BASE_PORT}"
echo "Port range: ${BASE_PORT} to $((BASE_PORT + TOTAL_SHARDS - 1))"
echo "Log directory: ${LOG_DIR}"
echo "=========================================="
echo ""

# Run all shards in parallel, each with its own port
for ((i=0; i<TOTAL_SHARDS; i++)); do
    run_shard $i
    sleep 1  # Small delay to stagger startup
done

# Wait for all processes to complete
echo ""
echo "All concept generation shards started. Waiting for completion..."
echo "Monitor progress with: tail -f ${LOG_DIR}/concept_shard_*.log"
echo ""

wait

echo ""
echo "=========================================="
echo "All concept generation shards completed."
echo "=========================================="
echo ""

# Merge results (optional - implement based on your needs)
# echo "Merging concept results..."
# python "${SCRIPT_DIR}/merge_concept_shards.py"

echo "Concept generation complete."
echo "Check logs at: ${LOG_DIR}/concept_shard_*.log"
