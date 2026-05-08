#!/bin/bash
set -e  # Exit on error

SCRIPT_DIR="/home/httsangaj/projects/AutoSchemaKG/example/example_scripts/parallel_generation"

echo "=========================================="
echo "Full Knowledge Graph Generation Pipeline"
echo "=========================================="
echo ""

# Check if scripts are executable
if [ ! -x "${SCRIPT_DIR}/1_slice_kg_extraction.sh" ]; then
    echo "Making 1_slice_kg_extraction.sh executable..."
    chmod +x "${SCRIPT_DIR}/1_slice_kg_extraction.sh"
fi

if [ ! -x "${SCRIPT_DIR}/2_concept_generation.sh" ]; then
    echo "Making 2_concept_generation.sh executable..."
    chmod +x "${SCRIPT_DIR}/2_concept_generation.sh"
fi

# Stage 1: Triple Extraction
echo ""
echo "=========================================="
echo "Stage 1: Triple Extraction"
echo "=========================================="
"${SCRIPT_DIR}/1_slice_kg_extraction.sh"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Stage 1 completed successfully!"
    echo ""
    
    # Small delay to ensure all files are properly written
    sleep 2
    
    # Stage 2: Concept Generation
    echo "=========================================="
    echo "Stage 2: Concept Generation"
    echo "=========================================="
    "${SCRIPT_DIR}/2_concept_generation.sh"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ Full pipeline completed successfully!"
        echo "=========================================="
        echo ""
        echo "Results available in output directory."
        echo "Logs available at:"
        echo "  - Triple extraction: /home/httsangaj/projects/AutoSchemaKG/log/shard_*.log"
        echo "  - Concept generation: /home/httsangaj/projects/AutoSchemaKG/log/concept_shard_*.log"
        exit 0
    else
        echo ""
        echo "✗ Stage 2 (Concept Generation) failed!"
        echo "Check logs at: /home/httsangaj/projects/AutoSchemaKG/log/concept_shard_*.log"
        exit 1
    fi
else
    echo ""
    echo "✗ Stage 1 (Triple Extraction) failed!"
    echo "Check logs at: /home/httsangaj/projects/AutoSchemaKG/log/shard_*.log"
    exit 1
fi
