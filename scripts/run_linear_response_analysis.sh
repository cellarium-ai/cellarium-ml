#!/bin/bash
# run_linear_response_analysis.sh
# Usage: ./run_linear_response_analysis.sh <cuda_device_index>
# Example: ./run_linear_response_analysis.sh 1

# Check if a CUDA device index was provided; exit if not.
if [ -z "$1" ]; then
    echo "Usage: $0 <cuda_device_index>"
    exit 1
fi

CUDA_DEVICE_INDEX="$1"

# Default parameters (feel free to modify these defaults)
CHECKPOINT_PATH="/home/mehrtash/data/100M_long_run/run_001/lightning_logs/version_3/checkpoints/epoch=5-step=504000.ckpt"
REF_ADATA_PATH="/home/mehrtash/data/data/extract_0.h5ad"
GENE_INFO_TSV_PATH="/home/mehrtash/data/gene_info/gene_info.tsv"
VALIDATION_ADATA_PATH="/home/mehrtash/data/data/cellariumgpt_artifacts/cell_types_for_validation_filtered.h5ad"
OUTPUT_PATH="/home/mehrtash/data/data/linear_response/100M_long_run_last_checkpoint"

QUERY_CHUNK_SIZE=1000
TOTAL_PROB_MASS=0.5
MAX_COUNTS=1000
SYMMETRIC_RANGE_PAD=1

# Leave these empty to use the default (None) behavior in the python script.
MAX_QUERY_GENES=""
TOTAL_MRNA_UMIS=""

N_POINTS=5
QUERY_CHUNK_SIZE_LINEAR_RESPONSE=64

# Settings for cell index splitting
CELLS_PER_GPU=8

# Compute the start and end cell indices for this device.
START_CELL=$((CUDA_DEVICE_INDEX * CELLS_PER_GPU))
END_CELL=$((START_CELL + CELLS_PER_GPU - 1))
echo "CUDA device index: $CUDA_DEVICE_INDEX"
echo "Processing cell indices from $START_CELL to $END_CELL"

# Loop through the designated cell indices using seq for portability
for cell_index in $(seq $START_CELL $END_CELL); do
    echo "Processing cell index: $cell_index"
    
    # Build the command; conditionally add parameters if they are not empty.
    CMD="python linear_response_analysis.py \
        --cuda_device_index $CUDA_DEVICE_INDEX \
        --checkpoint_path \"$CHECKPOINT_PATH\" \
        --ref_adata_path \"$REF_ADATA_PATH\" \
        --gene_info_tsv_path \"$GENE_INFO_TSV_PATH\" \
        --validation_adata_path \"$VALIDATION_ADATA_PATH\" \
        --output_path \"$OUTPUT_PATH\" \
        --cell_index $cell_index \
        --query_chunk_size $QUERY_CHUNK_SIZE \
        --total_prob_mass $TOTAL_PROB_MASS \
        --max_counts $MAX_COUNTS \
        --symmetric_range_pad $SYMMETRIC_RANGE_PAD"
    
    # Append --max_query_genes if set
    if [ -n "$MAX_QUERY_GENES" ]; then
        CMD="$CMD --max_query_genes $MAX_QUERY_GENES"
    fi
    # Append --total_mrna_umis if set
    if [ -n "$TOTAL_MRNA_UMIS" ]; then
        CMD="$CMD --total_mrna_umis $TOTAL_MRNA_UMIS"
    fi

    CMD="$CMD --n_points $N_POINTS \
        --query_chunk_size_linear_response $QUERY_CHUNK_SIZE_LINEAR_RESPONSE"

    echo "Running command: $CMD"
    eval $CMD
done
