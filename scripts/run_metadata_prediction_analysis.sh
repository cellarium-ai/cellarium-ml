#!/bin/bash
# run_metadata_prediction_analysis.sh

# Usage check: GPU_ID is required as the first argument.
if [ -z "$1" ]; then
    echo "Usage: $0 <GPU_ID>"
    exit 1
fi

GPU_ID=$1

# ----------------------------
# Default parameters (CLI args)
# ----------------------------
CHECKPOINT_PATH="/home/mehrtash/data/100M_long_run/run_001/lightning_logs/version_3/checkpoints/epoch=5-step=504000.ckpt"
REF_ADATA_PATH="/home/mehrtash/data/data/extract_0.h5ad"
GENE_INFO_PATH="/home/mehrtash/data/gene_info/gene_info.tsv"
ONTOLOGY_RESOURCE_PATH="/home/mehrtash/data/data/cellariumgpt_artifacts/ontology"
OUTPUT_PATH="/home/mehrtash/data/data/cellariumgpt_artifacts/metadata_predictions/100M_long_run_last"
RNG_SEED=42
N_CELLS=1000
N_GENES=4091
GENE_SELECTION_METHOD="random"
CHUNK_SIZE=16

# ----------------------------
# Distribution of val_adata_index values
# ----------------------------
TOTAL_INDICES=110
NUM_GPUS=8

# Calculate the start and end indices for the given GPU.
# Indices range from 1 to TOTAL_INDICES (inclusive).
START_INDEX=$(python3 -c "import math; print(int(math.floor(${GPU_ID} * ${TOTAL_INDICES} / ${NUM_GPUS}))+1)")
END_INDEX=$(python3 -c "import math; print(int(math.floor((${GPU_ID}+1) * ${TOTAL_INDICES} / ${NUM_GPUS})))")

echo "GPU ${GPU_ID} processing val_adata_index from ${START_INDEX} to ${END_INDEX}"

# ----------------------------
# Loop over the assigned indices and run the Python CLI tool
# ----------------------------
for i in $(seq ${START_INDEX} ${END_INDEX}); do
    echo "Running metadata prediction for val_adata_index = ${i} on GPU ${GPU_ID}"
    python3 metadata_prediction_analysis.py \
        --cuda_device_index ${GPU_ID} \
        --val_adata_index ${i} \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --ref_adata_path "${REF_ADATA_PATH}" \
        --gene_info_path "${GENE_INFO_PATH}" \
        --ontology_resource_path "${ONTOLOGY_RESOURCE_PATH}" \
        --output_path "${OUTPUT_PATH}" \
        --rng_seed ${RNG_SEED} \
        --n_cells ${N_CELLS} \
        --n_genes ${N_GENES} \
        --gene_selection_method ${GENE_SELECTION_METHOD} \
        --chunk_size ${CHUNK_SIZE}
done
