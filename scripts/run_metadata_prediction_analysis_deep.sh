#!/bin/bash
# run_metadata_prediction_analysis.sh

# Usage check: All three arguments are required.
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <GPU_ID> <CHECKPOINT_PATH> <OUTPUT_PATH>"
    exit 1
fi

GPU_ID=$1

# ----------------------------
# Default parameters (CLI args)
# ----------------------------
CHECKPOINT_PATH=$2
REF_ADATA_PATH="/home/mehrtash/data/data/extract_0.h5ad"
GENE_INFO_PATH="/home/mehrtash/data/gene_info/gene_info.tsv"
ONTOLOGY_RESOURCE_PATH="/home/mehrtash/data/data/cellariumgpt_artifacts/ontology"
FIXED_PROMPT_VARS_SUBLIST_PATH="/home/mehrtash/data/data/cellariumgpt_artifacts/empty_gene_ids.txt"
RAND_PROMPT_VARS_SUBLIST_PATH="/home/mehrtash/data/data/cellariumgpt_artifacts/autosomal_gene_ids.txt"
VAL_ADATA_ROOT_PATH="/home/mehrtash/data/data/cellariumgpt_validation"
VAL_ADATA_FILE_LIST="/home/mehrtash/data/data/cellariumgpt_validation/all_files.txt"
OUTPUT_PATH=$3
RNG_SEED=42
#N_CELLS=2000
N_GENES=8192
GENE_SELECTION_METHOD="highly_expressed"
CHUNK_SIZE=32
METRIC_STYLE="hop_k_call"

# ----------------------------
# Infer the total number of non-empty lines in VAL_ADATA_FILE_LIST
# ----------------------------
TOTAL_INDICES=$(grep -cve '^\s*$' "${VAL_ADATA_FILE_LIST}")
echo "Total number of validation files: ${TOTAL_INDICES}"

NUM_GPUS=1

# Calculate the start and end indices for the given GPU.
# Indices range from 1 to TOTAL_INDICES (inclusive).
START_INDEX=$(python3 -c "import math; print(int(math.floor(${GPU_ID} * ${TOTAL_INDICES} / ${NUM_GPUS}))+1)")
END_INDEX=$(python3 -c "import math; print(int(math.floor((${GPU_ID}+1) * ${TOTAL_INDICES} / ${NUM_GPUS})))")

echo "GPU ${GPU_ID} processing validation files from line ${START_INDEX} to ${END_INDEX}"

# ----------------------------
# Loop over the assigned lines and run the Python CLI tool
# ----------------------------
for i in $(seq ${START_INDEX} ${END_INDEX}); do
    # Read the i-th non-empty line from VAL_ADATA_FILE_LIST
    file=$(sed -n "${i}p" "${VAL_ADATA_FILE_LIST}")
    val_adata_path="${VAL_ADATA_ROOT_PATH}/${file}"
    echo "Running metadata prediction for file ${val_adata_path} on GPU ${GPU_ID}"
    python3 metadata_prediction_analysis.py \
        --cuda_device_index ${GPU_ID} \
        --val_adata_path "${val_adata_path}" \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --ref_adata_path "${REF_ADATA_PATH}" \
        --gene_info_path "${GENE_INFO_PATH}" \
        --ontology_resource_path "${ONTOLOGY_RESOURCE_PATH}" \
        --output_path "${OUTPUT_PATH}" \
        --rng_seed ${RNG_SEED} \
        --n_genes ${N_GENES} \
        --gene_selection_method ${GENE_SELECTION_METHOD} \
        --fixed_prompt_vars_sublist_path ${FIXED_PROMPT_VARS_SUBLIST_PATH} \
        --rand_prompt_vars_sublist_path ${RAND_PROMPT_VARS_SUBLIST_PATH} \
        --chunk_size ${CHUNK_SIZE} \
        --metric_style ${METRIC_STYLE}
done
