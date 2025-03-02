#!/bin/bash

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <argument>"
  exit 1
fi

arg="$1"

CHECKPOINT_PATH_1="/mnt/cellariumgpt-xfer/mb_checkpoints/10M_001_bs1536/epoch=1-step=29161__updated.ckpt"
OUTPUT_PATH_1="/mnt/cellariumgpt-inference/metadata_predictions_rand_4091/10M_001_bs1536"

CHECKPOINT_PATH_2="/mnt/cellariumgpt-xfer/mb_checkpoints/19M_001_bs2048/epoch=1-step=28244__updated.ckpt"
OUTPUT_PATH_2="/mnt/cellariumgpt-inference/metadata_predictions_rand_4091/19M_001_bs2048"

CHECKPOINT_PATH_3="/mnt/cellariumgpt-xfer/mb_checkpoints/30M_001_bs2560/epoch=2-step=43129__updated.ckpt"
OUTPUT_PATH_3="/mnt/cellariumgpt-inference/metadata_predictions_rand_4091/30M_001_bs2560"

CHECKPOINT_PATH_4="/mnt/cellariumgpt-xfer/mb_checkpoints/59M_001_bs3072/epoch=3-step=53770__updated.ckpt"
OUTPUT_PATH_4="/mnt/cellariumgpt-inference/metadata_predictions_rand_4091/59M_001_bs3072"

# Call the scripts with the provided argument
./run_metadata_prediction_analysis.sh "$arg" ${CHECKPOINT_PATH_1} ${OUTPUT_PATH_1}
./run_metadata_prediction_analysis.sh "$arg" ${CHECKPOINT_PATH_2} ${OUTPUT_PATH_2}
./run_metadata_prediction_analysis.sh "$arg" ${CHECKPOINT_PATH_3} ${OUTPUT_PATH_3}
./run_metadata_prediction_analysis.sh "$arg" ${CHECKPOINT_PATH_4} ${OUTPUT_PATH_4}
