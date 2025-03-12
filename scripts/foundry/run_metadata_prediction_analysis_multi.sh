#!/bin/bash

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <argument>"
  exit 1
fi

arg="$1"

CHECKPOINT_PATH="/mnt/cellariumgpt-xfer/mb_checkpoints/10M_001_bs1536/epoch=1-step=29161__updated.ckpt"
OUTPUT_PATH="/mnt/cellariumgpt-inference/metadata_predictions_high__4096_with_XY/10M_001_bs1536"
./run_metadata_prediction_analysis_with_XY.sh "$arg" ${CHECKPOINT_PATH} ${OUTPUT_PATH}

CHECKPOINT_PATH="/mnt/cellariumgpt-xfer/mb_checkpoints/19M_001_bs2048/epoch=1-step=28244__updated.ckpt"
OUTPUT_PATH="/mnt/cellariumgpt-inference/metadata_predictions_high__4096_with_XY/19M_001_bs2048"
./run_metadata_prediction_analysis_with_XY.sh "$arg" ${CHECKPOINT_PATH} ${OUTPUT_PATH}

CHECKPOINT_PATH="/mnt/cellariumgpt-xfer/mb_checkpoints/30M_001_bs2560/epoch=2-step=43129__updated.ckpt"
OUTPUT_PATH="/mnt/cellariumgpt-inference/metadata_predictions_high__4096_with_XY/30M_001_bs2560"
./run_metadata_prediction_analysis_with_XY.sh "$arg" ${CHECKPOINT_PATH} ${OUTPUT_PATH}

CHECKPOINT_PATH="/mnt/cellariumgpt-xfer/mb_checkpoints/59M_001_bs3072/epoch=3-step=53770__updated.ckpt"
OUTPUT_PATH="/mnt/cellariumgpt-inference/metadata_predictions_high__4096_with_XY/59M_001_bs3072"
./run_metadata_prediction_analysis_with_XY.sh "$arg" ${CHECKPOINT_PATH} ${OUTPUT_PATH}

CHECKPOINT_PATH="/mnt/cellariumgpt-xfer/mb_checkpoints/10M_001_bs1536/epoch=1-step=29161__updated.ckpt"
OUTPUT_PATH="/mnt/cellariumgpt-inference/metadata_predictions_high__4096_without_XY/10M_001_bs1536"
./run_metadata_prediction_analysis_without_XY.sh "$arg" ${CHECKPOINT_PATH} ${OUTPUT_PATH}

CHECKPOINT_PATH="/mnt/cellariumgpt-xfer/mb_checkpoints/19M_001_bs2048/epoch=1-step=28244__updated.ckpt"
OUTPUT_PATH="/mnt/cellariumgpt-inference/metadata_predictions_high__4096_without_XY/19M_001_bs2048"
./run_metadata_prediction_analysis_without_XY.sh "$arg" ${CHECKPOINT_PATH} ${OUTPUT_PATH}

CHECKPOINT_PATH="/mnt/cellariumgpt-xfer/mb_checkpoints/30M_001_bs2560/epoch=2-step=43129__updated.ckpt"
OUTPUT_PATH="/mnt/cellariumgpt-inference/metadata_predictions_high__4096_without_XY/30M_001_bs2560"
./run_metadata_prediction_analysis_without_XY.sh "$arg" ${CHECKPOINT_PATH} ${OUTPUT_PATH}

CHECKPOINT_PATH="/mnt/cellariumgpt-xfer/mb_checkpoints/59M_001_bs3072/epoch=3-step=53770__updated.ckpt"
OUTPUT_PATH="/mnt/cellariumgpt-inference/metadata_predictions_high__4096_without_XY/59M_001_bs3072"
./run_metadata_prediction_analysis_without_XY.sh "$arg" ${CHECKPOINT_PATH} ${OUTPUT_PATH}
