#!/bin/bash
#SBATCH -J nmallina-cellarium
#SBATCH -o /work/hdd/bbjr/mallina1/nn-cellarium-logs/%j.log

#SBATCH -p gpuA100x4
#SBATCH --account bbjr-delta-gpu
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 24
#SBATCH --mem 48G
#SBATCH --gpus-per-node 1
#SBATCH --time 04:00:00

source /u/mallina1/envs/torch_jax2/bin/activate
cd /u/mallina1/research/cellarium-ml/notebooks/embeddings

# target_assay=$1
# target_assay = '10x Chromium (v3)'
# target_assay = '10x Chromium (v2) A'
# target_assay = '10x Chromium (v2) B'
# target_assay = 'Drop-seq'
# target_assay = 'Seq-Well'
# target_assay = 'inDrops'
python 06_03_assay_conversion.py
