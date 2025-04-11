#!/bin/bash
#SBATCH -J nmallina-cellarium
#SBATCH -o /work/hdd/bbjr/mallina1/nn-cellarium-logs/%j.log

#SBATCH -p gpuA100x4
#SBATCH --account bbjr-delta-gpu
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 32G
#SBATCH --gpus-per-node 1
#SBATCH --time 02:00:00

source /u/mallina1/envs/torch_jax2/bin/activate
cd /u/mallina1/research/cellarium-ml/notebooks/assay_conversion

target_assay=$1
batch_size=$2
n_genes=$3
max_counts=$4
min_umis=$5
max_umis=$6
# target_cell=$7
python 01_assay_conversion.py \
    --target_assay="${target_assay}" \
    --batch_size="${batch_size}" \
    --n_fixed_query_genes="${n_genes}" \
    --max_counts="${max_counts}" \
    --min_umis="${min_umis}" \
    --max_umis="${max_umis}" 
