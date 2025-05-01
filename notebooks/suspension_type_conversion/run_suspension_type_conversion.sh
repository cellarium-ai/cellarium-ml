#!/bin/bash
#SBATCH -J nmallina-cellarium
#SBATCH -o /work/hdd/bbjr/mallina1/nn-cellarium-logs/%j.log

#SBATCH -p gpuA100x4
#SBATCH --account bbjr-delta-gpu
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 48G
#SBATCH --gpus-per-node 1
#SBATCH --time 12:00:00

source /u/mallina1/envs/torch_jax2/bin/activate
cd /u/mallina1/research/cellarium-ml/notebooks/suspension_type_conversion

python 02_01_suspension_type_conversion.py
