#!/bin/bash
#SBATCH --job-name="Merge_job"
#SBATCH --output="x_Merge_%j.out"
#SBATCH --error="x_Merge_%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --account=bdhi-delta-gpu

# Source Conda environment setup
source /sw/external/python/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate /u/pmaldonadocatala/.conda/envs/ml_env_2

# Run the inference script
srun python merge_embeddings_features.py