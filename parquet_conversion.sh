#!/bin/bash
#SBATCH --job-name="PCA_job"
#SBATCH --output="x_PCA_%j.out"
#SBATCH --error="x_PCA_%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --account=bdhi-delta-gpu

# Source Conda environment setup
source /sw/external/python/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate /u/pmaldonadocatala/.conda/envs/ml_env_2

# Run the inference script
srun python parquet_conversion.py