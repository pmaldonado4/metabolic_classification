#!/bin/bash
#SBATCH --job-name="PCA_plot"
#SBATCH --output="PCA_plot_%j.out"
#SBATCH --error="PCA_plot_%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --account=bdhi-delta-gpu

# Set library path for correct `libstdc++.so.6`
export LD_LIBRARY_PATH=/sw/external/cpe/cpe-23.03-rhel-8-6-rpm/opt/cray/pe/gcc/12.2.0/snos/lib64:$LD_LIBRARY_PATH

# Source Conda environment setup
source /sw/external/python/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate /u/pmaldonadocatala/.conda/envs/ml_env_2

# Run the inference script
srun python pca_plot.py