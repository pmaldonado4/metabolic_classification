#!/bin/bash
#SBATCH --job-name=hyena_inference       # Job name
#SBATCH --output=x_hyena_inference_%j.out # Output file
#SBATCH --error=x_hyena_inference_%j.err  # Error file
#SBATCH --partition=gpuA40x4             # Partition to use 
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --gpus=2                      # Number of GPUs per node
#SBATCH --mem=32G                       # Memory per node
#SBATCH --time=00:05:00                 # Time limit 
#SBATCH --export=ALL
#SBATCH --account=bdhi-delta-gpu
# Source Conda environment setup

source /sw/external/python/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate /u/pmaldonadocatala/.conda/envs/ml_env_2


# Print job details
echo "Job started on $(hostname) at $(date)"
echo "Using the following GPUs:"
nvidia-smi

# Set the number of GPUs
export WORLD_SIZE=2
export OMP_NUM_THREADS=2
# Run the Python script
srun torchrun --nproc_per_node=$WORLD_SIZE inference_run_multi_gpu.py

# Print job completion details
echo "Job finished at $(date)"