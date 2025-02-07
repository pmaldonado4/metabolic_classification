#!/bin/bash
#SBATCH --job-name=multi_node_inference
#SBATCH --nodes=4                # Number of nodes
#SBATCH --ntasks-per-node=4      # Number of tasks per node (GPUs per node)
#SBATCH --gpus=4        # GPUs per node
#SBATCH --cpus-per-task=8        # CPUs per GPU
#SBATCH --time=24:00:00          # Job time limit
#SBATCH --partition=gpuA40x4             # Partition to use 
#SBATCH --output=job_%j.out      # Output file
#SBATCH --error=job_%j.err       # Error file
#SBATCH --account=bdhi-delta-gpu

# Source Conda environment setup

source /sw/external/python/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate /u/pmaldonadocatala/.conda/envs/ml_env_2


# Set dynamic master port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((10000 + RANDOM % 10000))

export WORLD_SIZE=$(($SLURM_NTASKS))  # Total tasks across nodes

# Run the script
srun python inference_run_multi_node.py