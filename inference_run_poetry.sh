#!/bin/bash
#SBATCH --job-name="hyena_inference"
#SBATCH --output="x_inference_%j.out"
#SBATCH --error="x_inference_%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --account=bdhi-delta-gpu 
# Load modules
module load nvidia/24.5
module load python/3.11.6
module load cuda/11.8.0


# Run inference
srun poetry run python inference_run.py

