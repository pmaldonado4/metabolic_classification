#!/bin/bash
#SBATCH --job-name="hyena_inference"
#SBATCH --account=bdhi-delta-gpu       # Replace with your actual account name
#SBATCH --output="inference_%j.out"
#SBATCH --error="inference_%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=8            # Allocate 8 CPUs per task
#SBATCH --gpus=2                     # Request 2 GPUs
#SBATCH --mem=64G                    # Memory allocation
#SBATCH --time=2:00:00               # Adjust time as needed
#SBATCH --mail-type=BEGIN,END,FAIL   # Notifications
#SBATCH --mail-user=pablo.maldonado@utah.edu

# Load necessary modules
module load nvidia/24.5
module load python/3.11.6
module load cuda/11.8.0

# Activate the Conda environment
source ~/miniconda3/bin/activate ml_env_2

# Set up the PyTorch distributed backend
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=2                  # Number of GPUs being used
export OMP_NUM_THREADS=1             # Prevent CPU oversubscription

# Run the inference script
srun python inference_run.py --batch_size 16 --max_length 4400 --gpus 2