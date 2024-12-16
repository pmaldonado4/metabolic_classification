#!/bin/bash
#SBATCH --job-name="hyena_inference"
#SBATCH --output="x_inference_%j.out"
#SBATCH --error="x_inference_%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --account=bdhi-delta-gpu

# Source Conda environment setup
source /sw/external/python/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate /u/pmaldonadocatala/.conda/envs/ml_env_2

# Confirm the environment is activated
echo "=== ENVIRONMENT ACTIVATION CHECK ==="
echo "Using Python: $(which python)"
echo "Python Version: $(python --version)"
echo "Conda Environment: $(conda info --envs | grep '*')"

# Confirm Torch is Available
python -c "import torch; print('Torch Version:', torch.__version__)"

# Run the inference script
srun python inference_run.py