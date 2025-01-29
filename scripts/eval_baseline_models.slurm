#!/bin/bash

#SBATCH --job-name=eval_baseline_prompt_injection_detector
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=djacob18@berkeley.edu

# Model parameters
MODEL_NAME="protectai-v2"

# Misc.
TRIAL=2
BATCH_SIZE=1

# GPU info
DEVICE="cuda:0"

conda run -n torch_env python eval_baselines.py --model $MODEL_NAME --trial $TRIAL --batch-size $BATCH_SIZE --device $DEVICE