#!/bin/bash

#SBATCH --job-name=finetuning_prompt_injection_detector
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=22:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=djacob18@berkeley.edu

# Model parameters
MODEL_NAME="google/flan-t5-large"
LR_ARR=(5e-6 5e-5 5e-4)

# Misc.
TRIAL=1

# GPU info
DEVICE="cuda:0"

for LR in ${LR_ARR[@]};
do
    conda run -n torch_env python general_finetune.py --model-name $MODEL_NAME --lr $LR --trial $TRIAL --device $DEVICE
done