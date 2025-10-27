#!/bin/bash

#SBATCH --job-name=eval_finetuned_prompt_injection_detector
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=djacob18@berkeley.edu

# Model parameters
MODEL_NAME="google/flan-t5-large"
MODEL_PATH="/home/dennis/prompt-injection-detection/small_finetuned_models/small_finetuned_models/2025-03-26/google/flan-t5-large/trial_with_newline_train_lr_5e-05/epoch_1"

# Misc.
TRIAL=2
BATCH_SIZE=1

# GPU info
DEVICE="cuda:0"

conda run -n torch_env python eval_general_finetune.py --model-name $MODEL_NAME --model-path $MODEL_PATH --trial $TRIAL --batch-size $BATCH_SIZE --device $DEVICE