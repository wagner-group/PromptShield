#!/bin/bash

#SBATCH --job-name=finetuning_prompt_injection_detector
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --time=7:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=djacob18@berkeley.edu

# Model parameters
#MODEL_NAME_ARR=("google/flan-t5-base" "google/flan-t5-small")
MODEL_NAME="google/flan-t5-large"
#LR_ARR=(5e-6 5e-5)
LR_ARR=(5e-7)

# Misc.
TRIAL=2
BATCH_SIZE=1

# GPU info
DEVICE="cuda:0"

for LR in ${LR_ARR[@]};
do
    conda run -n torch_env python general_finetune.py --model-name $MODEL_NAME --lr $LR --trial $TRIAL --batch-size $BATCH_SIZE --device $DEVICE
done

#for MNAME in ${MODEL_NAME_ARR[@]};
#do
#    for LR in ${LR_ARR[@]};
#    do
#        python general_finetune.py --model-name $MNAME --lr $LR --trial $TRIAL --batch-size $BATCH_SIZE --device $DEVICE
#    done
#done