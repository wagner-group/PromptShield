from tqdm.auto import tqdm
import argparse
from datetime import date
from pathlib import Path
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from hashlib import sha256
from utils.metrics import computeROC

parser = argparse.ArgumentParser(description='Plot ROC curve')

# Arguments
parser.add_argument('--output-dir')
parser.add_argument('--trial', default=1)

args = parser.parse_args()

# Initiailize variables
model_name_list = []
dataset_name_list = []
scores_prompt_injection = []
labels = []

# Import outputs one file at a time
output_files = glob.glob(args.output_dir + os.sep + "*.npz")
for output_file in output_files:
    outputs = np.load(output_file)
    
    model_name_list.extend(outputs["model_name"].reshape(1,))
    dataset_name_list.extend(outputs["dataset_name"].reshape(1,))
    scores_prompt_injection.extend(outputs["scores_prompt_injection"])
    labels.extend(outputs["labels"])

# Make sure outputs from different models are not imported
model_name_set = set(model_name_list)
if len(model_name_set) != 1: 
    raise Exception("The model used must be consistent throughout the set of outputs.") 

model_name = list(model_name_set)[0]

# Check if outputs from multiple datasets were imported
dataset_name_set = set(dataset_name_list)
if len(dataset_name_set) == 1: 
    dataset_name = list(dataset_name_set)[0]
else:
    sorted_datasets = sorted(list(dataset_name_set))
    combined_datasets_str = "+".join(sorted_datasets)
    combined_datasets_id = sha256((combined_datasets_str).encode('utf-8')).hexdigest()
    dataset_name = f"combined_dataset_{combined_datasets_id[0:6]}"

# Set up save path
todaystring = date.today().strftime("%Y-%m-%d")
save_dir = f"roc_curves/{dataset_name}/{todaystring}/trial_{args.trial}_{model_name}/"
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Plot ROC curve
fpr, tpr, thre, auc_score = computeROC(scores_prompt_injection, labels)
fig = plt.figure(1)
plt.plot(fpr, tpr, 'b-', label="ROC curve")
plt.plot(fpr, fpr, 'k--', label="Random classifier")   # Random classifier
plt.xlabel("FPR")
plt.ylabel("Recall")
plt.title(f"ROC curve for {model_name} on {dataset_name} dataset")
plt.annotate(f"AUC: {auc_score:0.3f}", (0.45, 0.95))
plt.grid()
plt.legend()
plt.savefig(save_dir + "roc_curve.png", bbox_inches="tight")