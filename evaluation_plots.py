from tqdm.auto import tqdm
import argparse
from datetime import date
from pathlib import Path
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from hashlib import sha256
from utils.metrics import computeROC, interpolateROC, interpolateThre

parser = argparse.ArgumentParser(description='Plot ROC curve')

# Arguments
parser.add_argument('--output-dir')
parser.add_argument('--trial', default=1)

args = parser.parse_args()

#########################################################################
#                           Importing stage...                          #
#########################################################################

# Initiailize variables
model_name_list = []
dataset_name_list = []
scores_prompt_injection = []
labels = []

# Import outputs one file at a time
output_files = glob.glob(args.output_dir + os.sep + "*.npz")
for output_file in output_files:
    outputs = np.load(output_file)
    
    # Check model name and scores
    model_name_list.extend(outputs["model_name"].reshape(1,))
    scores_prompt_injection.extend(outputs["scores_prompt_injection"])

    # Check dataset labels and dataset name
    dataset_labels = outputs["labels"]
    labels.extend(dataset_labels)
    dataset_name_list.extend([outputs["dataset_name"].item() for i in range(len(dataset_labels))])

# Make sure outputs from different models are not imported
model_name_set = set(model_name_list)
if len(model_name_set) != 1: 
    raise Exception("The model used must be consistent throughout the set of outputs.") 

model_name = list(model_name_set)[0]

# Check if outputs from multiple datasets were imported
dataset_name_set = set(dataset_name_list)
sorted_datasets = sorted(list(dataset_name_set))
if len(dataset_name_set) == 1: 
    dataset_name = sorted_datasets[0]
else:
    combined_datasets_str = "+".join(sorted_datasets)
    combined_datasets_id = sha256((combined_datasets_str).encode('utf-8')).hexdigest()
    dataset_name = f"combined_dataset_{combined_datasets_id[0:6]}"

# Set up save path
todaystring = date.today().strftime("%Y-%m-%d")
save_dir = f"{args.output_dir}/plots/"
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Convert to numpy
dataset_names = np.array(dataset_name_list)
scores_prompt_injection = np.array(scores_prompt_injection)
labels = np.array(labels).astype("int")

#breakpoint()

#########################################################################
#                           Plotting stage...                           #
#########################################################################

def plot_score_histogram(ax, scores, bins, label):    
    # Plot only if scores exist
    if len(scores) > 0:
        ax.hist(scores, bins, alpha=0.5, weights=np.ones(len(scores),)/len(scores), label=label)

    return ax

def save_score_histogram(ax, title, save_dir):    
    ax.set_title(title)
    ax.set_xlabel("Score for injection class")
    ax.set_ylabel("Probability")
    ax.legend(loc='upper right')
    ax.figure.savefig(save_dir, bbox_inches="tight")

# Create histograms for scores corresponding to TP and TN
bins = np.linspace(0.0, 1.0, 100)
_, ax_tp = plt.subplots()
_, ax_tn = plt.subplots()
for ds_name in sorted_datasets:
    filtered_locations_tp = np.logical_and((labels == 1), (dataset_names == ds_name))
    filtered_locations_tn = np.logical_and((labels == 0), (dataset_names == ds_name))

    ax_tp = plot_score_histogram(ax_tp, scores_prompt_injection[filtered_locations_tp], bins, ds_name)
    ax_tn = plot_score_histogram(ax_tn, scores_prompt_injection[filtered_locations_tn], bins, ds_name)

# Save histograms
save_score_histogram(ax_tp, "Injection scores for TP", save_dir + "scores_tp_hist.png")
save_score_histogram(ax_tn, "Injection scores for TN", save_dir + "scores_tn_hist.png")

# Create ROC curve
low_fprs = [0.0005, 0.001, 0.005, 0.01]
num_targets = len(low_fprs)
num_targets = len(low_fprs)
fpr_list, tpr_list, thre, auc_score = computeROC(scores_prompt_injection, labels)

def compute_FPR(preds, labels):
    FP = np.sum((preds - labels) == 1)
    TN = np.sum((preds + labels) == 0)
    return FP / (FP + TN)

# Bisection scheme to find threshold that results in target FPR
def find_FPR_threshold(scores, labels, target_FPR, tolerance):
    low_thre = 0.0
    high_thre = 1.0

    for iteration in range(50):
        mid_thre = (low_thre + high_thre) / 2
        preds = (scores > mid_thre).astype("int")
        computed_FPR = compute_FPR(preds, labels)

        if np.abs(computed_FPR - target_FPR) < (tolerance * target_FPR):
            return mid_thre, computed_FPR 
        elif computed_FPR > target_FPR:
            low_thre = mid_thre
        else:
            high_thre = mid_thre 

    # If the loop completes without finding a threshold, return the last computed threshold
    return mid_thre, computed_FPR

interpolated_thres = [interpolateThre(fpr_val, fpr_list, thre) for fpr_val in low_fprs]
tolerance = 0.25
with open(save_dir + "thresholds_with_tolerance2.txt", "w") as f:
    for i in range(num_targets):
        initial_thres = interpolated_thres[i]
        preds = (scores_prompt_injection > initial_thres).astype("int")
        computed_FPR = compute_FPR(preds, labels)

        # Ensure that the thresholds result in a FPR value that is within 10% of the target
        if np.abs(computed_FPR - low_fprs[i]) > (tolerance * low_fprs[i]):
            initial_thres, computed_FPR = find_FPR_threshold(scores_prompt_injection, labels, low_fprs[i], tolerance)
            interpolated_thres[i] = initial_thres

        print(f"Threshold at target {low_fprs[i] * 100:.8f}% FPR and computed {computed_FPR * 100:.8f}% FPR is: {interpolated_thres[i] * 100:.8f}%", file=f)

# Compute TPR using interpolated thresholds
# computed_tpr_list = interpolateROC(fpr_list, tpr_list, low_fprs)
with open(save_dir + "low_fprs_with_tolerance2.txt", "w") as f:
    for i in range(num_targets):
        preds = (scores_prompt_injection > interpolated_thres[i]).astype("int")
        FN = np.sum((preds - labels) == -1)
        TP = np.sum((preds + labels) == 2)
        computed_TPR = TP / (TP + FN)

        print(f"TPR at target {low_fprs[i] * 100:g}% FPR is: {computed_TPR * 100:g}%", file=f)

# Plot and save ROC curve
fig = plt.figure()
plt.plot(fpr_list, tpr_list, 'b-', label="ROC curve")
plt.plot(fpr_list, fpr_list, 'k--', label="Random classifier")   # Random classifier
plt.xlabel("FPR")
plt.ylabel("Recall")
plt.title(f"ROC curve for {model_name} on {dataset_name} dataset")
plt.annotate(f"AUC: {auc_score:0.3f}", (0.45, 0.95))
plt.grid()
plt.legend()
plt.savefig(save_dir + "roc_curve.png", bbox_inches="tight")