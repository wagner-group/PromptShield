from tqdm.auto import tqdm
import argparse
from datetime import date
from pathlib import Path
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from hashlib import sha256
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    print(model_name_list)
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
save_dir = f"eval_plots/{dataset_name}/{todaystring}/{args.trial}/{model_name}_thresholds/"
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Convert to numpy
dataset_names = np.array(dataset_name_list)
scores_prompt_injection = np.array(scores_prompt_injection)
labels = np.array(labels).astype("int")

#########################################################################
#                           Plotting stage...                           #
#########################################################################
######################################################################
#------------------------
#Thresholds from evaluating app-only dataset (table 8)
#Model 76:
# Threshold at target 0.05000000% FPR and computed 0.04207869% FPR is: 99.99975261%
# Threshold at target 0.10000000% FPR and computed 0.09467705% FPR is: 99.99966320%
# Threshold at target 0.50000000% FPR and computed 0.48390490% FPR is: 99.99904913%
# Threshold at target 1.00000000% FPR and computed 0.99936882% FPR is: 99.99796083%

#Model 77:
# Threshold at target 0.05000000% FPR and computed 0.04207869% FPR is: 99.99910888%
# Threshold at target 0.10000000% FPR and computed 0.09467705% FPR is: 99.99896877%
# Threshold at target 0.50000000% FPR and computed 0.49442457% FPR is: 99.99459925%
# Threshold at target 1.00000000% FPR and computed 0.99936882% FPR is: 99.95612916%

low_fprs = ["0.05", "0.1", "0.5", "1"]
interpolated_thres = [0.9999975261, 0.9999966320, 0.9999904913, 0.9999796083]

results = []


# Helper function to parse results from the Predibase model based on an explicit threshold
def parse_results(scores_prompt_injection, threshold):
    preds = []

    for score in scores_prompt_injection:
        # If the score is higher or equal to threshold, count it as 1, 0 otherwise
        new_value = 1 if score >= threshold else 0 
        preds.append(int(new_value))

    return preds

def calculate_tpr_fpr(confusion_matrix):
    # Extract values from the confusion matrix
    TN, FP, FN, TP = confusion_matrix.ravel().astype(np.float64)

    if TP == 0 and FN == 0: #(benign dataset)
        return 0, (FP / (FP + TN))

    # Calculate TPR (True Positive Rate)
    TPR = TP / (TP + FN)

    # Calculate FPR (False Positive Rate)
    FPR = FP / (FP + TN)

    return TPR, FPR


#plot and save confusion matrices at different thresholds
with open(save_dir + "thresholds.txt", "w") as f:

    for index, threshold in enumerate(interpolated_thres):
        preds = parse_results(scores_prompt_injection, interpolated_thres[index])
        preds = np.array(preds)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "injection"])
        disp.plot().figure_.savefig(save_dir + low_fprs[index]+'confusion_matrix.png', bbox_inches='tight')
        # Calculate TPR and FPR
        tpr, fpr = calculate_tpr_fpr(cm)
        if tpr == 0: #benign data, no TPR
            print(f"FPR for benign data at {low_fprs[index]}% FPR is {fpr*100:g}%", file=f)
        else: 
            print(f"TPR at {fpr*100.0:g}% at FPR is {tpr*100:g}%", file=f)
