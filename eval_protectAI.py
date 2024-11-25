from tqdm.auto import tqdm
import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.metrics import computeROC
from utils.data_collections.benchmark_datasets import BenchmarkDataset

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
import math


parser = argparse.ArgumentParser(description='PromptGuard Model Evaluation')

# Evaluation specifics
parser.add_argument('--offset', default=0)
parser.add_argument('--trial', default=1)
parser.add_argument('--batch-size', default=2)
parser.add_argument('--file-date', default="")

args = parser.parse_args()

# Evaluation loop helper function
def evaluate_batch(model, data_loader):

    device = "cuda:3"
    model.to(device)

    # Initialize softmax
    softmax = torch.nn.functional.softmax

    # Initiate loop
    preds = []
    scores_prompt_injection = []
    print("\nStarting the evaluation process...")
    tqdm._instances.clear()
    progress_bar = tqdm(range(len(data_loader)), position=0, leave=True, ascii=True)
    for index, batch in enumerate(data_loader):
        batch = [b.to(device) for b in batch]
        input_ids, attention_mask, _ = batch

        with torch.no_grad():
            logits = model(input_ids = input_ids, attention_mask = attention_mask).logits

        normalized_logits = softmax(logits, dim=1)
        
        predicted_class_id = (logits.argmax(dim=1).cpu().numpy()).astype(int)
        score = normalized_logits[:, -1].cpu().numpy()

        # Add values to overall list
        preds.extend(predicted_class_id)
        scores_prompt_injection.extend(score)
        progress_bar.update(1)

    return preds, scores_prompt_injection


offset = int(args.offset)
dataset_str = "evaluation_benchmark"

# Create a folder in which to save results
model_str = "protectAI"
todaystring = date.today().strftime("%Y-%m-%d")
save_dir = f"dataset_evals/{dataset_str}/{todaystring}/trial_{args.trial}_{model_str}_offset_{offset}/"
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Set up dataset
if args.file_date == "":
    benchmark_file = f"data/evaluation_data/{todaystring}/{todaystring}_evaluation_benchmark.json"
else:
    benchmark_file = f"data/evaluation_data/{args.file_date}/{args.file_date}_evaluation_benchmark.json"

try:
    data_collection = BenchmarkDataset(benchmark_file)
except:
    print("Benchmark file not found, enter correct date (--file-date=2024-10-20)")
    raise
dataset, data_labels = data_collection.get_dataset()
print(f"There are a total of {len(dataset)} datapoints...\n")



# Load the model
model_id = "meta-llama/Prompt-Guard-86M"
tokenizer = AutoTokenizer.from_pretrained("laiyer/deberta-v3-base-injection")
model = ORTModelForSequenceClassification.from_pretrained("laiyer/deberta-v3-base-injection")

#ONNX Runtime models, are by design in "evaluation mode" 
#model.eval()

all_preds = []
all_scores = []

subset_size = 5000

if(subset_size > len(dataset)):
    subset_size = len(dataset) 


n = math.ceil(len(dataset)/subset_size)

start_index = 0
end_index = 0+subset_size
counter = 0

#TODO find a better way to handle this
while end_index <= len(dataset):
    print(f"round: {counter} \n from {start_index} to {end_index}")
    subset_data = dataset.select(range(start_index,end_index))
    subset_labels = data_labels[start_index:end_index]

    encoded_dataset = data_collection.convert2torch(tokenizer=tokenizer, dataset=subset_data, labels=subset_labels )
    data_loader = torch.utils.data.DataLoader(encoded_dataset, batch_size=int(args.batch_size))
    preds, scores_prompt_injection = evaluate_batch(model, data_loader)
    all_preds.extend(preds)
    all_scores.extend(scores_prompt_injection)

    start_index= end_index
    if start_index == len(dataset):
        break
    end_index = (start_index + subset_size) if (start_index + subset_size) <=  len(dataset) else len(dataset)
    counter +=1



# Perform evaluation
# encoded_dataset = data_collection.convert2torch(tokenizer=tokenizer, dataset=dataset, labels=data_labels )
# data_loader = torch.utils.data.DataLoader(encoded_dataset, batch_size=int(args.batch_size))
# preds, scores_prompt_injection = evaluate_batch(model, data_loader)

# Visualization
labels = (data_collection.get_labels()).numpy()
fig = plt.figure(1)
cm = confusion_matrix(labels, np.array(all_preds), labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "injection"])
disp.plot().figure_.savefig(save_dir + 'confusion_matrix.png', bbox_inches='tight')

# Save scores
outputs = {"model_name": model_str, "dataset_name": dataset_str, "scores_prompt_injection": all_scores, "labels": labels}
outputs_dir = f"cached_outputs/{dataset_str}/{todaystring}/trial_{args.trial}_{model_str}/"
Path(outputs_dir).mkdir(parents=True, exist_ok=True)
np.savez(outputs_dir + f"{model_str}_{dataset_str}_outputs.npz", **outputs)
