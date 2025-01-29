from tqdm.auto import tqdm
import json
import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.utils.data import Subset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser(description='Eval Torch Language Models')

# Evaluation specifics
available_models = ["google/flan-t5-large", "google/flan-t5-base", "google/flan-t5-small", "microsoft/mdeberta-v3-base"]
parser.add_argument('--model-name', choices=available_models, default="google/flan-t5-large")
parser.add_argument('--model-path')

# Misc.
parser.add_argument('--trial', default=1)
parser.add_argument('--batch-size', default=1)
parser.add_argument('--device', default="cuda:0")

args = parser.parse_args()

# Eval loop helper function
def eval_batch(model, eval_data_loader, args):
    eval_data_loader_len = len(eval_data_loader.dataset)

    # Initialize optimizers
    device = args.device
    model.to(device)

    # Initialize softmax
    softmax = torch.nn.functional.softmax

    # Initiate eval loop
    scores_prompt_injection = []
    model.eval()
    print("\nStarting the evaluation process...")
    tqdm._instances.clear()
    progress_bar = tqdm(range(len(eval_data_loader)), position=0, leave=True, ascii=True)
    for index, batch in enumerate(eval_data_loader):
        batch = [b.to(device) for b in batch]
        input_ids, attention_mask, _ = batch

        with torch.no_grad():
            logits = model(input_ids = input_ids, attention_mask = attention_mask).logits
        
        normalized_logits = softmax(logits, dim=1)
        score = normalized_logits[:, -1].cpu().numpy()

        # Save score
        scores_prompt_injection.extend(score)
        progress_bar.update(1)
        
    return scores_prompt_injection

# System prompt not needed when evaluating with <1B param models
batch_size = args.batch_size

# Load model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data_collections.benchmark_datasets import BenchmarkDataset

model_id = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2, trust_remote_code=True)
model.eval()

# Custom datset processing - don't need to worry about filtering the dataset for evaluation
benchmark_file = "/home/dennis/prompt-injection-detection/data/evaluation_data/2024-11-28/2024-11-28_evaluation_benchmark.json"
try:
    data_collection = BenchmarkDataset(benchmark_file, dataset_partition="test")
except:
    print("Benchmark file not found, enter correct date (--file-date=2024-10-20)")
    raise
dataset, data_labels = data_collection.get_dataset()
print(f"There are a total of {len(dataset)} datapoints...\n")
encoded_dataset = data_collection.convert2torch(tokenizer=tokenizer, dataset=dataset, labels=data_labels)

# ####
# # Because the model_max_length is 512 for the Flan models, we remove all data with more than 512 tokens; we filter
# # using the attention masks from the tokenizer
# filtered_idx = [i for i in range(len(encoded_dataset)) if torch.sum(encoded_dataset[i][1]) < 512]
# encoded_dataset = torch.utils.data.Subset(encoded_dataset, filtered_idx)
# ####

# Perform evaluation
eval_data_loader = torch.utils.data.DataLoader(encoded_dataset, batch_size=int(batch_size), shuffle=False)
scores_prompt_injection = eval_batch(model, eval_data_loader, args)

# Save scores
dataset_str = "evaluation_benchmark"
labels = (data_collection.get_labels()).numpy()

#outputs = {"model_name": model_id, "dataset_name": dataset_str, "scores_prompt_injection": scores_prompt_injection, "labels": labels[filtered_idx]}
outputs = {"model_name": model_id, "dataset_name": dataset_str, "scores_prompt_injection": scores_prompt_injection, "labels": labels}
outputs_dir = f"{args.model_path}/evaluations/{dataset_str}/trial_{args.trial}/"
Path(outputs_dir).mkdir(parents=True, exist_ok=True)
np.savez(outputs_dir + f"{dataset_str}_outputs.npz", **outputs)

