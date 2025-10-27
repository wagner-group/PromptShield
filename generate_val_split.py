from tqdm.auto import tqdm
import json
import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler

parser = argparse.ArgumentParser(description='Generate validation split')

# Model name
available_models = ["google/flan-t5-large", "google/flan-t5-base", "google/flan-t5-small", "microsoft/mdeberta-v3-base"]
parser.add_argument('--model-name', choices=available_models, default="google/flan-t5-base")

args = parser.parse_args()

# Load model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data_collections.benchmark_datasets import BenchmarkDataset

model_id = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, trust_remote_code=True)
model.eval()

# Custom datset processing
benchmark_file = "/home/dennis/prompt-injection-detection/data/training_data/2024-12-07/2024-12-07_20k_full_en.json"
try:
    data_collection = BenchmarkDataset(benchmark_file)
except:
    print("Benchmark file not found, enter correct date (--file-date=2024-10-20)")
    raise
dataset, data_labels = data_collection.get_dataset()
print(f"There are a total of {len(dataset)} datapoints...\n")
encoded_dataset = data_collection.convert2torch(tokenizer=tokenizer, dataset=dataset, labels=data_labels)

# Because the model_max_length is 512 for the Flan models, we remove all data with more than 512 tokens; we filter
# using the attention masks from the tokenizer
model_max_length = tokenizer.model_max_length
filtered_idx = [i for i in range(len(encoded_dataset)) if torch.sum(encoded_dataset[i][1]) < model_max_length]

# Create indices for a small validation set of 1000 samples
rng = np.random.default_rng(12345)
idx_list = np.arange(len(filtered_idx))
val_idx = rng.choice(idx_list, 1000, replace=False)
train_idx = list(set(idx_list) - set(val_idx))

# Filter and store the splits
dataset_after_filtering = [dataset[i] for i in filtered_idx]
for elem in dataset_after_filtering:
    del elem["labels"]

val_dataset = [dataset_after_filtering[i] for i in val_idx]
train_dataset = [dataset_after_filtering[i] for i in train_idx]

import json
with open("/home/dennis/prompt-injection-detection/data/training_data/2024-12-07/val_split/flan_val_2024-12-07_20k_full_en.json", "w") as f:
    json.dump(val_dataset, f, indent=5)

with open("/home/dennis/prompt-injection-detection/data/training_data/2024-12-07/val_split/flan_train_2024-12-07_20k_full_en.json", "w") as f:
    json.dump(train_dataset, f, indent=5)