import argparse

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Compare JSON and HF datasets')

# Misc.
parser.add_argument('--trial', default=1)
parser.add_argument('--batch-size', default=1)
parser.add_argument('--device', default="cuda:0")

args = parser.parse_args()

# Custom dataset class for HuggingFace dataset
class HuggingFaceDataset(Dataset):
    def __init__(self, tokenizer, dataset, labels):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.labels = labels
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['prompt']
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        return (
            encoding['input_ids'].squeeze(),
            encoding['attention_mask'].squeeze(),
            torch.tensor(label, dtype=torch.long)
        )

# System prompt not needed when evaluating with <1B param models
batch_size = args.batch_size

# Load model
model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

# Load dataset from HuggingFace
print("Loading dataset from HuggingFace...")
hf_dataset = load_dataset("hendzh/PromptShield", split="test")
hf_prompts = [item['prompt'] for item in hf_dataset]
hf_labels = [item['label'] for item in hf_dataset]
print(f"There are a total of {len(hf_prompts)} datapoints in HF...\n")

hf_encoded_dataset = HuggingFaceDataset(tokenizer=tokenizer, dataset=hf_dataset, labels=hf_labels)

# Load in data from JSON
from utils.data_collections.benchmark_datasets import BenchmarkDataset
benchmark_file = "/home/dennis/prompt-injection-detection/camera_ready_datasets/en_dataset_no_dups/2024-11-28_evaluation_benchmark_en.json"
try:
    json_data_collection = BenchmarkDataset(benchmark_file, dataset_partition="test")
except:
    print("Benchmark file not found, enter correct date (--file-date=2024-10-20)")
    raise
json_prompts, json_labels = json_data_collection.get_dataset()
print(f"There are a total of {len(json_prompts)} datapoints in JSON...\n")
json_encoded_dataset = json_data_collection.convert2torch(tokenizer=tokenizer, dataset=json_prompts, labels=json_labels)

def check_dataset_subset_str(hf_dataset, json_dataset):
    hf_dict = {}
    for hf_data in hf_dataset:
        res = hf_dict.get(hf_data)
        if res is not None:
            hf_dict[hf_data] = res + 1
        else:
            hf_dict[hf_data] = 1

    json_dict = {}
    for json_data in json_dataset:
        res = json_dict.get(json_data)
        if res is not None:
            json_dict[json_data] = res + 1
        else:
            json_dict[json_data] = 1

    return hf_dict, json_dict

json_combined_prompts = []
for data in json_prompts:
    combined_prompt = (data["instruction"] + "\n" + data["input"]) if data["input"] != "" else data["instruction"]
    json_combined_prompts.append(combined_prompt)

# Check subset relationship
hf_dict, json_dict = check_dataset_subset_str(hf_prompts, json_combined_prompts)