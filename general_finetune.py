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

parser = argparse.ArgumentParser(description='Fine-Tuning Torch Language Models')

# Fine-tuning specifics
available_models = ["google/flan-t5-large", "google/flan-t5-base", "google/flan-t5-small", "microsoft/mdeberta-v3-base"]
parser.add_argument('--model-name', choices=available_models, default="google/flan-t5-large")
parser.add_argument('--lr', type=float, default=5e-4)

# Misc.
parser.add_argument('--trial', default=1)
parser.add_argument('--batch-size', default=1)
parser.add_argument('--device', default="cuda:0")

args = parser.parse_args()

# Save model checkpoints at this epoch
def save_model_checkpoints(model, epoch, train_loss, val_loss, save_dir):
    new_save_dir = save_dir + f"epoch_{epoch}/"
    Path(new_save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(new_save_dir, from_pt=True)

    with open(new_save_dir + "model_metrics.txt", "w") as f:
        f.write(f"==========EPOCH {epoch}==========\n")
        f.write(f"avg per sample train loss: {train_loss}\n")
        f.write(f"avg per sample val loss: {val_loss}\n")

# Training loop helper function
def train_batch(model, train_data_loader, val_data_loader, args):
    train_data_loader_len = len(train_data_loader.dataset)
    val_loss_arr = []
    train_loss_arr = []
    
    # Hyperparameters
    lr = args.lr
    num_epochs = 3
    num_training_steps = num_epochs * len(train_data_loader)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize optimizers
    device = args.device
    model.to(device)
    optimizer = AdamW(model.parameters(), lr = lr)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Initiate training loop
    print("Starting the fine-tuning process...")
    for epoch in range(num_epochs):
        model.train()   # Enter training mode
        train_loss_total = 0.0

        tqdm._instances.clear()
        progress_bar = tqdm(range(len(train_data_loader)), position=0, leave=True, ascii=True)
        print(f"\nEpoch [{epoch}/{num_epochs}]")
        for index, batch in enumerate(train_data_loader):
            batch = [b.to(device) for b in batch]
            input_ids, attention_mask, labels = batch

            # Compute loss
            logits = model(input_ids = input_ids, attention_mask = attention_mask).logits
            loss = criterion(logits, labels.long())
            train_loss_total += (loss * len(labels))

            # Update parameters
            model.zero_grad()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
        
        model.eval() # Enter evaluation mode
        train_loss_value = train_loss_total.detach().cpu().numpy() / train_data_loader_len
        val_loss_value = val_batch(model, val_data_loader, args)
        train_loss_arr.append(train_loss_value)
        val_loss_arr.append(val_loss_value)

        # Save model checkpoints at this epoch
        save_model_checkpoints(model, epoch, train_loss_value, val_loss_value, args.save_dir)

    return model, train_loss_arr, val_loss_arr

# Validation loop helper function
def val_batch(model, val_data_loader, args):
    val_data_loader_len = len(val_data_loader.dataset)
    val_loss_total = 0.0
    
    criterion = torch.nn.CrossEntropyLoss()
    device = args.device
    model.to(device)

    # Initiate loop
    model.eval()
    preds = []

    print("\nStarting the validation process...")
    tqdm._instances.clear()
    progress_bar = tqdm(range(len(val_data_loader)), position=0, leave=True, ascii=True)
    for index, batch in enumerate(val_data_loader):
        batch = [b.to(device) for b in batch]
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            logits = model(input_ids = input_ids, attention_mask = attention_mask).logits

        # Compute validation loss
        val_loss = criterion(logits, labels.long())
        val_loss_total += (val_loss * len(labels))
        progress_bar.update(1)

    return val_loss_total.cpu().numpy() / val_data_loader_len

# System prompt not needed when training with <1B param models
batch_size = args.batch_size

# Load model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data_collections.benchmark_datasets import BenchmarkDataset

model_id = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, trust_remote_code=True)
model.eval()

# Custom datset processing
benchmark_file = "/home/dennis/prompt-injection-detection/data/training_data/2024-12-07/2024-12-07_20k_full.json"
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
filtered_idx = [i for i in range(len(encoded_dataset)) if torch.sum(encoded_dataset[i][1]) < 512]
encoded_dataset = torch.utils.data.Subset(encoded_dataset, filtered_idx)

# Create a small validation set
rng = np.random.default_rng(12345)
idx_list = np.arange(len(encoded_dataset))
val_idx = rng.choice(idx_list, 1000, replace=False)
train_idx = np.array(list(set(idx_list) - set(val_idx)))

val_data_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(encoded_dataset, val_idx), batch_size=int(batch_size), shuffle=False)
train_data_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(encoded_dataset, train_idx), batch_size=int(batch_size), shuffle=True)

# Create a folder in which to save results
todaystring = date.today().strftime("%Y-%m-%d")
args.save_dir = f"small_finetuned_models/small_finetuned_models/{todaystring}/{model_id}/trial_{args.trial}_lr_{args.lr}/"
Path(args.save_dir).mkdir(parents=True, exist_ok=True)

# Perform training and validation
model, train_loss_arr, val_loss_arr = train_batch(model, train_data_loader, val_data_loader, args)

# Save the final metrics
print(f"train_loss_arr: {train_loss_arr}\n")
print(f"val_loss_arr: {val_loss_arr}")

