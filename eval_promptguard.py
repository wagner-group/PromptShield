from tqdm.auto import tqdm
import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.benign_datasets import LMSYS, UltraChat, NaturalInstructions

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser(description='PromptGuard Model Evaluation')

# Evaluation specifics
parser.add_argument('--dataset-name', choices=["lmsys", "natural-instructions", "ultrachat"], default="lmsys")
parser.add_argument('--offset', default=0)
parser.add_argument('--trial', default=1)
parser.add_argument('--batch-size', default=4)

args = parser.parse_args()

# Evaluation loop helper function
def evaluate_batch(model, data_loader):

    device = "cuda:0"
    model.to(device)

    # Initiate loop
    preds = []
    model.eval()
    print("\nStarting the evaluation process...")
    tqdm._instances.clear()
    progress_bar = tqdm(range(len(data_loader)), position=0, leave=True, ascii=True)
    for index, batch in enumerate(data_loader):
        batch = [b.to(device) for b in batch]
        input_ids, attention_mask, _ = batch

        with torch.no_grad():
            logits = model(input_ids = input_ids, attention_mask = attention_mask).logits
        
        # Determine predicted class, ignoring the distinction between "benign" and "injection"
        predicted_class_id = (logits.argmax(dim=1).cpu().numpy() > 1).astype(int)
        preds.extend(predicted_class_id)
        progress_bar.update(1)

    return preds

# Set up dataset
offset = int(args.offset)
if args.dataset_name == "lmsys":
    toxicity_threshold = 0.01
    data_collection = LMSYS("train", 1500, random_sample=True, toxicity_threshold=toxicity_threshold, offset=offset)
    if toxicity_threshold < 1.0:
        dataset_str = f"lmsys/content_moderated/thre_{toxicity_threshold * 100:g}_percent"
    else:
        dataset_str = f"lmsys/vanilla"
    
elif args.dataset_name == "natural-instructions":
    data_collection = NaturalInstructions("train", 1500, random_sample=True)
    dataset_str = "natural-instructions"

dataset = data_collection.get_dataset()
print(f"There are a total of {len(dataset)} datapoints...\n")

# Create a folder in which to save results
model_str = "promptguard"
todaystring = date.today().strftime("%Y-%m-%d")
save_dir = f"dataset_evals/{dataset_str}/{todaystring}/trial_{args.trial}_{model_str}_offset_{offset}/"
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Load the model
model_id = "meta-llama/Prompt-Guard-86M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

# Perform evaluation
encoded_dataset = data_collection.convert2torch(tokenizer=tokenizer)
data_loader = torch.utils.data.DataLoader(encoded_dataset, batch_size=int(args.batch_size))

preds = evaluate_batch(model, data_loader)

# Visualization
labels = data_collection.get_labels()
cm = confusion_matrix(labels.numpy(), np.array(preds))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "injection"])
disp.plot().figure_.savefig(save_dir + 'confusion_matrix.png', bbox_inches='tight')