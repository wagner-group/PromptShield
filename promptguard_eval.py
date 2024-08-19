import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# from transformers import pipeline
# pipeline_classifier = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M")
# output = pipeline_classifier("test")

# Evaluation loop helper function
def evaluate_batch(model, dataset, batch_size=32):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Initiate loop
    preds = []
    for index, batch in enumerate(data_loader):
        print(f"Currently at batch {index} / {len(data_loader)}")

        input_ids, attention_mask = batch

        with torch.no_grad():
            logits = model(input_ids = input_ids, attention_mask = attention_mask).logits
        
        # Determine predicted class, ignoring the distinction between "benign" and "injection"
        predicted_class_id = (logits.argmax(dim=1).numpy() > 1).astype(int)
        # predicted_class_prob = torch.squeeze(torch.nn.functional.softmax(logits, dim=1))[predicted_class_id]
        # predicted_class = model.config.id2label[predicted_class_id]
        preds.extend(predicted_class_id)

    return preds

# Load the model
model_id = "meta-llama/Prompt-Guard-86M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

# Set up Synapse AI dataset
dataset = load_dataset("synapsecai/synthetic-prompt-injections")
test_dataset = dataset['test'].select(range(1500))
encoded_texts = tokenizer(test_dataset["text"], return_tensors="pt", max_length=512, padding=True, truncation=True)
labels = [int(d) for d in test_dataset["label"]]

# Hend's custom dataset: The "proper" way to do this is to make a correpsonding Dataset sub-class and describe the get_item() method!!!

# with open('../custom_datasets/combined_dataset.json') as f:
#     dataset = json.load(f)

# #dataset = dataset[:500] # take a slice of the first 500 elements
# text_inputs = [d['instruction'] + "\n" + d['input'] for d in dataset]
# encoded_texts = tokenizer(text_inputs, return_tensors="pt", max_length=512, padding=True, truncation=True)
# labels = [d['flag'] for d in dataset]

# Perform evaluation
encoded_dataset = torch.utils.data.TensorDataset(encoded_texts['input_ids'], encoded_texts['attention_mask'])
preds = evaluate_batch(model, encoded_dataset)

# Visualization
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign/injection", "jailbreak"])
disp.plot().figure_.savefig('confusion_matrix1.png', bbox_inches='tight')