from tqdm.auto import tqdm
import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.collections.benign_datasets import LMSYS, NaturalInstructions
from utils.collections.prompt_injection_datasets import SPMLChatbotPromptInjection, OpenPromptInjection
from utils.collections.training_datasets import StruQAttacks
from utils.metrics import computeROC

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser(description='PromptGuard Model Evaluation')

# Evaluation specifics
parser.add_argument('--dataset-name', choices=["lmsys", "natural-instructions", "spml", "StruQ-SPP", "usenix"], default="lmsys")
parser.add_argument('--offset', default=0)
parser.add_argument('--trial', default=1)
parser.add_argument('--batch-size', default=4)

args = parser.parse_args()

# Evaluation loop helper function
def evaluate_batch(model, data_loader):

    device = "cuda:2"
    model.to(device)

    # Initialize softmax
    softmax = torch.nn.functional.softmax

    # Initiate loop
    preds = []
    scores_prompt_injection = []
    model.eval()
    print("\nStarting the evaluation process...")
    tqdm._instances.clear()
    progress_bar = tqdm(range(len(data_loader)), position=0, leave=True, ascii=True)
    for index, batch in enumerate(data_loader):
        batch = [b.to(device) for b in batch]
        input_ids, attention_mask, _ = batch

        with torch.no_grad():
            logits = model(input_ids = input_ids, attention_mask = attention_mask).logits

        normalized_logits = softmax(logits, dim=1)
        
        # Determine predicted class, ignoring the distinction between "benign" and "injection"
        predicted_class_id = (logits.argmax(dim=1).cpu().numpy() > 1).astype(int)
        score = normalized_logits[:, -1].cpu().numpy()

        # Add values to overall list
        preds.extend(predicted_class_id)
        scores_prompt_injection.extend(score)
        progress_bar.update(1)

    return preds, scores_prompt_injection

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
elif args.dataset_name == "spml":
    data_collection = SPMLChatbotPromptInjection("train", 1500, random_sample=True)
    dataset_str = "spml"
elif args.dataset_name == "StruQ-SPP":
    data_collection = StruQAttacks(1500, seed_dataset_name="SPP")
    dataset_str = "StruQ-SPP"
elif args.dataset_name == "usenix":
    data_collection = OpenPromptInjection(2000, random_sample=True)
    dataset_str = "usenix"

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
preds, scores_prompt_injection = evaluate_batch(model, data_loader)

# Visualization
labels = (data_collection.get_labels()).numpy()
fig = plt.figure(1)
cm = confusion_matrix(labels, np.array(preds), labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "injection"])
disp.plot().figure_.savefig(save_dir + 'confusion_matrix.png', bbox_inches='tight')

# Save scores
outputs = {"model_name": model_str, "dataset_name": args.dataset_name, "scores_prompt_injection": scores_prompt_injection, "labels": labels}
outputs_dir = f"cached_outputs/{args.dataset_name}/{todaystring}/trial_{args.trial}_{model_str}/"
Path(outputs_dir).mkdir(parents=True, exist_ok=True)
np.savez(outputs_dir + f"{model_str}_{args.dataset_name}_outputs.npz", **outputs)

# save into npz the scores + fpr associated with this
# make a script which reads from dict and can combine multiple dict and then create ROC
# add support for predibase model
# figure out openai structure outputs