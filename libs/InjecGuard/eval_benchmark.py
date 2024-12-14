from transformers import AutoTokenizer
import os
import torch
import json
from InjecGuard import InjecGuard
from util import set_seed, get_logger
from params import parse_args
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from datetime import date

# Evaluation loop helper function
def evaluate_inputs(model, target_set):

    # Initialize softmax
    softmax = torch.nn.functional.softmax

    # Initiate loop
    preds = []
    scores_prompt_injection = []
    model.eval()
    print("\nStarting the evaluation process...")
    tqdm._instances.clear()

    progress_bar = tqdm(range(len(target_set)), position=0, leave=True, ascii=True)
    with torch.no_grad():
        for idx, sample in enumerate(target_set):
            class_logits = model.classify(sample['prompt'])
            pred = (class_logits.argmax(dim=1).cpu().numpy()).astype(int)
            target_label = sample['label']
            normalized_logits = softmax(class_logits, dim=1)
            score = normalized_logits[:, -1].cpu().numpy()

            del class_logits
            torch.cuda.empty_cache()
            # Add values to overall list
            preds.extend(pred)
            scores_prompt_injection.extend(score)
            progress_bar.update(1)

    return preds, scores_prompt_injection


def benchmark_eval(model, dataset_root="datasets"):
    set = []

    with open(os.path.join(dataset_root, "2024-11-28_evaluation_benchmark.json"), "r") as f:
        benchmark_dataset = json.load(f)

    benchmark_subset = benchmark_dataset
    labels = []

    for sample in benchmark_subset:       
            dict = {}   
            prompt = sample["instruction"]+'\n'+sample["input"] if sample["input"] != "" else sample["instruction"]
            dict['prompt'] = prompt
            dict['label'] = sample['flag']
            set.append(dict)
            labels.append(sample['flag'])

    preds, scores_prompt_injection = evaluate_inputs(model, set)

    dataset_str = "evaluation_benchmark"
    model_str = "InjecGuard"

    todaystring = date.today().strftime("%Y-%m-%d")
    save_dir = f"dataset_evals/{dataset_str}/{todaystring}/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Visualization
    fig = plt.figure(1)
    cm = confusion_matrix(labels, np.array(preds), labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "injection"])
    disp.plot().figure_.savefig(save_dir + 'confusion_matrix.png', bbox_inches='tight')

    # Save scores
    outputs = {"model_name": model_str, "dataset_name": dataset_str, "scores_prompt_injection": scores_prompt_injection, "labels": labels}
    outputs_dir = f"cached_outputs/"
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    np.savez(outputs_dir + f"benchmark_outputs.npz", **outputs)


if __name__ == "__main__":
    global logger
    args = parse_args()

    set_seed(args)
    logger = get_logger(os.path.join(args.logs, "log_{}.txt".format(args.name)))

    logger.info("Effective parameters:")

    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = InjecGuard('microsoft/deberta-v3-base', num_labels=2, device=device)
    model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)

    model.to(device)

    dataset_root = args.dataset_root
    benchmark_eval(model, dataset_root)
