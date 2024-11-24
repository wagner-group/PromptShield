from tqdm.auto import tqdm
import argparse
from datetime import date
from pathlib import Path
import json
import csv
from hashlib import sha256
from utils.data_collections.benchmark_datasets import  BenchmarkDataset

parser = argparse.ArgumentParser(description='Plot ROC curve')

# Arguments
parser.add_argument('--output-date')
parser.add_argument('--model')
parser.add_argument('--trial', default=1)
parser.add_argument('--file-date', default="")


args = parser.parse_args()


# Initiailize variables
model_name_list = []
dataset_name_list = []
scores_prompt_injection = []
labels = []

todaystring = date.today().strftime("%Y-%m-%d")

# Set up benchmark dataset
if args.file_date == "":
    benchmark_file = f"data/evaluation_data/{todaystring}/{todaystring}_evaluation_benchmark.json"
else:
    benchmark_file = f"data/evaluation_data/{args.file_date}/{args.file_date}_evaluation_benchmark.json"

try:
    data_collection = BenchmarkDataset(benchmark_file)
except:
    print("Benchmark file not found, enter correct date (--file-date=2024-10-20)")
    raise


# Import outputs
outputs = []
output_file = f"dataset_evals/evaluation_benchmark/{args.output_date}/trial_2_model_v{args.model}_offset_0/results.json"

with open(output_file, encoding='utf-8') as data_file:
    outputs = json.loads(data_file.read())
    
benchmark_dataset, _ = data_collection.get_dataset()

fps = []
fns = []

for index, data in enumerate(benchmark_dataset):        
    user_prompt = data_collection.extract_prompt(data)
    prompt_id = data_collection.get_id(data)
    label = data_collection.get_label(data)
    predicted = outputs[prompt_id]["response"]
    data_dict = data_collection.get_dict(data)
    if int(label) != int(predicted):
        dict = {}
        dict["prompt_id"] = prompt_id
        dict["prompt"] = user_prompt
        dict["label"] = label
        dict["prediction"] = predicted
        dict["source"] = data_dict["source"]
        dict["type"] = data_dict["type"]

        if label == 1:
            dict["error"] = "FN"
            fns.append(dict)
                
        elif label == 0:
            dict["error"] = "FP"
            fps.append(dict)



print(f'FNs length: {len(fns)}\n\n\n')

print(f'FPs length: {len(fps)}\n\n\n')

save_dir = f"fps_fns/modelv_{args.model}/"
Path(save_dir).mkdir(parents=True, exist_ok=True)


header = ["prompt_id", "prompt", "Dataset", "Type", "label", "prediction", "source", "type", "error"]
with open(f'{save_dir}/fps.csv', 'w') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(fps)


with open(f'{save_dir}/fns.csv', 'w') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(fns)

all = fns + fps
with open(f'{save_dir}/all.jsonl', 'w') as file:
    for entry in all:
        json.dump(entry, file)
        file.write('\n')