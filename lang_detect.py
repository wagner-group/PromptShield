import json
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm
import argparse


# Set seed for consistent language detection results
DetectorFactory.seed = 0

parser = argparse.ArgumentParser(description='Add Language to dataset')

parser.add_argument('--data-file', default="data/evaluatin_data/2024-11-28/2024-11-28_evaluation_benchmark")

args = parser.parse_args()

args = parser.parse_args()

# Input and output file paths
input_filepath = f'{args.data_file}.json'
filtered_output_filepath = f'{args.data_file}_en.json'

# Load the dataset
with open(input_filepath, encoding="utf-8") as data_file:
    datasetJSON = json.load(data_file)

# Process dataset and filter English-only entries
english_only_data = []

print(len(datasetJSON))

for item in tqdm(datasetJSON, desc="Processing dataset", unit="entry"):
    try:
        detected_lang = detect(item["instruction"]+"\n"+item["input"])
        item["lang"] = detected_lang
        if detected_lang == "en":
            english_only_data.append(item)
    except LangDetectException:
        item["lang"] = "unknown"

# Save the full modified dataset with language detection
with open(input_filepath, "w", encoding="utf-8") as output_file:
    json.dump(datasetJSON, output_file, ensure_ascii=False, indent=4)

# Save the English-only filtered dataset
with open(filtered_output_filepath, "w", encoding="utf-8") as filtered_output_file:
    json.dump(english_only_data, filtered_output_file, ensure_ascii=False, indent=4)

print(f"Processed dataset saved to {input_filepath}")
print(f"English-only dataset saved to {filtered_output_filepath}")