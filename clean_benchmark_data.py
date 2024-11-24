import json
import argparse
from datetime import date
from utils.data_collections.benchmark_datasets import BenchmarkDataset


parser = argparse.ArgumentParser(description='clean benchmark data, remove jailbreaks from LMSYS')

parser.add_argument('--file-date', default="")

args = parser.parse_args()

benchmark_data = []

#manually detected jailbreak datapoints
#modify this list if benchmark file was updated
jailbreak_ids = ["b54564522405513573d83ae5fe0e22761b6b90432c889da148c494f91c869089", 
                 "4d6d0e9472078aa604c30d4cf08502a48f8feee2248da5c00c43384fa1bab5b2", 
                 "6db3b0e829b67f4903dbd07b1bf4ace8756acd374583394365c5e6448c092a36", 
                 "82ce9f7c4451264b7908a6507aeefd377eb199daa5362e0e8e3e0afae9208676",
                 "9ff251e751800f45f11e7a5dc3d66c059bd2fbc46362f3aff8991f6867858d39",
                 "aecd8233c8298fd38e310f132f0a719e5cac7b0fdb788719eb0a8c5a1a3c063c",
                 "f7f6d4303c1b0b1022e30db37f693e026a4c569a540eb3e770637c20e79f196c", 
                 ]

todaystring = date.today().strftime("%Y-%m-%d")
#load benchmark from file 
# Set up dataset
if args.file_date == "":
    benchmark_file = f"data/evaluation_data/{todaystring}/{todaystring}_evaluation_benchmark.json"
else:
    benchmark_file = f"data/evaluation_data/{args.file_date}/{args.file_date}_evaluation_benchmark.json"


try:
    data_collection = BenchmarkDataset(benchmark_file)
except:
    print("Benchmark file not found, enter correct date (--file-date=2024-10-20)")
    raise

with open(benchmark_file, encoding='utf-8') as data_file:
    benchmark_data = list(json.loads(data_file.read()))

clean_data = []

for index, item in enumerate(benchmark_data):
    id = data_collection.get_id(item)
    if id not in jailbreak_ids:
        clean_data.append(item)


# Overwrite the benchmark file with clean dataset
out_file = open(benchmark_file, "w")
json.dump(clean_data, out_file, indent = 4, sort_keys = False)
out_file.close()