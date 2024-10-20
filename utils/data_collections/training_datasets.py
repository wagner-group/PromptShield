from datasets import load_dataset
from datasets import Dataset
from datasets.utils.info_utils import VerificationMode
import os
import pandas as pd

import torch
import numpy as np
from hashlib import sha256
import json
from .generics import loadDatasetJSON, GenericDataset
from ..injection_methods.StruQ import create_random_prompt_injected_llm_input

#########################################################################
#                            Benign datasets                            #
#########################################################################

# Ultrachat dataset - conversational data only
class Ultrachat(GenericDataset):
  # Set up Ultrachat dataset
  def __init__(self, dataset_split="train_sft"):
    loaded_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")[dataset_split]

    # Assign benign labels to this task
    labels = torch.zeros(len(loaded_dataset))

    super().__init__(loaded_dataset, labels, "Ultrachat_200k")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    return data["prompt"]

  # Create a dict object from the provided data point
  def get_dict(self, data):

    # This dataset represents chat data, and thus has no data field
    return {"instruction":  self.extract_prompt(data), "input": "", "output": "", "source": self.name, "type": "Benign - chat_data", "flag": 0}

  # Returns the id associated with the provided data point
  def get_id(self, data):
    return data["prompt_id"]

# Alpaca dataset
class Alpaca(GenericDataset):
  # Set up Alpaca dataset
  def __init__(self, dataset_split="train", data_type="all"):
    loaded_dataset = load_dataset("tatsu-lab/alpaca")[dataset_split]
    
    # Choose between closed domain and open domain partition
    if data_type == "closed_domain":
      loaded_dataset = loaded_dataset.filter(lambda example: example['input'] != "")
    elif data_type == "open_domain":
      loaded_dataset = loaded_dataset.filter(lambda example: example['input'] == "")
    self.data_type = data_type

    # Assign benign labels to this task
    labels = torch.zeros(len(loaded_dataset))

    super().__init__(loaded_dataset, labels, "Alpaca")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    user_prompt = data["instruction"] + "\n" + data["input"]
    return user_prompt

  # Create a dict object from the provided data point
  def get_dict(self, data):
    # Only store the response if the data is closed domain
    output = data["output"] if data["input"] != "" else ""
    return {"instruction":  data["instruction"], "input": data["input"], "output": output, "source": self.name,"type": f"Benign - {self.data_type}", "flag": 0}

# DatabricksDolly dataset
class DatabricksDolly(GenericDataset):
  # Set up DatabricksDolly dataset
  def __init__(self, dataset_split="train", data_type="all"):
    loaded_dataset = load_dataset("databricks/databricks-dolly-15k")[dataset_split]

    # Choose between closed domain and open domain partition
    if data_type == "closed_domain":
      loaded_dataset = loaded_dataset.filter(lambda example: example["context"] != "")
    elif data_type == "open_domain":
      loaded_dataset = loaded_dataset.filter(lambda example: example["context"] == "")
    self.data_type = data_type

    # Assign benign labels to this task
    labels = torch.zeros(len(loaded_dataset))

    super().__init__(loaded_dataset, labels, "databricks-dolly-15k")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    user_prompt = data["instruction"] + "\n" + data["context"]
    return user_prompt

  # Create a dict object from the provided data point
  def get_dict(self, data):
    # Only store the response if the data is closed domain
    output = data["response"] if data["context"] != "" else ""
    return {"instruction":  data["instruction"], "input": data["context"], "output": output, "source": self.name,"type": f"Benign - {self.data_type}", "flag": 0}

# IFEval dataset - open domain data only
class IFEval(GenericDataset):
  # Set up IFEval dataset
  def __init__(self, dataset_split="train"):
    loaded_dataset = load_dataset("google/IFEval")[dataset_split]

    # Assign benign labels to this task
    labels = torch.zeros(len(loaded_dataset))

    super().__init__(loaded_dataset, labels, "IFEval")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    return data["prompt"]

  # Create a dict object from the provided data point
  def get_dict(self, data):

    # This dataset consists of open domain data and has no user input field
    return {"instruction":  data["prompt"], "input": "", "output": "", "source": self.name,"type": "Benign - open_domain", "flag": 0}

#########################################################################
#                         Adversarial datasets                          #
#########################################################################

# Loads the JSON-formatted version of the PurpleLlama dataset 
#### NOTE: for future we can put filepath in args
class PurpleLlama(GenericDataset):
  # Set up PurpleLlama dataset by reading from a JSON file
  def __init__(self):
    filepath = os.path.dirname(__file__) + '/../../data/data_sources/purplellama_prompt_injection.json'
    loaded_dataset = loadDatasetJSON(filepath)
    
    # Filter out direct attacks
    loaded_dataset = loaded_dataset.filter(lambda example: example['injection_type'] == "indirect")

    # Create training labels associated with the prompt injection detection task
    labels = torch.ones(len(loaded_dataset))

    super().__init__(loaded_dataset, labels, "PurpleLlama")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    user_prompt = data["test_case_prompt"] + "\n" + data["user_input"]
    return user_prompt

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return {"instruction":  data['test_case_prompt'], "input": data['user_input'], "output": "", "source": self.name, "type": "Injection - Indirect", "flag": 1}

# HackAPrompt dataset
class HackAPrompt(GenericDataset):
  # Set up HackAPrompt dataset
  def __init__(self, dataset_split="train"):
    loaded_dataset = load_dataset("hackaprompt/hackaprompt-dataset")[dataset_split]
    
    # Filter for successful prompt injections only
    loaded_dataset = loaded_dataset.filter(lambda example: example["correct"] == True)
    
    # Find and filter out duplicates as described in the dataset README
    first_loc_dups = []
    hashmap = {}
    for data_index, data in enumerate(loaded_dataset):
      prompt_hash = sha256((data["prompt"]).encode('utf-8')).hexdigest()

      # Check if this prompt has already been encountered
      if hashmap.get(prompt_hash) != None: continue

      # This task_id has now been seen once, thus we note down the location of first appearance
      hashmap[prompt_hash] = 1
      first_loc_dups.append(data_index)

    loaded_dataset = loaded_dataset.select(np.array(first_loc_dups))

    # Create classification labels associated with the prompt injection detection task
    labels = torch.ones(len(loaded_dataset))

    super().__init__(loaded_dataset, labels, "HackAPrompt")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    return data["prompt"]

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return {"instruction":  data["prompt"], "input": data["user_input"], "output": "", "source": self.name, "type": "Injection - Natural", "flag": 1}

# A dataset comprised of StruQ attacks - requires a benign seed dataset
class StruQAttacks(GenericDataset):
  # Set up StruQ dataset
  def __init__(self, seed_dataset_collection, dataset_partition="subset", dataset_status="test"):
    seed_dataset_name = seed_dataset_collection.name
    loaded_dataset, _ = seed_dataset_collection.get_subset_dataset() if dataset_partition == "subset" else seed_dataset_collection.get_dataset()

    struq_data = create_random_prompt_injected_llm_input(seed_dataset_collection, loaded_dataset, seed_dataset_name, dataset_status)
    loaded_dataset = Dataset.from_list(struq_data)

    # Create training labels associated with the prompt injection detection task
    labels = torch.ones(len(loaded_dataset))

    super().__init__(loaded_dataset, labels, f"StruQ - {seed_dataset_name}")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    prompt = data["instruction"] + "\n" + data["input"]
    return prompt

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return {"instruction":  data['instruction'], "input": data['input'], "output": "", "source": self.name, "type": f"Injection - StruQ_{data['type']}", "flag": 1}
  
# class ScienceQA(TrainingDataset):
#   # Set up ScienceQA dataset; 
#   def __init__(self, dataset_split, subset_amount=1000, random_sample=True, offset=0):
#     loaded_dataset = load_dataset("tasksource/ScienceQA_text_only")
#     super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)

#     # Create classification labels associated with the prompt injection detection task
#     self.labels = torch.zeros(len(self.dataset))

#   # Create a dict object from the provided data point
#   def get_dict(self, data):
#     return {"instruction":  data['question'], "input": str(data["choices"]), "source": "ScienceQA_text_only","type": "Benign", "flag": 0}
  