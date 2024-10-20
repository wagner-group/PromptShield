from datasets import load_dataset
from datasets import Dataset
from datasets.utils.info_utils import VerificationMode
import os
import pandas as pd

import torch
import numpy as np
import json
from .generics import GenericDataset

#########################################################################
#                            Benign datasets                            #
#########################################################################

# LMSYS dataset - conversational data only
class LMSYS(GenericDataset):
  # Set up LMSYS dataset
  def __init__(self, dataset_split="train", toxicity_threshold=0.1, language="English"):
    loaded_dataset = load_dataset("lmsys/lmsys-chat-1m")[dataset_split]
    
    # Filter dataset according to desired language
    lang_idx = (np.array(loaded_dataset["language"]) == language)
    loaded_dataset = loaded_dataset.select(np.arange(len(loaded_dataset))[lang_idx])
    
    # #### TEST
    # loaded_datset = loaded_dataset.select(np.arange(10000))

    # Filter data based on the OpenAI moderation scores to remove jailbreak prompts
    def check_moderation(example):
      moderation = example["openai_moderation"]

      # Consider the scores associated with the first user prompt (i.e., index "0" of each moderation item)
      scores = np.array(list(moderation[0]["category_scores"].values()))
      no_violation = np.all(scores < toxicity_threshold)

      return no_violation

    loaded_dataset = loaded_dataset.filter(check_moderation) if toxicity_threshold < 1.0 else loaded_dataset

    # Assign benign labels to this task
    labels = torch.zeros(len(loaded_dataset))

    super().__init__(loaded_dataset, labels, "LMSYS")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    user_prompts = [conv for conv in data["conversation"] if conv["role"] == "user"]
    first_user_prompt = user_prompts[0]["content"]

    return first_user_prompt

  # Create a dict object from the provided data point
  def get_dict(self, data):

    # This dataset represents chat data, and thus has no data field
    return {"instruction":  self.extract_prompt(data), "input": "", "output": "", "source": self.name, "type": "Benign - chat_data", "flag": 0}

  # Returns the id associated with the provided data point
  def get_id(self, data):
    return data["conversation_id"]
  
# Natural-Instructions dataset
class NaturalInstructions(GenericDataset):
  # Set up Natural-Instructions dataset
  def __init__(self, dataset_split="test"):
    # This dataset might have loading issues; it can be loaded with the
    # parameter VerificationMode.NO_CHECKS (albeit this results in a subset of the overall data)
    loaded_dataset = load_dataset("jayelm/natural-instructions")[dataset_split]
    # loaded_dataset = load_dataset("Muennighoff/natural-instructions", verification_mode=VerificationMode.NO_CHECKS)

    # #### TEST
    # loaded_dataset = loaded_dataset.select(np.arange(10000))

    # Find and filter out duplicates as described in the dataset README
    first_loc_dups = []
    task_id_map = {}
    for data_index, task in enumerate(loaded_dataset):
      task_id = task["id"]

      # Check if this task_id has already been encountered
      if task_id_map.get(task_id) != None: continue

      # This task_id has now been seen once, thus we note down the location of first appearance
      task_id_map[task_id] = 1
      first_loc_dups.append(data_index)

    loaded_dataset = loaded_dataset.select(np.array(first_loc_dups))

    # Assign benign labels to this task
    labels = torch.zeros(len(loaded_dataset))

    super().__init__(loaded_dataset, labels, "natural-instructions")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    user_prompt = data["definition"] + "\n" + data["inputs"]
    return user_prompt

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return {"instruction":  data["definition"], "input": data["inputs"], "output": data["targets"], "source": self.name,"type": "Benign", "flag": 0}

  # Returns the id associated with the provided data point
  def get_id(self, data):
    return data["id"]

# SPP_30K_reasoning_tasks dataset
class SPP(GenericDataset):
  # Set up SPP dataset
  def __init__(self, dataset_split="train"):
    loaded_dataset = load_dataset("Nan-Do/SPP_30K_reasoning_tasks")[dataset_split]

    # Assign benign labels to this task
    labels = torch.zeros(len(loaded_dataset))

    super().__init__(loaded_dataset, labels, "SPP_30K")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    user_prompt = data["instruction"] + "\n" + data["input"]
    return user_prompt

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return {"instruction":  data["instruction"], "input": data["input"], "output": data["output"], "source": self.name, "type": "Benign", "flag": 0}

#########################################################################
#                         Adversarial datasets                          #
#########################################################################

# Loads the dataset generated in JSON format from the OpenPromptInjection USENIX paper: https://github.com/liu00222/Open-Prompt-Injection.
#### NOTE: for future we can combine the two JSON for benign and adversarial into one -> this allows us to require only one filepath for args
class OpenPromptInjection(GenericDataset):
  # Set up OpenPromptInjection dataset by reading from a JSON file
  def __init__(self):
    # Load data from a JSON file
    combined_data = []

    filepath = os.path.dirname(__file__) + '/../../data/data_sources/usenix_attacks.json'
    with open(filepath, encoding='utf-8') as data_file:
      combined_data = list(json.loads(data_file.read()))

    filepath = os.path.dirname(__file__) + '/../../data/data_sources/usenix_benign.json'
    with open(filepath, encoding='utf-8') as data_file:
      combined_data.extend(json.loads(data_file.read()))

    loaded_dataset = Dataset.from_list(combined_data)

    # Create classification labels associated with the prompt injection detection task
    labels = torch.Tensor(loaded_dataset["flag"])

    super().__init__(loaded_dataset, labels, "OpenPromptInjection")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    user_prompt = data["instruction"] + "\n" + data["input"]
    return user_prompt

  # Create a dict object from the provided data point
  def get_dict(self, data):
    data_type = "Benign - OpenPromptInjection" if data["flag"] == 0 else f"Injection - OpenPromptInjection_{data['type']}"
    return {"instruction":  data["instruction"], "input": data["input"], "output": "", "source": self.name, "type": data_type, "flag": data["flag"]}