from datasets import load_dataset
from datasets import Dataset
from datasets.utils.info_utils import VerificationMode
import os
import pandas as pd

from hashlib import sha256
import torch
import numpy as np
import json
from .generics import TrainingDataset
from .inject_alpaca import create_random_prompt_injected_llm_input


class PurpleLlama(TrainingDataset):
  #load the json file
  def __init__(self):
    dataset = []
    filepath = os.path.dirname(__file__) + '/../../data/data_sources/purplellama_prompt_injection.json'
    with open(filepath, encoding='utf-8') as data_file:
      dataset = json.loads(data_file.read())
    
    # Filter out direct attacks
    filtered_data = [d for d in dataset if d['injection_type'] == "indirect"]
    self.dataset = filtered_data


    # Create training labels associated with the prompt injection detection task
    self.labels = torch.ones(len(self.dataset))

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return {"instruction":  data['test_case_prompt'], "input": data['user_input'], "source": "Purplellama","type": "Indirect", "flag": 1}
  

class Alpaca(TrainingDataset):
  # Set up Alpaca dataset; set subset_amount to -1 if the entire dataset is desired
  def __init__(self, dataset_split, subset_amount, random_sample=True, offset=0):
    loaded_dataset = load_dataset("tatsu-lab/alpaca")
    # Filter out null values
    loaded_dataset = loaded_dataset.filter(lambda example: example['input'] != "")

    super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)

    # Create training labels associated with the prompt injection detection task
    self.labels = torch.zeros(len(self.dataset))

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return {"instruction":  data['instruction'], "input": data['input'], "source": "Alpaca","type": "Benign", "flag": 0}
  
# Interfaces with the LMSYS dataset from Hugging Face
class LMSYS(TrainingDataset):
  # Set up LMSYS dataset; set subset_amount to -1 if the entire dataset is desired
  def __init__(self, dataset_split, subset_amount=15000, random_sample=False, offset=0):
    loaded_dataset = load_dataset("lmsys/lmsys-chat-1m")
    super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)
    self.toxicity_threshold=0.01
    # Filter dataset according to desired language
    lang_idx = (np.array(self.dataset["language"]) == "English")
    self.dataset = self.dataset.select(np.arange(len(self.dataset))[lang_idx])
    
    # Filter data based on the OpenAI moderation scores to remove jailbreak prompts
    content_violations = []
    for moderation in self.dataset["openai_moderation"]:
      # Consider the scores associated with the first user prompt (i.e., index "0" of each moderation item)
      scores = np.array(list(moderation[0]["category_scores"].values()))
      violation = np.any(scores > self.toxicity_threshold)

      content_violations.append(violation)

    content_violations = np.array(content_violations) if self.toxicity_threshold < 1.0 else np.zeros(len(self.dataset)).astype(bool)
    number_mask = np.arange(len(self.dataset))
    self.dataset = self.dataset.select(number_mask[np.logical_not(content_violations)])

    # Create classification labels associated with the prompt injection detection task
    self.labels = torch.zeros(len(self.dataset))

  # Create a dict object from the provided data point
  def get_dict(self, data):
    user_prompts = [conv for conv in data["conversation"] if conv["role"] == "user"]
    first_user_prompt = user_prompts[0]["content"]
    return {"instruction":  "", "input": first_user_prompt, "source": f'LMSYS-{str(self.toxicity_threshold)}',"type": "Benign", "flag": 0}
 

class StruQAttacks(TrainingDataset):

  def __init__(self, sample_size=1000, seed_dataset_name="Alpaca"):

    #TODO: add other options for the struQ data generation?
    #create the seed dataset, default is Alpaca
    if seed_dataset_name == "Alpaca":
      seed_data_collection = Alpaca("train", sample_size, random_sample=True)
    elif seed_dataset_name == "SPP":
      seed_data_collection = SPP("train", sample_size, random_sample=True)
    

    struq_data = create_random_prompt_injected_llm_input(seed_data_collection.dataset)

    # struq_df = pd.DataFrame.from_records(struq_data)

    self.dataset =  Dataset.from_list(struq_data)
    self.seed_dataset_name = seed_dataset_name

    # Create training labels associated with the prompt injection detection task
    self.labels = torch.ones(len(self.dataset))

    # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    prompt = data["instruction"] + "\n" + data["input"]

    return prompt

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return {"instruction":  data['instruction'], "input": data['input'], "source": f'StruQ - {self.seed_dataset_name}',"type": data['type'], "flag": 1}
  
  # Returns the id associated with the provided data point
  def get_id(self, data):
    user_prompt = self.extract_prompt(data)

    return sha256((user_prompt).encode('utf-8')).hexdigest()
  

# Interfaces with the SPP_30K_reasoning_tasks dataset from Hugging Face
class SPP(TrainingDataset):
  # Set up SPP dataset; 
  def __init__(self, dataset_split, subset_amount=1000, random_sample=True, offset=0):
    loaded_dataset = load_dataset("Nan-Do/SPP_30K_reasoning_tasks")
    super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)

    # Create classification labels associated with the prompt injection detection task
    self.labels = torch.zeros(len(self.dataset))

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return {"instruction":  data['instruction'], "input": data["input"], "source": "SPP_30K_reasoning_tasks","type": "Benign", "flag": 0}
  
class ScienceQA(TrainingDataset):
  # Set up ScienceQA dataset; 
  def __init__(self, dataset_split, subset_amount=1000, random_sample=True, offset=0):
    loaded_dataset = load_dataset("tasksource/ScienceQA_text_only")
    super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)

    # Create classification labels associated with the prompt injection detection task
    self.labels = torch.zeros(len(self.dataset))

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return {"instruction":  data['question'], "input": str(data["choices"]), "source": "ScienceQA_text_only","type": "Benign", "flag": 0}
  
  
class DatabricksDolly(TrainingDataset):
  # Set up ScienceQA dataset; 
  def __init__(self, dataset_split, subset_amount=1000, random_sample=True, offset=0):
    loaded_dataset = load_dataset("databricks/databricks-dolly-15k")
    super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)

    # Create classification labels associated with the prompt injection detection task
    self.labels = torch.zeros(len(self.dataset))

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return {"instruction":  data['instruction'], "input": data["context"], "source": "databricks-dolly-15k","type": "Benign", "flag": 0}


class IFEval(TrainingDataset):
  # Set up ScienceQA dataset; 
  def __init__(self, dataset_split, subset_amount=1000, random_sample=True, offset=0):
    loaded_dataset = load_dataset("google/IFEval")
    super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)

    # Create classification labels associated with the prompt injection detection task
    self.labels = torch.zeros(len(self.dataset))

  # Create a dict object from the provided data point
  def get_dict(self, data):

    #this dataset has no user input field
    return {"instruction":  data['prompt'], "input": "", "source": "IFEval","type": "Benign", "flag": 0}

class Ultrachat(TrainingDataset):
  # Set up ScienceQA dataset; 
  def __init__(self, dataset_split= "train_sft", subset_amount=1000, random_sample=True, offset=0):
    loaded_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)

    # Create classification labels associated with the prompt injection detection task
    self.labels = torch.zeros(len(self.dataset))

  # Create a dict object from the provided data point
  def get_dict(self, data):

    #this dataset has no user input field
    return {"instruction":  data['prompt'], "input": "", "source": "Ultrachat_200k","type": "Benign", "flag": 0}