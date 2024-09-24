from datasets import load_dataset
from datasets import Dataset
from datasets.utils.info_utils import VerificationMode

from hashlib import sha256
import torch
import numpy as np

from .generics import ClassificationDataset

# Interfaces with the SPML Chatbot Prompt Injection dataset from Hugging Face
class SPMLChatbotPromptInjection(ClassificationDataset):
  # Set up SPML Chatbot Prompt Injection dataset; set subset_amount to -1 if the entire dataset is desired
  def __init__(self, dataset_split, subset_amount, random_sample=True, offset=0):
    loaded_dataset = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")
    super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)

    # Create classification labels associated with the prompt injection detection task
    self.labels = torch.Tensor(self.dataset["Prompt injection"])

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    user_prompt = data["System Prompt"] + "\n" + data["User Prompt"]

    return user_prompt

  # Returns the id associated with the provided data point
  def get_id(self, data):
    user_prompt = self.extract_prompt(data)

    return sha256((user_prompt).encode('utf-8')).hexdigest()
  
# Interfaces with the LMSYS dataset from Hugging Face
class HackAPrompt(ClassificationDataset):
  # Set up HackAPrompt dataset; set subset_amount to -1 if the entire dataset is desired
  def __init__(self, dataset_split, subset_amount, random_sample=True, offset=0):
    loaded_dataset = load_dataset("hackaprompt/hackaprompt-dataset")

    #filter for successful prompt injections only
    loaded_dataset = loaded_dataset.filter(lambda example: example["correct"] == True)

    super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)
    
    # Find and filter out duplicates as described in the dataset README
    first_loc_dups = []
    hashmap = {}
    for data_index, data in enumerate(self.dataset):
      prompt_hash = sha256((data["prompt"]).encode('utf-8')).hexdigest()

      # Check if this prompt has already been encountered
      if hashmap.get(prompt_hash) != None: continue

      # This task_id has now been seen once, thus we note down the location of first appearance
      hashmap[prompt_hash] = 1
      first_loc_dups.append(data_index)

    self.dataset = self.dataset.select(np.array(first_loc_dups))
    # Create classification labels associated with the prompt injection detection task
    self.labels = torch.ones(len(self.dataset))


  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    user_prompt = data["prompt"]
    return user_prompt

  # Returns the id associated with the provided data point
  def get_id(self, data):
    user_prompt = self.extract_prompt(data)

    return sha256((user_prompt).encode('utf-8')).hexdigest()