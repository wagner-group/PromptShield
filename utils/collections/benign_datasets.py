from datasets import load_dataset
from datasets import Dataset
from datasets.utils.info_utils import VerificationMode

from hashlib import sha256
import torch
import numpy as np

from .generics import ClassificationDataset

# Interfaces with the LMSYS dataset from Hugging Face
class LMSYS(ClassificationDataset):
  # Set up LMSYS dataset; set subset_amount to -1 if the entire dataset is desired
  def __init__(self, dataset_split, subset_amount, random_sample=True, offset=0, toxicity_threshold=0.1, language="English"):
    loaded_dataset = load_dataset("lmsys/lmsys-chat-1m")
    super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)
    
    # Filter dataset according to desired language
    lang_idx = (np.array(self.dataset["language"]) == language)
    self.dataset = self.dataset.select(np.arange(len(self.dataset))[lang_idx])
    
    # Filter data based on the OpenAI moderation scores to remove jailbreak prompts
    content_violations = []
    for moderation in self.dataset["openai_moderation"]:
      # Consider the scores associated with the first user prompt (i.e., index "0" of each moderation item)
      scores = np.array(list(moderation[0]["category_scores"].values()))
      violation = np.any(scores > toxicity_threshold)

      content_violations.append(violation)

    content_violations = np.array(content_violations) if toxicity_threshold < 1.0 else np.zeros(len(self.dataset)).astype(bool)
    number_mask = np.arange(len(self.dataset))
    self.dataset = self.dataset.select(number_mask[np.logical_not(content_violations)])

    # Create classification labels associated with the prompt injection detection task
    self.labels = torch.zeros(len(self.dataset))

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    user_prompts = [conv for conv in data["conversation"] if conv["role"] == "user"]
    first_user_prompt = user_prompts[0]["content"]

    return first_user_prompt

  # Returns the id associated with the provided data point
  def get_id(self, data):
    return data["conversation_id"]
  
# Interfaces with the Natural-Instructions dataset from Hugging Face
class NaturalInstructions(ClassificationDataset):
  # Set up Natural-Instructions dataset; set subset_amount to -1 if the entire dataset is desired
  def __init__(self, dataset_split, subset_amount, random_sample=True, offset=0):
    # This dataset might have loading issues; it can be loaded with the
    # parameter VerificationMode.NO_CHECKS (albeit this results in a subset of the overall data)
    loaded_dataset = load_dataset("jayelm/natural-instructions")
    # loaded_dataset = load_dataset("Muennighoff/natural-instructions", verification_mode=VerificationMode.NO_CHECKS)
    super().__init__(loaded_dataset, dataset_split, subset_amount, random_sample, offset)

    # Find and filter out duplicates as described in the dataset README
    first_loc_dups = []
    task_id_map = {}
    for data_index, task in enumerate(self.dataset):
      task_id = task["id"]

      # Check if this task_id has already been encountered
      if task_id_map.get(task_id) != None: continue

      # This task_id has now been seen once, thus we note down the location of first appearance
      task_id_map[task_id] = 1
      first_loc_dups.append(data_index)

    self.dataset = self.dataset.select(np.array(first_loc_dups))

    # Create classification labels associated with the prompt injection detection task
    self.labels = torch.zeros(len(self.dataset))

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    user_prompt = data["definition"] + "\n" + data["inputs"]

    return user_prompt

  # Returns the id associated with the provided data point
  def get_id(self, data):
    return data["id"]