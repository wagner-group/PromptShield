from datasets import load_dataset
from datasets import Dataset
from datasets.utils.info_utils import VerificationMode
import numpy as np
import torch

def random_indices(dataset_len, selected_range):
  rng = np.random.default_rng(12345)
  return rng.choice(dataset_len, selected_range, replace=False)  

# Abstract class which represents a classification task for a HuggingFace dataset
# Inspired by: https://pytorch.org/vision/main/_modules/torchvision/datasets/vision.html#VisionDataset
class ClassificationDataset():
  def __init__(self, dataset, dataset_split, subset_amount, random_sample=True, offset=0):
    total_dataset_len = len(dataset[dataset_split])

    # Select the entire dataset if subset_amount is -1
    selected_range = subset_amount 
    if subset_amount == -1:
      selected_range = total_dataset_len
      offset = 0

    # Slice the dataset according to the offset
    offset_dataset = dataset[dataset_split].select(np.arange(total_dataset_len)[offset:])

    # Check if a random sample is desired
    use_random = (random_sample) and (subset_amount != -1)
    indices = random_indices(len(offset_dataset), selected_range) if use_random else range(selected_range)

    self.dataset = offset_dataset.select(indices)

    # Default value for labels
    self.labels = torch.zeros(len(self.dataset))

  # Return the underlying dataset
  def get_dataset(self):
    return self.dataset
  
  # Return classification labels
  def get_labels(self):
    return self.labels
  
  # This method is implementation dependent; it should take in individual elements 
  # from self.dataset and return a single processed value
  def extract_prompt(self, data):
    raise NotImplementedError

  # Given a tokenizer, convert underlying dataset into a usable representation for torch.utils.data.DataLoader
  # NOTE: self.extract_prompt is applied to each datapoint to ensure resulting elements are of equal size
  def convert2torch(self, tokenizer, text_transform = None):
    # Combine self.extract_prompt with text_transform
    def combined_transform(data):
      extracted_prompt = self.extract_prompt(data)
      if text_transform is not None:
        extracted_prompt = text_transform(extracted_prompt)

      return {"extracted_prompt": extracted_prompt}

    # Apply combined_transform to each element of self.dataset, then use the tokenizer    
    extracted_dataset = self.dataset.map(combined_transform)
    encoded_texts = tokenizer(extracted_dataset["extracted_prompt"], return_tensors="pt", padding=True, truncation=True)

    # Return PyTorch dataset
    labels = self.get_labels()
    encoded_dataset = torch.utils.data.TensorDataset(encoded_texts['input_ids'], encoded_texts['attention_mask'], labels)

    return encoded_dataset

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
    # This dataset seems to have loading issues; thus, it must be loaded with the
    # parameter VerificationMode.NO_CHECKS (albeit this results in a subset of the overall data)
    loaded_dataset = load_dataset("Muennighoff/natural-instructions", verification_mode=VerificationMode.NO_CHECKS)
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

class UltraChat:
  # Set up UltraChat dataset; set subset_amount to -1 if the entire dataset is desired
  def __init__(self, dataset_split, subset_amount):
    loaded_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    selected_range = subset_amount if subset_amount > -1 else len(loaded_dataset[dataset_split])
    self.dataset = loaded_dataset[dataset_split].select(range(selected_range))
  
  # Return the underlying dataset
  def get_dataset(self):
    return self.dataset

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    datapoint_id = data["prompt_id"]
    user_prompt = data["prompt"]

    return user_prompt, datapoint_id