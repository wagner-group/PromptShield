from datasets import Dataset

import torch
import numpy as np
from hashlib import sha256

def random_indices(dataset_len, selected_range, seed=12345):
  rng = np.random.default_rng(seed)
  return rng.choice(dataset_len, selected_range, replace=False)  

# Helper function which converts a JSON dataset to a HuggingFace dataset
def loadDatasetJSON(filepath):
  # Load data from a JSON file
  datasetJSON = []
  with open(filepath, encoding='utf-8') as data_file:
    datasetJSON = list(json.loads(data_file.read()))
  
  # Convert to HuggingFace dataset
  loaded_dataset = Dataset.from_list(dataset)

  return loaded_dataset

# Abstract class which represents a classification task for a HuggingFace dataset
# Inspired by: https://pytorch.org/vision/main/_modules/torchvision/datasets/vision.html#VisionDataset
class GenericDataset():
  def __init__(self, dataset, labels, name):
    self.dataset = dataset.add_column("labels", labels.numpy().astype("int"))
    self.labels = labels
    self.name = name

    # Default value of the subset is the overall dataset
    self.subset_dataset = self.dataset
    self.subset_labels = labels

  # Return the underlying dataset and labels
  def get_dataset(self):
    return self.dataset, self.labels
  
  # Return the length of the underlying dataset
  def get_dataset_len(self):
    return len(self.dataset)

  # Creates a subset of the underlying dataset; set subset_amount to -1 if the entire dataset is desired
  def create_subset_dataset(self, subset_amount=-1, offset=0, random_sample=True, random_seed=12345):
    total_dataset_len = self.get_dataset_len()

    # Select the entire dataset if subset_amount is -1
    selected_range = subset_amount 
    if subset_amount == -1:
      selected_range = total_dataset_len
      offset = 0

    # Slice the dataset according to the offset
    offset_dataset = self.dataset.select(np.arange(total_dataset_len)[offset:])

    # Check if a random sample is desired
    use_random = (random_sample) and (subset_amount != -1)
    indices = random_indices(len(offset_dataset), selected_range, random_seed) if use_random else range(selected_range)

    # Update the underlying subset dataset
    self.subset_dataset = offset_dataset.select(indices)
    self.subset_labels = self.labels[indices]

    return self.subset_dataset, self.subset_labels

  # Return the underlying dataset subset and labels
  def get_subset_dataset(self):
    return self.subset_dataset, self.subset_labels
  
  # Return the length of the underlying subset dataset
  def get_subset_dataset_len(self):
    return len(self.subset_dataset)

  # This method is dataset dependent; it should convert individual datapoints from self.dataset into a single prompt
  def extract_prompt(self, data):
    raise NotImplementedError

  # This method is dataset dependent; it should convert individual datapoints from self.dataset into a dict
  def get_dict(self, data):
    raise NotImplementedError

  # Returns the id associated with the provided data point
  def get_id(self, data):
    user_prompt = self.extract_prompt(data)

    return sha256((user_prompt).encode('utf-8')).hexdigest()

  # Given a tokenizer, dataset, and labels, convert into a usable representation for torch.utils.data.DataLoader
  # NOTE: self.extract_prompt is applied to each datapoint to ensure resulting elements are of equal size
  def convert2torch(self, tokenizer, dataset, labels, text_transform = None):
    # Combine self.extract_prompt with text_transform
    def combined_transform(data):
      extracted_prompt = self.extract_prompt(data)
      if text_transform is not None:
        extracted_prompt = text_transform(extracted_prompt)

      return {"extracted_prompt": extracted_prompt}

    # Apply combined_transform to each element of self.dataset, then use the tokenizer    
    extracted_dataset = dataset.map(combined_transform)
    encoded_texts = tokenizer(extracted_dataset["extracted_prompt"], return_tensors="pt", padding=True, truncation=True)

    # Return PyTorch dataset
    encoded_dataset = torch.utils.data.TensorDataset(encoded_texts['input_ids'], encoded_texts['attention_mask'], labels)

    return encoded_dataset