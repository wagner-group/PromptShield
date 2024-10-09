from datasets import Dataset

import torch
import numpy as np

def random_indices(dataset_len, selected_range):
  rng = np.random.default_rng(12345)
  return rng.choice(dataset_len, selected_range, replace=False)  

# Abstract class which represents a classification task for a HuggingFace dataset
# Inspired by: https://pytorch.org/vision/main/_modules/torchvision/datasets/vision.html#VisionDataset
class ClassificationDataset():
  def __init__(self, dataset, dataset_split, subset_amount, random_sample=True, offset=0):
    if dataset_split != 0:
      total_dataset_len = len(dataset[dataset_split])
    else:
      total_dataset_len = len(dataset)

    # Select the entire dataset if subset_amount is -1
    selected_range = subset_amount 
    if subset_amount == -1:
      selected_range = total_dataset_len
      offset = 0

    # Slice the dataset according to the offset
    if dataset_split != 0:
      offset_dataset = dataset[dataset_split].select(np.arange(total_dataset_len)[offset:])
    else:
      offset_dataset = dataset

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
  

  # Abstract class which represents a training task for a HuggingFace dataset
# Inspired by: https://pytorch.org/vision/main/_modules/torchvision/datasets/vision.html#VisionDataset
class TrainingDataset():
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
  def get_dict(self, data):
    raise NotImplementedError