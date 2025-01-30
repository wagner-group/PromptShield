from datasets import load_dataset
from datasets import Dataset
from datasets.utils.info_utils import VerificationMode
import os
import pandas as pd

import torch
import numpy as np
import json
from .generics import GenericDataset, loadDatasetJSON

#########################################################################
#                          Benchmark datasets                          #
#########################################################################

# Loads the JSON-formatted version of a benchmark dataset
class BenchmarkDataset(GenericDataset):
  # Set up benchmark dataset by reading from a JSON file
  def __init__(self, filepath, dataset_partition="train"):
    self.dataset_partition = dataset_partition
    loaded_dataset = loadDatasetJSON(filepath)

    # Create training labels associated with the prompt injection detection task
    labels = torch.Tensor(loaded_dataset["flag"])

    super().__init__(loaded_dataset, labels, "benchmark_dataset")

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    # For training, leave the "\n" character. For evaluation, we remove it
    if self.dataset_partition == "train":
      user_prompt = data["instruction"] + "\n" + data["input"]
    elif self.dataset_partition == "test":
      user_prompt = (data["instruction"] + "\n" + data["input"]) if data["input"] != "" else data["instruction"]

    return user_prompt

  # Create a dict object from the provided data point
  def get_dict(self, data):
    return data

  def get_label(self, data):
    return data['flag']
  
  def get_labels(self):
    return self.labels