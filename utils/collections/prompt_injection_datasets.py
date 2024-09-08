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