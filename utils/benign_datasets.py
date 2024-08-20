from datasets import load_dataset
from datasets.utils.info_utils import VerificationMode
import numpy as np

def random_indices(dataset_len, selected_range):
  rng = np.random.default_rng(12345)
  return rng.choice(dataset_len, selected_range, replace=False)  

class LMSYS:
  # Set up LMSYS dataset; set subset_amount to -1 if the entire dataset is desired
  def __init__(self, dataset_split, subset_amount, random_sample=True, toxicity_threshold=0.1, offset=0, language="English"):
    loaded_dataset = load_dataset("lmsys/lmsys-chat-1m")
    total_dataset_len = len(loaded_dataset[dataset_split])

    # If subset_amount is -1, select the entire dataset
    if subset_amount == -1:
      selected_range = total_dataset_len
      offset = 0

    # Slice the dataset according to the offset
    selected_range = subset_amount 
    offset_dataset = loaded_dataset[dataset_split].select(np.arange(total_dataset_len)[offset:])

    # Check if a random sample is desired
    use_random = (random_sample) and (subset_amount != -1)
    indices = random_indices(len(offset_dataset), selected_range) if use_random else range(selected_range)

    self.dataset = offset_dataset.select(indices)

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
  
  # Return the underlying dataset
  def get_dataset(self):
    return self.dataset

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    datapoint_id = data["conversation_id"]
    user_prompts = [conv for conv in data["conversation"] if conv["role"] == "user"]
    first_user_prompt = user_prompts[0]["content"]

    return first_user_prompt, datapoint_id
  
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
  
class NaturalInstructions:
  # Set up Natural-Instructions dataset; set subset_amount to -1 if the entire dataset is desired
  def __init__(self, dataset_split, subset_amount, random_sample=True):
    # This dataset seems to have loading issues; thus, it must be loaded with the
    # parameter VerificationMode.NO_CHECKS (albeit this results in a subset of the overall data)
    loaded_dataset = load_dataset("Muennighoff/natural-instructions", verification_mode=VerificationMode.NO_CHECKS)
    dataset_len = len(loaded_dataset[dataset_split])
    selected_range = subset_amount if subset_amount > -1 else len(loaded_dataset[dataset_split])

    # Check if a random sample is desired
    use_random = (random_sample) and (subset_amount != -1)
    indices = random_indices(dataset_len, selected_range) if use_random else range(selected_range)

    self.dataset = loaded_dataset[dataset_split].select(indices)

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

  # Return the underlying dataset
  def get_dataset(self):
    return self.dataset

  # Extract the prompt from the provided data point
  def extract_prompt(self, data):
    datapoint_id = data["id"]
    user_prompt = data["definition"] + "\n" + data["inputs"]

    return user_prompt, datapoint_id