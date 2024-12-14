import OpenPromptInjection as PI
from OpenPromptInjection.utils import open_config
import json
import random

dataset = []

task_config_paths = ['./configs/task_configs/sst2_config.json', './configs/task_configs/rte_config.json',
                      './configs/task_configs/mrpc_config.json', './configs/task_configs/jfleg_config.json', './configs/task_configs/hsol_config.json'
                      ]

# Create the model
model_config = open_config(config_path='./configs/model_configs/palm2_config.json')
model = PI.create_model(config=model_config)

#for each comb, create a target app and generate injections
for i in range(len(task_config_paths)):
    target_task = PI.create_task(open_config(config_path=task_config_paths[i]), 600)
    # Create the LLM-integrated App
    target_app = PI.create_app(target_task, model, defense='no')
    # gnerate attacked data prompts 
    for i, (data_prompt, ground_truth_label) in enumerate(target_app):
        #create a dictionary
        dict = {}
        dict["target_task"] = target_task.task
        dict["injected_task"] = "None"
        dict["instruction"] = target_app.instruction
        dict["input"] = data_prompt
        dict["type"] = "benign"
        dict["flag"] = 0
        dataset.append(dict)


print(len(dataset))

random.shuffle(dataset)

out_file = open("usenix_benign.json", "w")
json.dump(dataset, out_file, indent = 4, sort_keys = False)
out_file.close()