import OpenPromptInjection as PI
from OpenPromptInjection.utils import open_config
import json
import random

dataset = []

task_config_paths = ['./configs/task_configs/sst2_config.json', './configs/task_configs/rte_config.json',
                      './configs/task_configs/mrpc_config.json', './configs/task_configs/jfleg_config.json', './configs/task_configs/hsol_config.json'
                      ]

# using list comprehension to create task comb
task_comb = [[a, b] for a in task_config_paths for b in task_config_paths if a != b]

attack_types = ['naive', 'escape', 'ignore', 'fake_comp', 'combine']

# Create the model
model_config = open_config(config_path='./configs/model_configs/palm2_config.json')
model = PI.create_model(config=model_config)

#for each comb, create a target app and generate injections
for i in range(len(task_comb)):
    target_task = PI.create_task(open_config(config_path=task_comb[i][0]), 250)
    inject_task = PI.create_task(open_config(config_path=task_comb[i][1]), 250, for_injection=True)
    attack_type = random.choice(attack_types)
    attacker = PI.create_attacker(attack_type, inject_task)
    # Create the LLM-integrated App
    target_app = PI.create_app(target_task, model, defense='no')
    # gnerate attacked data prompts 
    for i, (data_prompt, ground_truth_label) in enumerate(target_app):
        if attack_type == 'combine' or attack_type == 'fake_comp':
            data_prompt_after_attack = attacker.inject(data_prompt, i, target_task=target_task.task)
        else:
            data_prompt_after_attack = attacker.inject(data_prompt, i)

        #create a dictionary
        dict = {}
        dict["target_task"] = target_task.task
        dict["injected_task"] = inject_task.task
        dict["instruction"] = target_app.instruction
        dict["input"] = data_prompt_after_attack
        dict["type"] = attack_type
        dict["flag"] = 1
        dataset.append(dict)


print(len(dataset))

random.shuffle(dataset)

out_file = open("usenix_attacks.json", "w")
json.dump(dataset, out_file, indent = 4, sort_keys = False)
out_file.close()