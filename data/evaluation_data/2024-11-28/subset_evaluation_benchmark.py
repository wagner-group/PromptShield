import json

filepath = "2024-11-28_evaluation_benchmark.json"

with open(filepath, encoding='utf-8') as data_file:
    datasetJSON = list(json.loads(data_file.read()))


chat_data = []
app_data = []

for datapoint in datasetJSON:
    if datapoint['source'] == "LMSYS":
        chat_data.append(datapoint)
    else:
        app_data.append(datapoint)

print(f"There are {len(chat_data)} conversational datapoints.")
print(f"There are {len(app_data)} application datapoints.")


#save dataset in json file
out_file = open(f"2024-11-28_evaluation_benchmark_conversation.json", "w")
json.dump(chat_data, out_file, indent = 4, sort_keys = False)
out_file.close()

out_file = open(f"2024-11-28_evaluation_benchmark_application.json", "w")
json.dump(app_data, out_file, indent = 4, sort_keys = False)
out_file.close()