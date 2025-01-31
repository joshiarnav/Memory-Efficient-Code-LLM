import os
import json

avg_inefficient_memory = 0.0
avg_efficient_memory = 0.0
avg_finetuned_memory = 0.0
avg_raw_model_memory = 0.0

avg_inefficient_time = 0.0
avg_efficient_time = 0.0
avg_finetuned_time = 0.0
avg_raw_model_time = 0.0

for filename in os.listdir("./memory_metrics"): 
    if filename.endswith(".json"): 
        file_path = os.path.join("./memory_metrics", filename) 
        with open(file_path, 'r') as f: 
            data = json.load(f) 
            if data and "inefficient" in data and "efficient" in data and "finetuned" in data and "raw_model" in data: 
                # check if all their types are float
                if not isinstance(data["inefficient"]["avg_peak_memory"], float) or not isinstance(data["efficient"]["avg_peak_memory"], float) or not isinstance(data["finetuned"]["avg_peak_memory"], float) or not isinstance(data["raw_model"]["avg_peak_memory"], float):
                    continue
                # skip if any of their values are inf
                if data["inefficient"]["avg_peak_memory"] == float('inf') or data["efficient"]["avg_peak_memory"] == float('inf') or data["finetuned"]["avg_peak_memory"] == float('inf') or data["raw_model"]["avg_peak_memory"] == float('inf'):
                    continue
                avg_inefficient_memory += data["inefficient"]["avg_peak_memory"]
                avg_efficient_memory += data["efficient"]["avg_peak_memory"]
                avg_finetuned_memory += data["finetuned"]["avg_peak_memory"]
                avg_raw_model_memory += data["raw_model"]["avg_peak_memory"]

                # avg_inefficient_time += data["inefficient_time"]
                # avg_efficient_time += data["efficient_time"]
                # avg_finetuned_time += data["finetuned_time"]
                # avg_raw_model_time += data["raw_model_time"]

avg_inefficient_memory /= len(os.listdir("./memory_metrics"))
avg_efficient_memory /= len(os.listdir("./memory_metrics"))
avg_finetuned_memory /= len(os.listdir("./memory_metrics"))
avg_raw_model_memory /= len(os.listdir("./memory_metrics"))

# avg_inefficient_time /= len(os.listdir("./memory_metrics"))
# avg_efficient_time /= len(os.listdir("./memory_metrics"))
# avg_finetuned_time /= len(os.listdir("./memory_metrics"))
# avg_raw_model_time /= len(os.listdir("./memory_metrics"))

print(f"Average inefficient memory: {avg_inefficient_memory}")
print(f"Average efficient memory: {avg_efficient_memory}")
print(f"Average finetuned memory: {avg_finetuned_memory}")
print(f"Average raw_model memory: {avg_raw_model_memory}")

# print(f"Average inefficient time: {avg_inefficient_time}")
# print(f"Average efficient time: {avg_efficient_time}")
# print(f"Average finetuned time: {avg_finetuned_time}")
# print(f"Average raw_model time: {avg_raw_model_time}")