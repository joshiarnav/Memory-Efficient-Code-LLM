import os
import json

average_diff_bleu_model = 0.0
average_diff_bleu_raw = 0.0
total = 0.0 
count = 0

for filename in os.listdir("diff_bleu_eval"): 
    if filename.endswith(".json"): 
        file_path = os.path.join("diff_bleu_eval", filename) 
        with open(file_path, 'r') as f: 
            data = json.load(f) 
            if data and "diff_bleu_delta" in data: 
                total += data["diff_bleu_delta"]
                average_diff_bleu_model += data["finetuned_model_diff_bleu"]
                average_diff_bleu_raw += data["raw_model_diff_bleu"]
                count += 1

average = total / count if count > 0 else 0.0
average_diff_bleu_model = average_diff_bleu_model / count if count > 0 else 0.0
average_diff_bleu_raw = average_diff_bleu_raw / count if count > 0 else 0.0

print(f"Average diff_bleu_delta: {average:.4f}") 
print(f"Total files processed: {count}")
print(f"Average finetuned_model_diff_bleu: {average_diff_bleu_model:.4f}")
print(f"Average raw_model_diff_bleu: {average_diff_bleu_raw:.4f}")