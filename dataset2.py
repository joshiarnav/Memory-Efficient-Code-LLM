import json
from datasets import Dataset, DatasetDict
from scipy.sparse import data
from sklearn.model_selection import train_test_split

# Prompt template for the model
prompt_template = """Below is a programming problem description and an inefficient solution. Your task is to provide a more efficient solution.

### Problem Description:
{}

### Inefficient Solution:
{}

### Efficient Solution:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

def load_inefficient_solutions(file_path):
    """Load the inefficient solutions dataset from JSON"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract relevant fields and create lists
    examples = []
    for idx, item in data.items():
        if "error" not in item:  # Skip entries with errors
            example = {
                'problem_idx': idx,
                'description': item['markdown_description'],
                'inefficient_solution': item['inefficient_solution'],
                'efficient_solution': item['canonical_solution']
            }
            examples.append(example)
    
    return examples

def formatting_prompts_func(examples):
    """Format examples into prompt format"""
    descriptions = examples["description"]
    inefficient_sols = examples["inefficient_solution"]
    efficient_sols = examples["efficient_solution"]
    
    texts = []
    for desc, ineff_sol, eff_sol in zip(descriptions, inefficient_sols, efficient_sols):
        text = prompt_template.format(desc, ineff_sol, eff_sol) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}

# Load dataset
examples = load_inefficient_solutions("/Users/arnav/Documents/Cornell/CS6158/6158_final_project/inefficient_solutions_4/inefficient_solutions.json")

# Create train/test split
train_examples, test_examples = train_test_split(
    examples, 
    test_size=0.2,
    random_state=42
)

# Convert to HuggingFace datasets
train_dataset = Dataset.from_list(train_examples)
test_dataset = Dataset.from_list(test_examples)
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Save datasets
train_dataset.save_to_disk("/Users/arnav/Documents/Cornell/CS6158/6158_final_project/dataset/train_dataset")
test_dataset.save_to_disk("/Users/arnav/Documents/Cornell/CS6158/6158_final_project/dataset/test_dataset")
dataset.save_to_disk("/Users/arnav/Documents/Cornell/CS6158/6158_final_project/dataset/dataset")

# Apply formatting
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
test_dataset = test_dataset.map(formatting_prompts_func, batched=True)

# Combine train and test datasets
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Save dataset as formatted dataset
train_dataset.save_to_disk("/Users/arnav/Documents/Cornell/CS6158/6158_final_project/dataset/train_dataset_formatted")
test_dataset.save_to_disk("/Users/arnav/Documents/Cornell/CS6158/6158_final_project/dataset/test_dataset_formatted")
dataset.save_to_disk("/Users/arnav/Documents/Cornell/CS6158/6158_final_project/dataset/dataset_formatted")