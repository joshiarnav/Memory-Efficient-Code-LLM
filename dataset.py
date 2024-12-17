import json
from datasets import load_dataset, Dataset, DatasetDict
import random

efficiency_prompt = """Below is a programming problem description and an inefficient solution. Your task is to provide a more efficient solution.

### Problem Description:
{}

### Inefficient Solution:
{}

### Efficient Solution:
{}"""

EOS_TOKEN = tokenizer.eos_token

def load_inefficient_solutions(file_path):
    """Load the inefficient solutions from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
        # Convert to a dict with problem_idx as key and inefficient_solution as value
        return {str(k): v.get("inefficient_solution") for k, v in data.items() if "inefficient_solution" in v and "error" not in v}

def formatting_func(examples, inefficient_solutions):
    """Format examples into prompt format"""
    descriptions = examples["markdown_description"]
    problem_ids = examples["problem_idx"]
    efficient_solutions = examples["canonical_solution"]
    
    texts = []
    for desc, pid, eff_soln in zip(descriptions, problem_ids, efficient_solutions):
        # Find matching inefficient solution
        ineff_soln = inefficient_solutions.get(str(pid))
        if ineff_soln:
            text = efficiency_prompt.format(desc, ineff_soln, eff_soln) + EOS_TOKEN
            texts.append(text)
    
    return {"text": texts}

def prepare_dataset(inefficient_solutions_path, test_size=0.2, seed=42):
    """Prepare the dataset for training and testing"""
    # Load the EffiBench dataset
    dataset = load_dataset("DONG19/EffiBench", split="train")
    
    # Load inefficient solutions, excluding any with errors
    inefficient_solutions = load_inefficient_solutions(inefficient_solutions_path)
    
    # Filter dataset to only include problems that have valid inefficient solutions
    dataset = dataset.filter(lambda x: str(x["problem_idx"]) in inefficient_solutions)
    
    # Create train/test split
    dataset = dataset.shuffle(seed=seed)
    split_idx = int(len(dataset) * (1 - test_size))
    train_dataset = dataset.select(range(split_idx))
    test_dataset = dataset.select(range(split_idx, len(dataset)))
    
    # Format the datasets
    train_dataset = train_dataset.map(
        lambda x: formatting_func(x, inefficient_solutions),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    test_dataset = test_dataset.map(
        lambda x: formatting_func(x, inefficient_solutions),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

# Usage
dataset = prepare_dataset("inefficient_solutions.json")