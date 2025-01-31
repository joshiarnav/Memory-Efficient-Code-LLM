import os
import json
import time
import sys
import tracemalloc
from memory_profiler import memory_usage
from datasets import load_dataset

# Directory paths
diff_bleu_eval_dir = "./diff_bleu_eval"
# test_cases_dir = "./dataset/test"
output_dir = "./memory_metrics"

os.makedirs(output_dir, exist_ok=True)

# Helper function to dynamically run solution code
def run_solution(solution_code, test_case):
    try:
        global_vars = {}
        local_vars = {}
        # Combine solution code and test cases
        exec(solution_code, global_vars)
        exec(test_case, global_vars, local_vars)
        return True, "Passed"
    except Exception as e:
        return False, str(e)

# Measure memory and execution time for each solution
def measure_solution(solution_code, test_case):
    def wrapper():
        run_solution(solution_code, test_case)
    
    tracemalloc.start()
    mem_usage = memory_usage(wrapper, interval=0.01, max_usage=True)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return mem_usage, peak_memory / 1024 / 1024  # Convert to MB

# Main function to evaluate solutions
def evaluate_solutions():
    test_dataset = load_dataset("DONG19/EffiBench")
    test_dataset = test_dataset["train"]
    aggregate_metrics = {
        "inefficient": [],
        "efficient": [],
        "finetuned": [],
        "raw_model": []
    }

    for file_name in os.listdir(diff_bleu_eval_dir):
        if not file_name.endswith(".json"):
            continue

        with open(os.path.join(diff_bleu_eval_dir, file_name), 'r') as f:
            problem_data = json.load(f)

        problem_idx = problem_data["problem_idx"]
        solutions = problem_data["solutions"]

        test_case = None
        for example in test_dataset:
            # print(example)
            if example["problem_idx"] == problem_idx:
                test_case = example["test_case"]
                break

        if not test_case:
            print(f"No test case found for problem {problem_idx}")
            continue

        problem_metrics = {}
        print(f"Processing problem {problem_idx}...")

        for solution_name, solution_code in solutions.items():
            mem_usage, peak_mem = measure_solution(solution_code, test_case)
            success, message = run_solution(solution_code, test_case)

            problem_metrics[solution_name] = {
                "memory_usage_MB": mem_usage,
                "peak_memory_MB": peak_mem,
                "status": "Success" if success else f"Error: {message}"
            }

            if success:
                aggregate_metrics[solution_name].append(peak_mem)

        # Save metrics for the current problem
        with open(os.path.join(output_dir, f"{problem_idx}.json"), 'w') as out_file:
            json.dump(problem_metrics, out_file, indent=4)

    # Calculate and display average metrics
    print("\nAverage Memory Usage (MB):")
    for model, mem_list in aggregate_metrics.items():
        avg_memory = sum(mem_list) / len(mem_list) if mem_list else 0
        print(f"{model}: {avg_memory:.2f} MB")

if __name__ == "__main__":
    evaluate_solutions()
