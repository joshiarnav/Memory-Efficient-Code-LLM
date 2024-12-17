import json
import logging
from operator import is_
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from together import Together
import threading
import random
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
# MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
MODEL = "meta-llama/Llama-Vision-Free"

class RateLimiter:
    def __init__(self, rate=1.0):
        self.rate = rate
        self.tokens = 1.0
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self):
        with self.lock:
            current = time.time()
            time_passed = current - self.last_update
            self.tokens = min(1.0, self.tokens + time_passed * self.rate)
            
            if self.tokens < 1.0:
                sleep_time = (1.0 - self.tokens) / self.rate
                time.sleep(sleep_time)
                self.tokens = 0.0
                self.last_update = time.time()
            else:
                self.tokens -= 1.0
                self.last_update = current

# rate_limiter = RateLimiter(rate=29.9)
rate_limiter = RateLimiter(rate=(1/6.1))
thread_local = threading.local()

def get_client():
    if not hasattr(thread_local, "together_client"):
        thread_local.together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    return thread_local.together_client

def generate_inefficient_prompt(description, efficient_solution):
    return f"""You are an expert at writing inefficient Python code. Given a programming problem and its efficient solution, your task is to write a memory inefficient version of the solution. The inefficient solution should:
1. Use excessive memory allocation
2. Still produce the correct output
3. Be significantly less memory efficient than the original

Problem Description:
{description}

Efficient Solution:
{efficient_solution}

Write ONLY the inefficient solution code, without any explanations:"""

def extract_code(response):
    """Extract code from model response, handling various formats."""
    is_python = True
    # If response contains code blocks, extract from them
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0].strip()
    elif "```" in response:
        code = response.split("```")[1].strip()
    else:
        # If no code blocks, use the entire response
        code = response.strip()
        is_python = False
    return code, is_python

def save_problem(output_dir, problem_data):
    """Save a single problem to its own JSON file."""
    problem_idx = problem_data["problem_idx"]
    filename = os.path.join(output_dir, f"{problem_idx}.json")
    with open(filename, 'w') as f:
        json.dump(problem_data, f, indent=2)
    # logger.info(f"Saved problem {problem_idx} to {filename}")

def generate_inefficient_solution(example, output_dir):
    """Generate an inefficient solution for a single example."""
    base_result = {
        "problem_idx": example["problem_idx"],
        "task_name": example["task_name"],
        "markdown_description": example["markdown_description"],
        "canonical_solution": example["canonical_solution"],
        "test_case_generator": example["test_case_generator"],
        "test_case": example["test_case"]
    }
    
    try:
        description = example["markdown_description"]
        efficient_solution = example["canonical_solution"]
        
        prompt = generate_inefficient_prompt(description, efficient_solution)
        messages = [{"role": "user", "content": prompt}]
        
        rate_limiter.acquire()
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            # max_tokens=2000
        )
        
        inefficient_solution, is_python = extract_code(response.choices[0].message.content)
        
        result = {
            **base_result,
            "inefficient_solution": inefficient_solution,
            "is_python": is_python
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error generating solution for problem {example['problem_idx']}: {error_msg}")
        result = {
            **base_result,
            "error": error_msg
        }
    
    save_problem(output_dir, result)
    return result

def process_dataset(output_dir="inefficient_solutions", max_workers=1, sample_size=-1):
    """Process the entire dataset and generate inefficient solutions."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "inefficient_solutions.json")
    
    # Load dataset
    dataset = load_dataset("DONG19/EffiBench", split="train")
    total_problems = len(dataset)
    random.seed(42)
    if sample_size > 0:
        dataset = dataset.select(random.sample(range(total_problems), sample_size))
        total_problems = len(dataset)
    # logger.info(f"Loaded {total_problems} problems from EffiBench dataset")
    
    # Process examples in parallel
    solutions = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(generate_inefficient_solution, example, output_dir): example["problem_idx"]
            for example in dataset
        }
        
        progress_bar = tqdm(total=total_problems, desc="Generating Solutions")
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result:
                    solutions[str(idx)] = result
                    # logger.info(f"Generated solution for problem {idx} ({len(solutions)}/{total_problems})")
                progress_bar.update(1)
            except Exception as e:
                logger.error(f"Error in future for problem {idx}: {str(e)}")
                progress_bar.update(1)
        progress_bar.close()
    
    # Save combined results
    with open(output_file, 'w') as f:
        json.dump(solutions, f, indent=2)
    # logger.info(f"Saved {len(solutions)} solutions to {output_file}")

if __name__ == "__main__":
    process_dataset()
