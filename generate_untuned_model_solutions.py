import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_from_disk
from together import Together
import threading
import random
from tqdm import tqdm

output_dir = "raw_model_generated_solutions"
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

class RateLimiter:
    def __init__(self, rate=1.0):
        self.rate = rate
        self.last_request_time = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < 1.0 / self.rate:
                time.sleep(1.0 / self.rate - time_since_last_request)
            self.last_request_time = time.time()

rate_limiter = RateLimiter(rate=29.9)

thread_local = threading.local()

def get_together_client():
    if not hasattr(thread_local, "together_client"):
        thread_local.together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    return thread_local.together_client

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

def generate_efficient_prompt(description, inefficient_solution):
    return f"""Below is a programming problem description and an inefficient solution. Your task is to provide a more efficient solution.

### Problem Description:
{description}

### Inefficient Solution:
{inefficient_solution}

### Efficient Solution:"""

def generate_solution(example):
    try:
        client = get_together_client()
        rate_limiter.wait()

        prompt = generate_efficient_prompt(example['description'], example['inefficient_solution'])
        messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            messages=messages,
            model=MODEL,
            # max_tokens=2048,
            temperature=0.7,
            # top_k=50,
            # top_p=0.7,
            # repetition_penalty=1.0,
            # stop=["###"]
        )

        generated_text = response.choices[0].message.content

        # Find python solution which is everything after "### Efficient Solution:"
        model_solution, is_python = extract_code(generated_text)
        
        # Prepare output dictionary
        output_data = example.copy()
        output_data['model_generation'] = generated_text
        output_data['model_solution'] = model_solution
        output_data['is_python'] = is_python
        
        # Save to JSON file
        output_path = os.path.join(output_dir, f"{example['problem_idx']}.json")
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)
        
        logger.info(f"Successfully processed and saved problem_idx: {example['problem_idx']}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing problem_idx {example['problem_idx']}: {str(e)}")
        return False

def main(output_dir="raw_model_generated_solutions", max_workers=16):
    output_dir = "raw_model_generated_solutions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_from_disk("./dataset")
    dataset = dataset['test']
    logger.info(f"Loaded dataset with {len(dataset)} examples")
    
    successful_generations = 0
    failed_generations = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(generate_solution, example): example['problem_idx'] 
            for example in dataset
        }
        
        for future in tqdm(as_completed(future_to_idx), total=len(dataset), desc="Generating solutions"):
            idx = future_to_idx[future]
            try:
                success = future.result()
                if success:
                    successful_generations += 1
                else:
                    failed_generations += 1
            except Exception as e:
                logger.error(f"Error processing problem_idx {idx}: {str(e)}")
                failed_generations += 1
    
    logger.info(f"Generation complete. Successful: {successful_generations}, Failed: {failed_generations}")

if __name__ == "__main__":
    main()
