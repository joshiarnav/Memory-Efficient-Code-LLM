import os
import json
import time
import tracemalloc
from typing import List, Dict, Any, Optional, Union
import random
from datasets import load_from_disk
from statistics import mean
import multiprocessing as mp
from functools import partial
import tqdm

class MemoryMetricsCalculator:
    def __init__(self, diff_bleu_eval_dir: str, memory_metrics_dir: str, timeout_seconds: int = 5):
        self.diff_bleu_eval_dir = diff_bleu_eval_dir
        self.memory_metrics_dir = memory_metrics_dir
        self.timeout_seconds = timeout_seconds
        self.dataset = load_from_disk("./dataset/test")
        self.model_averages = {
            'inefficient': {'time': [], 'memory': [], 'success': []},
            'efficient': {'time': [], 'memory': [], 'success': []},
            'finetuned': {'time': [], 'memory': [], 'success': []},
            'raw_model': {'time': [], 'memory': [], 'success': []}
        }
        # Create memory_metrics directory if it doesn't exist
        os.makedirs(memory_metrics_dir, exist_ok=True)

    def load_solutions(self, problem_idx: str) -> Dict[str, str]:
        """Load solutions from diff_bleu_eval"""
        file_path = os.path.join(self.diff_bleu_eval_dir, f"{problem_idx}.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
            return {
                'inefficient': data['solutions']['inefficient'],
                'efficient': data['solutions']['efficient'],
                'finetuned': data['solutions']['finetuned'],
                'raw_model': data['solutions']['raw_model']
            }

    def generate_test_input(self, problem_idx: str) -> List[Dict[str, Any]]:
        """Generate test inputs based on the problem type"""
        examples = [ex for ex in self.dataset if ex['problem_idx'] == problem_idx]
        if not examples:
            raise ValueError(f"No examples found for problem {problem_idx}")
            
        example = examples[0]
        test_cases = []
        
        # Extract function name and signature from solution
        solution_code = example.get('efficient_solution', '')
        if not solution_code:
            print(f"No solution code found for problem {problem_idx}")
            return []
            
        try:
            # Parse the solution code to understand input types
            lines = solution_code.split('\n')
            method_name = None
            method_params = []
            
            # Find the first method definition after class Solution
            in_solution_class = False
            for line in lines:
                if 'class Solution' in line:
                    in_solution_class = True
                    continue
                if in_solution_class and 'def' in line and '(' in line and ')' in line:
                    if '__init__' not in line:  # Skip constructor
                        # Found function definition
                        method_name = line.split('def ')[1].split('(')[0].strip()
                        params = line.split('(')[1].split(')')[0]
                        if 'self' in params:
                            params = params.replace('self,', '').replace('self', '')
                        method_params = []
                        for param in params.split(','):
                            if param.strip():
                                param_parts = param.strip().split(':')
                                param_name = param_parts[0].strip()
                                param_type = param_parts[1].strip() if len(param_parts) > 1 else ''
                                method_params.append((param_name, param_type))
                        break
            
            if method_name:
                # Generate test inputs based on parameter types
                test_inputs = []
                for param_name, param_type in method_params:
                    if 'List[int]' in param_type or 'List' in param_type:
                        test_inputs.append([1, 2, 3, 4, 5])
                    elif 'str' in param_type.lower():
                        test_inputs.append("test")
                    elif 'int' in param_type.lower():
                        test_inputs.append(10)
                    elif 'float' in param_type.lower():
                        test_inputs.append(1.0)
                    elif 'bool' in param_type.lower():
                        test_inputs.append(True)
                    else:
                        test_inputs.append(None)
                
                # Create test case
                test_cases.append({
                    'function': method_name,
                    'args': test_inputs,
                    'description': example.get('description', '')
                })
                print(f"Generated test case for problem {problem_idx}:")
                print(f"  Function: {method_name}")
                print(f"  Parameter Types: {method_params}")
                print(f"  Generated Inputs: {test_inputs}")
                    
        except Exception as e:
            print(f"Error generating test cases for problem {problem_idx}: {str(e)}")
            print(f"Solution code:\n{solution_code}")
        
        return test_cases

    def get_test_cases(self, problem_idx: str) -> List[Dict[str, Any]]:
        """Get test cases for the problem"""
        return self.generate_test_input(problem_idx)

    def measure_execution(self, solution_code: str, test_case: Dict[str, Any]) -> Dict[str, float]:
        """Measure execution time and memory usage for a single test case"""
        setup_code = """
from typing import List, Dict, Any, Optional, Union, Tuple
import collections
import heapq
import bisect
import math
import random
import itertools
from collections import defaultdict, deque, Counter
import sys
from math import inf

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def list_to_tree(values):
    if not values:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root

def list_to_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head
"""
        # Extract the actual function name from the solution code
        try:
            namespace = {}
            exec(setup_code + solution_code, namespace)
            
            # Find the function name in the solution code
            solution_class = namespace['Solution']()
            
            function_name = test_case['function']
            if not hasattr(solution_class, function_name):
                print(f"Error: Function {function_name} not found in solution")
                return {'execution_time': float('inf'), 'peak_memory': float('inf')}
            
            print(f"Executing function: {function_name}")
            print(f"Arguments: {test_case['args']}")
            
            # Process arguments - convert lists to special data structures if needed
            processed_args = []
            for arg in test_case['args']:
                if isinstance(arg, list):
                    # Check if this might be a tree or linked list
                    if any(isinstance(x, (list, tuple)) for x in arg):
                        # Probably not a tree or linked list
                        processed_args.append(arg)
                    else:
                        # Try both tree and linked list
                        if len(arg) > 0 and isinstance(arg[0], (int, float)):
                            if 'tree' in function_name.lower():
                                processed_args.append(list_to_tree(arg))
                            elif 'list' in function_name.lower() and 'list' not in arg:
                                processed_args.append(list_to_linked_list(arg))
                            else:
                                processed_args.append(arg)
                        else:
                            processed_args.append(arg)
                else:
                    processed_args.append(arg)
            
            tracemalloc.start()
            start_time = time.time()
            
            result = getattr(solution_class, function_name)(*processed_args)
            
            execution_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            
            tracemalloc.stop()
            
            print(f"Execution successful. Result: {result}")
            return {
                'execution_time': execution_time,
                'peak_memory': peak / 1024 / 1024  # Convert to MB
            }
            
        except Exception as e:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            print(f"Error executing solution: {str(e)}")
            print(f"Solution code:\n{solution_code}")
            return {
                'execution_time': float('inf'),
                'peak_memory': float('inf')
            }

    def process_problem(self, problem_idx: str) -> Dict[str, Any]:
        """Process a single problem and return its metrics"""
        print(f"\nProcessing problem {problem_idx}...")
        try:
            solutions = self.load_solutions(problem_idx)
            test_cases = self.get_test_cases(problem_idx)
            metrics = {}
            
            for solution_name, solution_code in solutions.items():
                total_time = 0
                total_memory = 0
                successful_cases = 0
                
                for test_case in test_cases:
                    result = self.measure_execution(solution_code, test_case)
                    if result['execution_time'] != float('inf'):
                        total_time += result['execution_time']
                        total_memory += result['peak_memory']
                        successful_cases += 1
                
                if successful_cases > 0:
                    avg_time = total_time / successful_cases
                    avg_memory = total_memory / successful_cases
                    success_rate = successful_cases / len(test_cases)
                    metrics[solution_name] = {
                        'avg_execution_time': avg_time,
                        'avg_peak_memory': avg_memory,
                        'success_rate': success_rate
                    }
                else:
                    metrics[solution_name] = {
                        'avg_execution_time': float('inf'),
                        'avg_peak_memory': float('inf'),
                        'success_rate': 0.0
                    }
            
            # Save metrics to file
            output_path = os.path.join(self.memory_metrics_dir, f"{problem_idx}_metrics.json")
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return {'problem_idx': problem_idx, 'metrics': metrics}
            
        except Exception as e:
            print(f"Error processing problem {problem_idx}: {str(e)}")
            return {'problem_idx': problem_idx, 'error': str(e)}

    def run_parallel(self, num_processes: Optional[int] = None):
        """Run memory metrics calculation in parallel"""
        if num_processes is None:
            num_processes = mp.cpu_count() - 1  # Leave one CPU free
        
        # Get list of all problem IDs
        problem_ids = []
        for filename in os.listdir(self.diff_bleu_eval_dir):
            if filename.endswith('.json'):
                problem_ids.append(filename.split('.')[0])
        
        print(f"Processing {len(problem_ids)} problems using {num_processes} processes...")
        
        # Create a pool of workers
        with mp.Pool(processes=num_processes) as pool:
            # Process problems in parallel with progress bar
            results = list(tqdm.tqdm(
                pool.imap(self.process_problem, problem_ids),
                total=len(problem_ids),
                desc="Processing problems"
            ))
        
        # Aggregate results
        for result in results:
            if 'metrics' in result:
                problem_metrics = result['metrics']
                for model_name, metrics in problem_metrics.items():
                    if metrics['avg_execution_time'] != float('inf'):
                        self.model_averages[model_name]['time'].append(metrics['avg_execution_time'])
                        self.model_averages[model_name]['memory'].append(metrics['avg_peak_memory'])
                        self.model_averages[model_name]['success'].append(metrics['success_rate'])
        
        # Print overall averages
        print("\nOverall Model Averages:")
        for model_name, metrics in self.model_averages.items():
            if metrics['time']:  # Only print if we have data
                avg_time = mean([t for t in metrics['time'] if t != float('inf')] or [float('inf')])
                avg_memory = mean([m for m in metrics['memory'] if m != float('inf')] or [float('inf')])
                avg_success = mean(metrics['success'])
                
                print(f"\n{model_name}:")
                print(f"  Average execution time: {avg_time:.4f} seconds")
                print(f"  Average peak memory: {avg_memory:.4f} MB")
                print(f"  Average success rate: {avg_success:.2%}")

def main():
    calculator = MemoryMetricsCalculator(
        diff_bleu_eval_dir="/Users/arnav/Documents/Cornell/CS6158/6158_final_project/diff_bleu_eval",
        memory_metrics_dir="/Users/arnav/Documents/Cornell/CS6158/6158_final_project/memory_metrics"
    )
    
    # Run in parallel
    calculator.run_parallel()

if __name__ == "__main__":
    main()
