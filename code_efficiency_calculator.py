import os
import json
import time
import tracemalloc
from typing import List, Dict, Any, Optional, Union
import random

class CodeEfficiencyCalculator:
    def __init__(self, diff_bleu_eval_dir: str, timeout_seconds: int = 5):
        self.diff_bleu_eval_dir = diff_bleu_eval_dir
        self.timeout_seconds = timeout_seconds
        
    def load_solutions(self, problem_idx: str) -> Dict[str, str]:
        """Load all solutions for a given problem from diff_bleu_eval"""
        file_path = os.path.join(self.diff_bleu_eval_dir, f"{problem_idx}.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data['solutions']

    def parse_type_annotation(self, type_str: str) -> str:
        """Parse type annotation to determine parameter type"""
        type_str = type_str.strip()
        if 'List[int]' in type_str or 'List[float]' in type_str:
            return 'list_int'
        elif 'List[str]' in type_str:
            return 'list_str'
        elif 'List[List[int]]' in type_str:
            return 'list_list_int'
        elif 'str' in type_str:
            return 'str'
        else:
            return 'int'  # Default to int

    def generate_test_value(self, param_type: str) -> Any:
        """Generate a test value based on parameter type"""
        if param_type == 'int':
            return random.randint(1, 100)
        elif param_type == 'str':
            return ''.join(random.choices('abcdefghij', k=random.randint(1, 10)))
        elif param_type == 'list_int':
            return [random.randint(1, 100) for _ in range(random.randint(2, 10))]
        elif param_type == 'list_str':
            return [''.join(random.choices('abcdefghij', k=random.randint(1, 5))) 
                   for _ in range(random.randint(2, 5))]
        elif param_type == 'list_list_int':
            return [[random.randint(1, 100) for _ in range(random.randint(1, 5))]
                    for _ in range(random.randint(2, 5))]
        else:
            return random.randint(1, 100)  # Default to int

    def generate_test_cases(self, solution_code: str) -> List[Dict[str, Any]]:
        """Generate test cases based on function signature"""
        lines = solution_code.split('\n')
        for line in lines:
            if 'def ' in line:
                func_line = line.strip()
                func_name = func_line[func_line.index('def ')+4:func_line.index('(')]
                params_str = func_line[func_line.index('(')+1:func_line.index(')')].strip()
                
                if not params_str or params_str == 'self':
                    continue
                    
                params = [p.strip() for p in params_str.split(',') if p.strip() != 'self']
                param_types = []
                
                for param in params:
                    if ':' in param:
                        name, type_str = param.split(':')
                        param_types.append(self.parse_type_annotation(type_str))
                    else:
                        param_types.append('int')  # Default to int
                
                # Generate 5 test cases with random inputs
                test_cases = []
                for _ in range(5):
                    args = [self.generate_test_value(param_type) for param_type in param_types]
                    test_cases.append({
                        'function': func_name,
                        'args': args
                    })
                
                return test_cases
        
        return []

    def measure_execution(self, solution_code: str, test_case: Dict[str, Any]) -> Dict[str, float]:
        """Measure execution time and memory usage for a single test case"""
        # Add necessary imports and type definitions
        setup_code = """
from typing import List, Dict, Any, Optional, Union
from collections import Counter, defaultdict, deque
import collections
import heapq
import bisect
import math
import random
import itertools

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

"""
        full_code = setup_code + solution_code
        
        # Prepare the solution class
        try:
            # Create a new namespace for each execution
            namespace = {}
            exec(full_code, namespace)
            solution_class = namespace['Solution']()
            
            # Start memory tracking
            tracemalloc.start()
            start_time = time.time()
            
            # Execute solution
            _ = getattr(solution_class, test_case['function'])(*test_case['args'])
            
            execution_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            
            tracemalloc.stop()
            return {
                'execution_time': execution_time,
                'peak_memory': peak / 1024 / 1024  # Convert to MB
            }
            
        except Exception as e:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            print(f"Error executing solution: {str(e)}")
            return {
                'execution_time': float('inf'),
                'peak_memory': float('inf')
            }

    def calculate_efficiency_metrics(self, problem_idx: str) -> Dict[str, Dict[str, float]]:
        """Calculate efficiency metrics for all solutions of a problem"""
        try:
            solutions = self.load_solutions(problem_idx)
            metrics = {}
            
            # Use the first solution to generate test cases
            first_solution = next(iter(solutions.values()))
            test_cases = self.generate_test_cases(first_solution)
            
            if not test_cases:
                print(f"Could not generate test cases for problem {problem_idx}")
                return {}
            
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
                    metrics[solution_name] = {
                        'avg_execution_time': total_time / successful_cases,
                        'avg_peak_memory': total_memory / successful_cases,
                        'success_rate': successful_cases / len(test_cases)
                    }
                else:
                    metrics[solution_name] = {
                        'avg_execution_time': float('inf'),
                        'avg_peak_memory': float('inf'),
                        'success_rate': 0.0
                    }
            
            return metrics
        except Exception as e:
            print(f"Error calculating metrics for problem {problem_idx}: {str(e)}")
            return {}

def main():
    calculator = CodeEfficiencyCalculator(
        diff_bleu_eval_dir="/Users/arnav/Documents/Cornell/CS6158/6158_final_project/diff_bleu_eval"
    )
    
    # Process all problems in the diff_bleu_eval directory
    for filename in os.listdir(calculator.diff_bleu_eval_dir):
        if filename.endswith('.json'):
            problem_idx = filename.split('.')[0]
            print(f"\nProcessing problem {problem_idx}...")
            
            metrics = calculator.calculate_efficiency_metrics(problem_idx)
            if metrics:
                print(f"Efficiency metrics for problem {problem_idx}:")
                for solution_name, solution_metrics in metrics.items():
                    print(f"\n{solution_name}:")
                    for metric_name, value in solution_metrics.items():
                        print(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()