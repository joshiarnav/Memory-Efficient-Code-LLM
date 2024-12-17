import json
import os
import sys
from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import ast
from typing import Set, Dict, Any
from Levenshtein import distance
import statistics
from tqdm import tqdm
import re
import tempfile
import subprocess

# Directory paths
INEFFICIENT_SOLUTIONS_DIR = "inefficient_solutions"
OUTPUT_DIR = "diff_bleu_inefficient"

def remove_comments(code: str, language: str = "python") -> str:
    """Remove comments from code."""
    if language.lower() == "python":
        pattern = r"\'{3}[\s\S]*?\'{3}|\"{3}[\s\S]*?\"{3}|\#[^\n]*"
    else:
        pattern = r"\/\*[\s\S]*?\*\/|\/\/[^\n]*"
    return re.sub(pattern, '', code)

def remove_blank_lines(code: str) -> str:
    """Remove blank lines from code."""
    try:
        lines = code.split("\n")
        non_blank_lines = [line for line in lines if line.strip() != ""]
        return "\n".join(non_blank_lines)
    except:
        return code

def calculate_code_bleu(source_code: str, target_code: str, language: str = "python") -> float:
    """Calculate BLEU score between source and target code."""
    try:
        # Clean the code
        source = remove_blank_lines(remove_comments(source_code, language))
        target = remove_blank_lines(remove_comments(target_code, language))
        
        # Tokenize both codes
        source_tokens = wordpunct_tokenize(source)
        target_tokens = wordpunct_tokenize(target)
        
        if not source_tokens or not target_tokens:
            return 0.0
        
        return sentence_bleu([target_tokens], source_tokens, smoothing_function=SmoothingFunction().method1)
    except Exception as e:
        print(f"Error in calculate_code_bleu: {str(e)}")
        return 0.0

def calculate_normalized_edit_distance(source_code: str, target_code: str, language: str = "python") -> float:
    """Calculate normalized edit distance between source and target code."""
    try:
        # Clean the code
        source = remove_blank_lines(remove_comments(source_code, language))
        target = remove_blank_lines(remove_comments(target_code, language))
        
        # Calculate edit distance
        edit_dist = distance(source, target)
        max_len = max(len(source), len(target))
        
        # Return normalized similarity (1 - normalized_distance)
        return 1 - (edit_dist / max_len if max_len > 0 else 0)
    except Exception as e:
        print(f"Error in calculate_normalized_edit_distance: {str(e)}")
        return 0.0

def get_ast_nodes(code: str) -> Set[str]:
    """Get a set of AST node types from code."""
    try:
        tree = ast.parse(code)
        return {type(node).__name__ for node in ast.walk(tree)}
    except:
        return set()

def calculate_ast_similarity(source_code: str, target_code: str) -> float:
    """Calculate similarity based on AST node types."""
    try:
        source_nodes = get_ast_nodes(source_code)
        target_nodes = get_ast_nodes(target_code)
        
        if not source_nodes or not target_nodes:
            return 0.0
        
        # Jaccard similarity between node sets
        intersection = len(source_nodes.intersection(target_nodes))
        union = len(source_nodes.union(target_nodes))
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        print(f"Error in calculate_ast_similarity: {str(e)}")
        return 0.0

def calculate_diff_bleu(source_code: str, target_code: str, language: str = "python") -> float:
    """Calculate the Diff BLEU score between source and target code."""
    source_path = None
    target_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as source_temp, \
             tempfile.NamedTemporaryFile(mode='w', delete=False) as target_temp:
            
            # Clean and write the code to temporary files
            source_temp.write(remove_blank_lines(remove_comments(source_code, language)))
            target_temp.write(remove_blank_lines(remove_comments(target_code, language)))
            
            source_path = source_temp.name
            target_path = target_temp.name
        
        # Calculate diffs using git
        command_diff = f"git diff -U0 --no-index --ignore-all-space --ignore-blank-lines {source_path} {target_path} | tail -n +5 | grep -v 'No newline at end of file'"
        
        # Run git diff command with error checking
        diff_proc = subprocess.run(command_diff, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if diff_proc.returncode != 0 and diff_proc.returncode != 1:  # git diff returns 1 if files are different
            print(f"Error in diff command: {diff_proc.stderr.decode()}")
            return 0.0
        
        diff = diff_proc.stdout.decode()
        
        # If diff is empty, return 0
        if not diff.strip():
            return 0.0

        # Calculate BLEU score comparing the code against itself
        diff_tokens = wordpunct_tokenize(diff)
        
        if not diff_tokens:
            return 0.0
            
        return sentence_bleu([diff_tokens], diff_tokens, smoothing_function=SmoothingFunction().method1)
        
    except Exception as e:
        print(f"Error in calculate_diff_bleu: {str(e)}")
        return 0.0
    finally:
        # Clean up temporary files
        if source_path:
            try:
                os.remove(source_path)
            except:
                pass
        if target_path:
            try:
                os.remove(target_path)
            except:
                pass

def calculate_similarity_metrics(source_code: str, target_code: str, is_python: bool = True) -> Dict[str, float]:
    """Calculate all similarity metrics between source and target code."""
    language = "python" if is_python else "other"
    return {
        "bleu_score": calculate_code_bleu(source_code, target_code, language),
        "edit_similarity": calculate_normalized_edit_distance(source_code, target_code, language),
        "ast_similarity": calculate_ast_similarity(source_code, target_code),
        "diff_bleu_score": calculate_diff_bleu(source_code, target_code, language)
    }

def calculate_statistics(scores: list) -> Dict[str, float]:
    """Calculate statistics for a list of scores."""
    if not scores:
        return {
            "average": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "count": 0
        }
    
    return {
        "average": statistics.mean(scores),
        "median": statistics.median(scores),
        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "count": len(scores)
    }

def load_json_file(file_path: str) -> dict:
    """Load a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded {file_path}")
            print(f"Keys in file: {list(data.keys())}")
            return data
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def process_solutions():
    """Process solutions and calculate similarity metrics."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of all JSON files
    all_files = [f for f in os.listdir(INEFFICIENT_SOLUTIONS_DIR) if f.endswith('.json')]
    print(f"Found {len(all_files)} problems in inefficient solutions directory")
    
    # Track statistics
    total_processed = 0
    total_errors = 0
    all_metrics = {
        "bleu_score": [],
        "edit_similarity": [],
        "ast_similarity": [],
        "diff_bleu_score": []
    }
    
    # Process each problem
    for filename in tqdm(all_files, desc="Processing problems"):
        try:
            problem_id = filename.split('.')[0]
            print(f"\nProcessing problem {problem_id}:")
            
            # Load solution
            file_path = os.path.join(INEFFICIENT_SOLUTIONS_DIR, filename)
            print(f"Loading file: {file_path}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get the solutions
            inefficient_solution = data.get('inefficient_solution', '')
            canonical_solution = data.get('canonical_solution', '')
            
            if not inefficient_solution or not canonical_solution:
                print(f"Missing required solutions for problem {problem_id}")
                total_errors += 1
                continue
            
            # Calculate similarity metrics
            metrics = calculate_similarity_metrics(
                inefficient_solution,
                canonical_solution,
                data.get('is_python', True)
            )
            
            # Track metrics
            for metric, score in metrics.items():
                if score > 0:  # Only track non-zero scores
                    all_metrics[metric].append(score)
            
            # Save results
            output_data = {
                "problem_idx": problem_id,
                "metrics": metrics,
                "inefficient_solution": inefficient_solution,
                "canonical_solution": canonical_solution,
                "is_python": data.get('is_python', True)
            }
            
            output_path = os.path.join(OUTPUT_DIR, filename)
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            total_processed += 1
            
        except Exception as e:
            print(f"Error processing problem {problem_id}: {str(e)}")
            total_errors += 1
            continue
    
    # Calculate statistics for each metric
    statistics = {
        metric: calculate_statistics(scores)
        for metric, scores in all_metrics.items()
    }
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"Total problems processed successfully: {total_processed}")
    print(f"Total errors: {total_errors}")
    print("\nMetrics Statistics:")
    for metric, stats in statistics.items():
        print(f"\n{metric}:")
        print(f"  Average: {stats['average']:.4f}")
        print(f"  Median:  {stats['median']:.4f}")
        print(f"  StdDev:  {stats['std_dev']:.4f}")
        print(f"  Count:   {stats['count']}")
    
    # Save statistics
    stats_path = os.path.join(OUTPUT_DIR, "statistics.json")
    with open(stats_path, 'w') as f:
        json.dump({
            "total_processed": total_processed,
            "total_errors": total_errors,
            "metrics_statistics": statistics
        }, f, indent=2)

if __name__ == "__main__":
    process_solutions()
