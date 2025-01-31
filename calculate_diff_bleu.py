import json
import os
import sys
from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tempfile
import subprocess
import re
from tqdm import tqdm
import ast
from typing import Set, Dict, Any
from Levenshtein import distance
import statistics

# Directory paths
GENERATED_SOLUTIONS_DIR = "generated_solutions"
RAW_MODEL_SOLUTIONS_DIR = "raw_model_generated_solutions"
OUTPUT_DIR = "diff_bleu_eval"

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

def calculate_diff_bleu(source_code: str, target: str, generated_code: str, language: str = "python") -> float:
    """Calculate the Diff BLEU score between source, target and generated code."""
    source_path = None
    target_path = None
    generated_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as source_temp, \
             tempfile.NamedTemporaryFile(mode='w', delete=False) as target_temp, \
             tempfile.NamedTemporaryFile(mode='w', delete=False) as generated_temp:
            
            # Clean and write the code to temporary files
            source_temp.write(remove_blank_lines(remove_comments(source_code, language)))
            target_temp.write(remove_blank_lines(remove_comments(target, language)))
            generated_temp.write(remove_blank_lines(remove_comments(generated_code, language)))
            
            source_path = source_temp.name
            target_path = target_temp.name
            generated_path = generated_temp.name
        
        # Calculate diffs using git
        command_diff_generated = f"git diff -U0 --no-index --ignore-all-space --ignore-blank-lines {source_path} {generated_path} | tail -n +5 | grep -v 'No newline at end of file'"
        command_diff_target = f"git diff -U0 --no-index --ignore-all-space --ignore-blank-lines {source_path} {target_path} | tail -n +5 | grep -v 'No newline at end of file'"
        
        # Run git diff commands with error checking
        diff_generated_proc = subprocess.run(command_diff_generated, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        diff_target_proc = subprocess.run(command_diff_target, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if diff_generated_proc.returncode != 0 and diff_generated_proc.returncode != 1:  # git diff returns 1 if files are different
            print(f"Error in generated diff command: {diff_generated_proc.stderr.decode()}")
            return 0.0
            
        if diff_target_proc.returncode != 0 and diff_target_proc.returncode != 1:
            print(f"Error in target diff command: {diff_target_proc.stderr.decode()}")
            return 0.0
        
        diff_generated = diff_generated_proc.stdout.decode()
        diff_target = diff_target_proc.stdout.decode()
        
        # If either diff is empty, return 0
        if not diff_generated.strip() or not diff_target.strip():
            return 0.0

        # Calculate BLEU score
        diff_generated_tokens = wordpunct_tokenize(diff_generated)
        diff_target_tokens = wordpunct_tokenize(diff_target)
        
        if not diff_generated_tokens or not diff_target_tokens:
            return 0.0
            
        return sentence_bleu([diff_target_tokens], diff_generated_tokens, smoothing_function=SmoothingFunction().method1)
        
    except Exception as e:
        print(f"Error in calculate_diff_bleu: {str(e)}")
        return 0.0
    finally:
        # Clean up temporary files
        for path in [source_path, target_path, generated_path]:
            if path:
                try:
                    os.remove(path)
                except:
                    pass

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
    """Process solutions from both directories and calculate similarity metrics."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of problem IDs from both directories
    generated_files = {f.split('.')[0] for f in os.listdir(GENERATED_SOLUTIONS_DIR) if f.endswith('.json')}
    raw_model_files = {f.split('.')[0] for f in os.listdir(RAW_MODEL_SOLUTIONS_DIR) if f.endswith('.json')}
    
    # Find common problems
    common_problems = generated_files.intersection(raw_model_files)
    print(f"Found {len(common_problems)} problems in both directories")
    
    # Track statistics
    total_processed = 0
    total_errors = 0
    all_metrics = {
        "finetuned_bleu": [],
        "raw_model_bleu": [],
        "finetuned_edit_sim": [],
        "raw_model_edit_sim": [],
        "finetuned_ast_sim": [],
        "raw_model_ast_sim": [],
        "finetuned_diff_bleu": [],
        "raw_model_diff_bleu": []
    }
    
    # Process each common problem
    for problem_id in tqdm(common_problems, desc="Processing problems"):
        try:
            print(f"\nProcessing problem {problem_id}:")
            # Load both solutions
            generated_path = os.path.join(GENERATED_SOLUTIONS_DIR, f"{problem_id}.json")
            raw_model_path = os.path.join(RAW_MODEL_SOLUTIONS_DIR, f"{problem_id}.json")
            
            print(f"Loading files from:")
            print(f"- Generated: {generated_path}")
            print(f"- Raw Model: {raw_model_path}")
            
            generated_data = load_json_file(generated_path)
            raw_model_data = load_json_file(raw_model_path)
            
            if not generated_data or not raw_model_data:
                print(f"Failed to load data files for problem {problem_id}")
                total_errors += 1
                continue
                
            # Get the solutions
            finetuned_solution = generated_data.get('model_solution', '')
            raw_model_solution = raw_model_data.get('model_solution', '')
            inefficient_solution = raw_model_data.get('inefficient_solution', '')
            efficient_solution = generated_data.get('efficient_solution', '')
            
            if not finetuned_solution or not raw_model_solution or not inefficient_solution or not efficient_solution:
                print(f"Missing required solutions for problem {problem_id}")
                total_errors += 1
                continue
            
            language = "python" if raw_model_data.get('is_python', True) else "other"
            
            # Calculate all metrics
            # 1. BLEU scores
            finetuned_bleu = calculate_code_bleu(finetuned_solution, efficient_solution, language)
            raw_model_bleu = calculate_code_bleu(raw_model_solution, efficient_solution, language)
            
            # 2. Edit distance similarity
            finetuned_edit = calculate_normalized_edit_distance(finetuned_solution, efficient_solution, language)
            raw_model_edit = calculate_normalized_edit_distance(raw_model_solution, efficient_solution, language)
            
            # 3. AST similarity
            finetuned_ast = calculate_ast_similarity(finetuned_solution, efficient_solution)
            raw_model_ast = calculate_ast_similarity(raw_model_solution, efficient_solution)
            
            # 4. Diff BLEU scores
            finetuned_diff_bleu = calculate_diff_bleu(inefficient_solution, efficient_solution, finetuned_solution, language)
            raw_model_diff_bleu = calculate_diff_bleu(inefficient_solution, efficient_solution, raw_model_solution, language)
            
            # Collect metrics
            metrics = {
                "finetuned_model": {
                    "bleu_score": finetuned_bleu,
                    "edit_similarity": finetuned_edit,
                    "ast_similarity": finetuned_ast,
                    "diff_bleu": finetuned_diff_bleu
                },
                "raw_model": {
                    "bleu_score": raw_model_bleu,
                    "edit_similarity": raw_model_edit,
                    "ast_similarity": raw_model_ast,
                    "diff_bleu": raw_model_diff_bleu
                }
            }
            
            # Track non-zero scores
            if finetuned_bleu > 0: all_metrics["finetuned_bleu"].append(finetuned_bleu)
            if raw_model_bleu > 0: all_metrics["raw_model_bleu"].append(raw_model_bleu)
            if finetuned_edit > 0: all_metrics["finetuned_edit_sim"].append(finetuned_edit)
            if raw_model_edit > 0: all_metrics["raw_model_edit_sim"].append(raw_model_edit)
            if finetuned_ast > 0: all_metrics["finetuned_ast_sim"].append(finetuned_ast)
            if raw_model_ast > 0: all_metrics["raw_model_ast_sim"].append(raw_model_ast)
            if finetuned_diff_bleu > 0: all_metrics["finetuned_diff_bleu"].append(finetuned_diff_bleu)
            if raw_model_diff_bleu > 0: all_metrics["raw_model_diff_bleu"].append(raw_model_diff_bleu)
            
            # Calculate deltas
            deltas = {
                "bleu_delta": finetuned_bleu - raw_model_bleu,
                "edit_sim_delta": finetuned_edit - raw_model_edit,
                "ast_sim_delta": finetuned_ast - raw_model_ast,
                "diff_bleu_delta": finetuned_diff_bleu - raw_model_diff_bleu
            }
            
            # Save results
            output_data = {
                "problem_idx": problem_id,
                "metrics": metrics,
                "deltas": deltas,
                "solutions": {
                    "inefficient": inefficient_solution,
                    "efficient": efficient_solution,
                    "finetuned": finetuned_solution,
                    "raw_model": raw_model_solution
                },
                "is_python": raw_model_data.get('is_python', True)
            }
            
            output_path = os.path.join(OUTPUT_DIR, f"{problem_id}.json")
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            total_processed += 1
            
        except Exception as e:
            print(f"Error processing problem {problem_id}: {str(e)}")
            total_errors += 1
            continue
    
    # Calculate statistics for each metric
    statistics = {metric: calculate_statistics(scores) for metric, scores in all_metrics.items()}
    
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
