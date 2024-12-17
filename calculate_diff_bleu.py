import json
import os
import sys
from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tempfile
import subprocess
import re
from tqdm import tqdm

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

def calculate_diff_bleu(source_code: str, target: str, generated_code: str, language: str = "python") -> float:
    """Calculate the Diff BLEU score between source, target and generated code."""
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
            try:
                os.remove(path)
            except:
                pass

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
    """Process solutions from both directories and calculate diff_bleu scores."""
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
    total_zero_scores = 0
    
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
            canonical_solution = generated_data.get('efficient_solution', '')  # Changed from canonical_solution to efficient_solution
            
            print("Solution lengths:")
            print(f"- Finetuned solution: {len(finetuned_solution)} chars")
            print(f"- Raw model solution: {len(raw_model_solution)} chars")
            print(f"- Inefficient solution: {len(inefficient_solution)} chars")
            print(f"- Efficient solution: {len(canonical_solution)} chars")
            
            if not finetuned_solution or not raw_model_solution or not inefficient_solution or not canonical_solution:
                print(f"Missing required solutions for problem {problem_id}")
                print(f"- Has finetuned: {bool(finetuned_solution)}")
                print(f"- Has raw model: {bool(raw_model_solution)}")
                print(f"- Has inefficient: {bool(inefficient_solution)}")
                print(f"- Has canonical_solution: {bool(canonical_solution)}")
                total_errors += 1
                continue
                
            # Calculate diff_bleu scores for both models
            raw_model_diff_bleu = calculate_diff_bleu(
                inefficient_solution,
                canonical_solution,
                raw_model_solution
            )
            
            finetuned_diff_bleu = calculate_diff_bleu(
                inefficient_solution,
                canonical_solution,
                finetuned_solution
            )
            
            if raw_model_diff_bleu == 0.0 and finetuned_diff_bleu == 0.0:
                total_zero_scores += 1
            
            # Combine data and save
            output_data = {
                "problem_idx": problem_id,
                "description": raw_model_data.get('description', ''),
                "inefficient_solution": inefficient_solution,
                "canonical_solution": canonical_solution,
                "finetuned_model_solution": finetuned_solution,
                "raw_model_solution": raw_model_solution,
                "finetuned_model_diff_bleu": finetuned_diff_bleu,
                "raw_model_diff_bleu": raw_model_diff_bleu,
                "is_python": raw_model_data.get('is_python', True),
                "diff_bleu_delta": finetuned_diff_bleu - raw_model_diff_bleu,
            }
            
            # Save to output directory
            output_path = os.path.join(OUTPUT_DIR, f"{problem_id}.json")
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            total_processed += 1
            
        except Exception as e:
            print(f"Error processing problem {problem_id}: {str(e)}")
            total_errors += 1
            continue
    
    # Print summary statistics
    print(f"\nProcessing complete:")
    print(f"Total problems processed successfully: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Problems with zero diff_bleu scores: {total_zero_scores}")

if __name__ == "__main__":
    process_solutions()
