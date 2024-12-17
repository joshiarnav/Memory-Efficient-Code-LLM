
import json
import re 
import os 
import sys
import math
import tokenize 
import tiktoken
import tempfile
import jsonlines
import subprocess
import scipy.stats
from io import StringIO
from statistics import mean
from tree_sitter import Language, Parser
from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Function to load JSON data from a file
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to remove comments (using your provided utility)
def remove_comments(code: str, language: str) -> str:
    if language.lower() == "python":
        try:
            return remove_py_comments(code)
        except:
            pattern = r"\'{3}[\s\S]*?\'{3}|\"{3}[\s\S]*?\"{3}|\#[^\n]*"
    else:
        pattern = r"\/\*[\s\S]*?\*\/|\/\/[^\n]*"
    return re.sub(pattern, '', code)

# Function to remove blank lines
def remove_blank_lines(code) -> str:
    try:
        lines = code.split("\n")
        non_blank_lines = [line for line in lines if line.strip() != ""]
        return "\n".join(non_blank_lines)
    except:
        return code

# Function to calculate Diff BLEU
def diff_bleu(source_code, target, generated_code, language):
    """Calculating the Diff BLEU score."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as source_temp, \
         tempfile.NamedTemporaryFile(mode='w', delete=False) as target_temp, \
         tempfile.NamedTemporaryFile(mode='w', delete=False) as generated_temp:
        
        source_temp.write(remove_blank_lines(remove_comments(source_code, language)))
        target_temp.write(remove_blank_lines(remove_comments(target, language)))
        generated_temp.write(remove_blank_lines(remove_comments(generated_code, language)))
        
        source_path = source_temp.name
        target_path = target_temp.name
        generated_path = generated_temp.name
    
    command_diff_generated = f"git diff -U0 --no-index --ignore-all-space --ignore-blank-lines {source_path} {generated_path} | tail -n +5 | grep -v 'No newline at end of file'"
    command_diff_target = f"git diff -U0 --no-index --ignore-all-space --ignore-blank-lines {source_path} {target_path} | tail -n +5 | grep -v 'No newline at end of file'"
    
    diff_generated = subprocess.run(command_diff_generated, shell=True, stdout=subprocess.PIPE).stdout.decode()
    diff_target = subprocess.run(command_diff_target, shell=True, stdout=subprocess.PIPE).stdout.decode()

    diff_generated_tokens = wordpunct_tokenize(diff_generated)
    diff_target_tokens = wordpunct_tokenize(diff_target)

    diff_score_bleu = sentence_bleu([diff_target_tokens], diff_generated_tokens, smoothing_function=SmoothingFunction().method1)
    return diff_score_bleu

# Load target codes
def extract_target_codes(data):
    target_codes = []
    if isinstance(data, dict) and 'target_code' in data:
        target_codes.append(data['target_code'])
    else:
        print(f"Unexpected structure: {data}")
    return target_codes

target_codes = []
with open('test.json', 'r') as file:
    for line in file:
        try:
            data = json.loads(line)
            target_codes.extend(extract_target_codes(data))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e} - skipping this line.")
            continue

generations_base = load_json_file('generations_base.json')
generations_finetuned = load_json_file('cleaned_generations.json')


# Evaluate Diff BLEU
source_code = ""  # Replace with your actual source code as a string
diff_bleu_scores = []
for target, generation in zip(target_codes, generations_base):
    score = diff_bleu(source_code, target, generation, language="python")
    diff_bleu_scores.append(score)

print("Diff BLEU Scores:", diff_bleu_scores)

# Function to evaluate functional correctness
def test_code(target_code, generated_code):
    try:
        exec(target_code)
        target_output = locals()
        exec(generated_code)
        generated_output = locals()
        return target_output == generated_output
    except Exception as e:
        print(f"Error during code execution: {e}")
        return False

# Evaluate functional correctness
correctness_results = []
for target, generation in zip(target_codes, generations_finetuned):
    is_correct = test_code(target, generation)
    correctness_results.append(is_correct)

print("Functional correctness results:", correctness_results)

# Summary Statistics
print("Average Diff BLEU:", mean(diff_bleu_scores))
print("Functional Correctness Percentage:", 100 * sum(correctness_results) / len(correctness_results), "%")
