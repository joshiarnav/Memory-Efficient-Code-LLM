from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
from transformers import pipeline
import pandas as pd
import json
import requests
from sklearn.model_selection import train_test_split

def loadTokenizerAndModel(model_name):
    '''
    Load a tokenizer and model for a given model name
    
    Args:
        model_name: The name of the model to load
        
    Returns:
        A tuple containing the tokenizer and model
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def getRawMemoryDataset():
    '''
    Load the raw memory dataset
    
    Returns:
        A pandas DataFrame containing the raw memory dataset
    '''
    url = "https://raw.githubusercontent.com/microsoft/NoFunEval/refs/heads/main/datasets/resource_util.jsonl"
    data = [json.loads(line) for line in requests.get(url).text.split('\n')[:-1]]
    # Convert to a pandas DataFrame
    df = pd.DataFrame(data)
    return df

def reformatMemoryDataset(df):
    '''
    Reformat the raw memory dataset
    
    Args:
        df: A pandas DataFrame containing the raw memory dataset
    
    Returns:
        A pandas DataFrame containing the reformatted memory dataset
    '''
    # Assuming df is already loaded as a pandas DataFrame
    # Combine relevant columns (adjust as needed, here I'm using 'base_prompt' as input and 'target_code' as the output)
    df['input_text'] = df['base_prompt'] + ' ' + df['coding_concepts'] + ' ' + df['chain_of_thought'] + ' ' + df['source_code']
    df['output_text'] = df['target_code']
    # Drop unnecessary columns
    # df = df.drop(columns=['base_prompt', 'coding_concepts', 'chain_of_thought', 'source_code', 'target_code'])
    return df

def splitMemoryDataset(df, test_size=0.2):
    '''
    Split the memory dataset into training and testing sets
    
    Args:
        df: A pandas DataFrame containing the reformatted memory dataset
        test_size: The proportion of the dataset to include in the test split
        
    Returns:
        A tuple containing the training and testing sets as Datasets (not DataFrames)
    '''
    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_data = Dataset.from_pandas(train_df)
    test_data = Dataset.from_pandas(test_df)
    return train_data, test_data

def tokenizeMemoryDataset(data, tokenizer, max_length=2048):
    '''
    Tokenize the memory dataset
    
    Args:
        data: A Dataset containing the memory dataset
        tokenizer: The tokenizer to use for tokenization
        max_length: The maximum sequence length
        
    Returns:
        A Dataset containing the tokenized memory dataset
    '''
    def tokenize_function(examples):
        return tokenizer(examples['input_text'], examples['output_text'], max_length=max_length, padding="max_length", truncation=True)

    tokenized_dataset = data.map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.map(
    lambda examples: {'labels': examples['input_ids']},
    batched=True
    )

    return tokenized_dataset

def prepareMemoryDataset(tokenizer, test_size=0.2):
    '''
    Prepare the memory dataset for training
    
    Returns:
        A tuple containing the training and testing sets as Datasets (not DataFrames)
    '''
    # Load the raw memory dataset
    df = getRawMemoryDataset()
    # Reformat the memory dataset
    df = reformatMemoryDataset(df)
    # Split the memory dataset
    train_data, test_data = splitMemoryDataset(df, test_size=test_size)
    # Tokenize the memory dataset
    # tokenizer, model = loadTokenizerAndModel(model_name)
    train_data = tokenizeMemoryDataset(train_data, tokenizer)
    test_data = tokenizeMemoryDataset(test_data, tokenizer)
    return train_data, test_data

def createTrainer(model, train_data, tokenizer):
    '''
    Create a Trainer for training the model
    
    Args:
        model: The model to train
        train_data: The training dataset
        test_data: The testing dataset
        training_args: The training arguments
        
    Returns:
        A Trainer object for training the model
    '''
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_data,
    #     eval_dataset=test_data,
    #     data_collator=data_collator,
    #     tokenizer=tokenizer
    # )

    # Define the LoRA configuration
    lora_config = LoraConfig(
        r=6,  # rank for the LoRA layers
        lora_alpha=16,  # scaling factor for LoRA
        lora_dropout=0.1,  # dropout rate for LoRA
        task_type="CAUSAL_LM"  # causal language modeling task
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./lora_model",
        evaluation_strategy="epoch",  # Evaluate every epoch
        learning_rate=2e-5,           # Learning rate
        per_device_train_batch_size=4,  # Batch size
        num_train_epochs=3,           # Number of epochs
        logging_dir='./logs',         # Where to save logs
        save_steps=500,               # Save checkpoints
        logging_steps=100,            # Log every 100 steps
        fp16=True                     # Enable mixed precision (if supported)
    )

    # Set up the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer
    )

    return trainer