import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import psutil
import os
import numpy as np

def format_example(example):
    """Format a single example from Effibench dataset."""
    return f"""Problem: {example['description']}

Solution:
{example['canonical_solution']}"""

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def prepare_dataset():
    dataset = load_dataset("DONG19/EffiBench")
    
    # Format the dataset
    def format_function(examples):
        texts = [format_example(ex) for ex in examples]
        return {"text": texts}
    
    formatted_dataset = dataset.map(
        format_function,
        remove_columns=dataset["train"].column_names,
        batched=True
    )
    
    return formatted_dataset

def main():
    # Initialize memory tracking
    initial_memory = get_memory_usage()
    memory_measurements = []

    # Model configuration
    model_name = "NousResearch/Llama-2-7b-hf"  # Using 7B parameter model as it's more consumer-friendly
    
    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    dataset = prepare_dataset()

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=100,
        learning_rate=2e-4,
        fp16=True,
        save_strategy="epoch",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    # Record pre-training memory
    pre_training_memory = get_memory_usage()
    memory_measurements.append(("Pre-training", pre_training_memory))

    # Train the model
    trainer.train()

    # Record post-training memory
    post_training_memory = get_memory_usage()
    memory_measurements.append(("Post-training", post_training_memory))

    # Print memory usage statistics
    print("\nMemory Usage Statistics:")
    print(f"Initial Memory Usage: {initial_memory:.2f} MB")
    for stage, memory in memory_measurements:
        print(f"{stage} Memory Usage: {memory:.2f} MB")

if __name__ == "__main__":
    main()
