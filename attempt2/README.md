# QLoRA Fine-tuning of Llama on EffiBench

This project implements QLoRA (Quantized Low-Rank Adaptation) fine-tuning of the Llama-2-7B model on the EffiBench dataset. The implementation focuses on memory efficiency and is designed to run on a single consumer GPU.

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

Run the training script:
```bash
python train.py
```

The script will:
1. Load the EffiBench dataset
2. Initialize a 4-bit quantized Llama-2-7B model
3. Apply QLoRA fine-tuning
4. Track and report memory usage throughout the process

## Implementation Details

- Uses 4-bit quantization with double quantization for memory efficiency
- Implements LoRA with rank=16 and alpha=32
- Targets query and value projection matrices for adaptation
- Tracks memory usage at different stages of training
- Uses mixed precision training (fp16)
- Implements gradient accumulation for better memory management

## Memory Usage

The script tracks memory usage at three points:
1. Initial memory usage (before model loading)
2. Pre-training memory usage (after model loading, before training)
3. Post-training memory usage (after training completion)

These measurements help evaluate the memory efficiency of the QLoRA implementation.
