#!/usr/bin/env python
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.model_loader import load_model_and_tokenizer
from src.data.stream_extractor import stream_corpus
from src.data.dataset import partition_and_align

def main():
    print("Loading model...")
    # Use a small model for testing if needed, or OLMoE-1B-7B
    model_id = "allenai/OLMoE-1B-7B-0924"
    try:
        model, tokenizer = load_model_and_tokenizer(model_id, device="cpu")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Generating synthetic corpus...")
    corpus = ["This is a test document. " * 50 for _ in range(100)]
    
    print("Extracting...")
    batches = []
    # Assuming stream_corpus yields batches
    for batch in stream_corpus(model, corpus, tokenizer=tokenizer, seq_len=64, tap_layer=3):
        batches.append(batch)
        if len(batches) >= 10:
            break
            
    print(f"Extracted {len(batches)} batches. Splitting...")
    train, calib = partition_and_align(batches)
    
    print("Saving...")
    out_dir = project_root / "data"
    out_dir.mkdir(exist_ok=True)
    
    train.save(str(out_dir / "train_data.safetensors"))
    calib.save(str(out_dir / "calib_data.safetensors"))
    
    print("Done!")

if __name__ == "__main__":
    main()
