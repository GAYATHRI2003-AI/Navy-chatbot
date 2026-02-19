import os
import sys
# Hack to remove conflicting global package
sys.path = [p for p in sys.path if "Object_Detection" not in p]

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NEW_MODEL_NAME = os.path.join(PROJECT_ROOT, "models", "qlora_adapters")
DATASET_FILE = os.path.join(PROJECT_ROOT, "data", "Miscellaneous", "indian_navy_qlora.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

def train():
    try:
        print(f"Loading model: {MODEL_NAME}")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load Base Model (Standard CPU compatible)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto" 
        )

        model.config.use_cache = False
        
        peft_config = LoraConfig(
            r=8, 
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )

        # 4. Load Dataset (JSONL)
        print(f"Loading local dataset: {DATASET_FILE}")
        dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

        # Format function for SFT (Instruction Tuning)
        def formatting_prompts_func(example):
            output_texts = []
            instructions = example['instruction']
            inputs = example['input']
            outputs = example['output']
            
            # Helper to ensure list
            if isinstance(instructions, str):
                instructions = [instructions]
                inputs = [inputs]
                outputs = [outputs]

            for i in range(len(instructions)):
                text = f"### Instruction: {instructions[i]}\n### Input: {inputs[i]}\n### Response: {outputs[i]}"
                output_texts.append(text)
            return {"text": output_texts}

        print("Formatting dataset...")
        dataset = dataset.map(formatting_prompts_func, batched=True)

        # 5. Args (SFTConfig for trl v0.26+)
        sft_config = None
        # Hack to remove max_seq_length if it fails
        try:
             sft_config = SFTConfig(
                output_dir=OUTPUT_DIR,
                dataset_text_field="text",
                max_seq_length=128,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=5,
                learning_rate=2e-4,
                logging_steps=1,
                use_cpu=True,
                save_strategy="no"
            )
        except TypeError:
             # Retry without max_seq_length
             print("SFTConfig failed with max_seq_length, retrying without...")
             sft_config = SFTConfig(
                output_dir=OUTPUT_DIR,
                dataset_text_field="text",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=5,
                learning_rate=2e-4,
                logging_steps=1,
                use_cpu=True,
                save_strategy="no"
            )

        print("Initializing Trainer...")
        # Try-catch for API changes
        try:
             trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                processing_class=tokenizer, # New API name
                args=sft_config,
                peft_config=peft_config,
            )
        except TypeError:
             print("Falling back to old API (tokenizer argument)...")
             trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                tokenizer=tokenizer, # Old API name
                args=sft_config,
                peft_config=peft_config,
            )

        print("Starting training...")
        trainer.train()
        
        print(f"Saving adapters to {NEW_MODEL_NAME}...")
        trainer.model.save_pretrained(NEW_MODEL_NAME)
        print("Training Complete!")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        with open("error.log", "w") as f:
            f.write(traceback.format_exc())

if __name__ == "__main__":
    train()
