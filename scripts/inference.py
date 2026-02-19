import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Configuration ---
BASE_MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit" # Must match train.py
ADAPTER_MODEL_NAME = "llama-3-8b-qlora-finetuned" # Must match train.py
PROMPT = "Write a review of a movie that is absolutely terrible:\n"

def generate():
    print(f"Loading base model: {BASE_MODEL_NAME}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    # 2. Load Base Model (Quantized)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # 3. Load Adapters
    print(f"Loading adapters from: {ADAPTER_MODEL_NAME}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_NAME)
    
    # 4. Inference
    print("Generating...")
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print("\n--- Output ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    generate()
