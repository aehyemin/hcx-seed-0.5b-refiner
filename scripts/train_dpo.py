import os
import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
import wandb

MODEL_DIR = "./clova_model"
DATA_DIR = "datasets/train_dataset"
TRAIN_PATH = "datasets/train_dataset/train_9.jsonl"
EVAL_PATH = "datasets/train_dataset/test_1.jsonl"
OUTPUT_DIR = "./DPO-0.5B-Refined_f"


use_cuda = torch.cuda.is_available() 
use_bf16 = use_cuda and torch.cuda.is_bf16_supported() 
dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32) 

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map = "auto" if use_cuda else None,
    torch_dtype=dtype,
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True) 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

model.gradient_checkpointing_enable()
model.config.use_cache = False
train_dataset = load_dataset("json", data_files=TRAIN_PATH, split="train")
eval_dataset  = load_dataset("json", data_files=EVAL_PATH,  split="train")  

# config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj", 
#         "gate_proj", "up_proj", "down_proj"  
#     ],
# )

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=30,
    num_train_epochs=1,
    logging_steps=8,
    beta=0.2,                         
    remove_unused_columns=False,      
    per_device_train_batch_size=1,    
    gradient_accumulation_steps=16,    
    learning_rate=5e-6,             
    bf16=use_bf16,                 
    max_length=256,               
    report_to="wandb"                
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # peft_config=config
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("모델저장", {OUTPUT_DIR})