import json
import ast
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "datasets/inverse_scaling/isp-data-json/redefine_classification.jsonl"
OUTPUT_DIR = "results/redefine_math"
SEED = 42
TEST_SIZE = 0.2
FEW_SHOT_K = 5

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Data Loading ---
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Parse classes string to list
            try:
                item['classes_list'] = ast.literal_eval(item['classes'])
            except:
                continue # Skip bad lines
            item['correct_answer'] = item['classes_list'][item['answer_index']]
            data.append(item)
    return data

print(f"Loading data from {DATA_FILE}...")
raw_data = load_data(DATA_FILE)
print(f"Loaded {len(raw_data)} examples.")

# Split
train_data, test_data = train_test_split(raw_data, test_size=TEST_SIZE, random_state=SEED)
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# --- Model & Tokenizer ---
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load model in 16-bit to save memory
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

# --- Evaluation Function ---
def evaluate(model, tokenizer, dataset, k_shot_examples=None):
    model.eval()
    correct = 0
    total = 0
    
    # Prepare few-shot context if needed
    context = ""
    if k_shot_examples:
        for ex in k_shot_examples:
            context += f"{ex['prompt']} {ex['correct_answer']}\n"
    
    for i, item in enumerate(dataset):
        prompt = context + item['prompt']
        choices = item['classes_list']
        correct_ans = item['correct_answer']
        
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            
            choice_scores = []
            debug_info = []
            for choice in choices:
                choice_ids = tokenizer.encode(choice, add_special_tokens=False)
                if len(choice_ids) == 0:
                    choice_scores.append(-float('inf'))
                    continue
                token_id = choice_ids[0]
                score = next_token_logits[token_id].item()
                choice_scores.append(score)
                debug_info.append(f"'{choice}': {score:.4f}")
        
        predicted_index = np.argmax(choice_scores)
        predicted_ans = choices[predicted_index]
        
        if i < 5:
            print(f"Ex {i}: {prompt[-30:]} | Pred: '{predicted_ans}' | Correct: '{correct_ans}' | Scores: {debug_info}")
        
        if predicted_ans == correct_ans:
            correct += 1
        total += 1
        
        if total % 50 == 0:
            print(f"Evaluated {total}/{len(dataset)}...")

    accuracy = correct / total
    return accuracy

# --- 1. Baseline Evaluation ---
print("\n--- Running Zero-Shot Baseline ---")
acc_zero = evaluate(model, tokenizer, test_data, k_shot_examples=None)
print(f"Zero-Shot Accuracy: {acc_zero:.4f}")

print("\n--- Running Few-Shot Baseline ---")
# Use examples from train set for few-shot
few_shot_ex = train_data[:FEW_SHOT_K]
acc_few = evaluate(model, tokenizer, test_data, k_shot_examples=few_shot_ex)
print(f"Few-Shot ({FEW_SHOT_K}) Accuracy: {acc_few:.4f}")

# --- 2. Fine-Tuning ---
print("\n--- Preparing for Fine-Tuning ---")

# Format data for training: "Prompt Answer"
def format_for_training(data_list):
    formatted = []
    for item in data_list:
        formatted.append({
            "text": f"{item['prompt']} {item['correct_answer']}" + tokenizer.eos_token
        })
    return formatted

train_dataset_hf = Dataset.from_list(format_for_training(train_data))

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    # For CausalLM, labels are same as input_ids
    outputs["labels"] = outputs["input_ids"].copy()
    # Mask padding in labels with -100
    # 0 is usually padding for many tokenizers, or tokenizer.pad_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Replace pad tokens with -100
    labels = []
    for i in range(len(outputs["input_ids"])):
        label = [t if t != pad_token_id else -100 for t in outputs["input_ids"][i]]
        labels.append(label)
    outputs["labels"] = labels
    return outputs

tokenized_train = train_dataset_hf.map(tokenize_function, batched=True)

# LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

print("Starting Training...")
trainer.train()

# --- 3. Post-Training Evaluation ---
print("\n--- Running Fine-Tuned Evaluation ---")
# Note: Trainer wraps model. To eval, we use the trainer's model or merge.
# We can just call evaluate on model (it has the adapter active)
acc_ft = evaluate(model, tokenizer, test_data, k_shot_examples=None)
print(f"Fine-Tuned Accuracy: {acc_ft:.4f}")

# Save results
results = {
    "task": "redefine-math (inverse scaling)",
    "model": MODEL_NAME,
    "n_train": len(train_data),
    "n_test": len(test_data),
    "accuracies": {
        "zero_shot": acc_zero,
        "few_shot": acc_few,
        "finetuned": acc_ft
    }
}

with open("results/redefine_math_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done! Results saved.")
