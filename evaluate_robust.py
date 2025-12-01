import json
import ast
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.model_selection import train_test_split

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "datasets/inverse_scaling/isp-data-json/redefine_classification.jsonl"
FT_MODEL_PATH = "results/redefine_math/checkpoint-747"
SEED = 42
TEST_SIZE = 0.2
FEW_SHOT_K = 5

# --- Data Loading ---
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            try:
                item['classes_list'] = ast.literal_eval(item['classes'])
            except:
                continue
            item['correct_answer'] = item['classes_list'][item['answer_index']]
            data.append(item)
    return data

print("Loading data...")
raw_data = load_data(DATA_FILE)
train_data, test_data = train_test_split(raw_data, test_size=TEST_SIZE, random_state=SEED)
print(f"Test set size: {len(test_data)}")

# --- Model ---
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

# --- Robust Evaluation Function ---
def get_log_prob(model, tokenizer, text, choice_start_idx):
    """Compute log prob of the choice tokens in the text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
    
    # We care about predictions FOR the choice tokens.
    # The choice starts at `choice_start_idx`.
    # The logit at `i` predicts token at `i+1`.
    # So we want logits at `choice_start_idx - 1` to predict `input_ids[choice_start_idx]`, etc.
    
    log_prob_sum = 0.0
    # range from start of choice to end of sequence
    # input_ids: [p1, p2, ..., c1, c2]
    # logits:    [l1, l2, ..., lc1, lc2]
    # l(p_last) predicts c1.
    
    for i in range(choice_start_idx, len(input_ids)):
        token_id = input_ids[i]
        # Logit that predicts this token is at i-1
        token_logits = logits[i-1]
        token_log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        log_prob_sum += token_log_probs[token_id].item()
        
    return log_prob_sum

def evaluate_robust(model, tokenizer, dataset, k_shot_examples=None):
    model.eval()
    correct = 0
    total = 0
    
    context = ""
    if k_shot_examples:
        for ex in k_shot_examples:
            context += f"{ex['prompt']} {ex['correct_answer']}\n"
            
    # Pre-calculate prompt length to know where choice starts?
    # Careful: tokenization of "A" + "B" != tokenization("A") + tokenization("B") sometimes.
    # But we can just encode `prompt` and `prompt + choice` and find the diff.
    
    for i, item in enumerate(dataset):
        prompt = context + item['prompt']
        choices = item['classes_list']
        correct_ans = item['correct_answer']
        
        choice_scores = []
        debug_info = []
        
        # Compute prompt tokens once
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        
        for choice in choices:
            full_text = prompt + choice
            # We verify where the choice starts
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            
            # If full_ids doesn't start with prompt_ids (due to merge), we approximate
            # We assume choice tokens are the new ones appended.
            # But usually safe to just take the new tokens.
            
            start_idx = len(full_ids) - len(tokenizer.encode(choice, add_special_tokens=False))
            # Better: start_idx = len(prompt_ids) ? 
            # Let's use len(prompt_ids) as start, assuming the prefix is stable.
            start_idx = len(prompt_ids)
            
            score = get_log_prob(model, tokenizer, full_text, start_idx)
            choice_scores.append(score)
            debug_info.append(f"'{choice}': {score:.2f}")
            
        predicted_index = np.argmax(choice_scores)
        predicted_ans = choices[predicted_index]
        
        if i < 3:
            print(f"Ex {i} Pred: {predicted_ans} | Correct: {correct_ans} | Scores: {debug_info}")

        if predicted_ans == correct_ans:
            correct += 1
        total += 1
        
        if total % 50 == 0:
            print(f"Evaluated {total}...")
            
    return correct / total

# --- 1. Zero-Shot ---
print("\n--- Zero-Shot (Robust) ---")
acc_zero = evaluate_robust(base_model, tokenizer, test_data, None)
print(f"Zero-Shot Accuracy: {acc_zero:.4f}")

# --- 2. Few-Shot ---
print("\n--- Few-Shot (Robust) ---")
few_shot_ex = train_data[:FEW_SHOT_K]
acc_few = evaluate_robust(base_model, tokenizer, test_data, few_shot_ex)
print(f"Few-Shot Accuracy: {acc_few:.4f}")

# --- 3. Finetuned ---
print("\n--- Finetuned (Robust) ---")
print(f"Loading adapter from {FT_MODEL_PATH}...")
ft_model = PeftModel.from_pretrained(base_model, FT_MODEL_PATH)
acc_ft = evaluate_robust(ft_model, tokenizer, test_data, None)
print(f"Finetuned Accuracy: {acc_ft:.4f}")

# Save
results = {
    "task": "redefine-math",
    "acc_zero": acc_zero,
    "acc_few": acc_few,
    "acc_ft": acc_ft
}
with open("results/redefine_math_robust.json", "w") as f:
    json.dump(results, f, indent=2)
