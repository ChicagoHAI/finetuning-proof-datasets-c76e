from transformers import AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

choices = [" 1", " 4", " Yes", " No"]

for c in choices:
    ids = tokenizer.encode(c, add_special_tokens=False)
    print(f"'{c}' -> {ids} -> {[tokenizer.decode([i]) for i in ids]}")
