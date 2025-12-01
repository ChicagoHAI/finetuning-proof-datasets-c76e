import json

file_path = 'datasets/inverse_scaling/isp-data-json/redefine_classification.jsonl'

print(f"Inspecting {file_path}...")
with open(file_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        data = json.loads(line)
        print(f"Sample {i}:")
        print(json.dumps(data, indent=2))
