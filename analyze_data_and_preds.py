import json
import ast
from collections import Counter

file_path = 'datasets/inverse_scaling/isp-data-json/redefine_classification.jsonl'

correct_answers = []
classes_set = set()

with open(file_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        try:
            cls_list = ast.literal_eval(item['classes'])
            ans = cls_list[item['answer_index']]
            correct_answers.append(ans)
            for c in cls_list:
                classes_set.add(c)
        except:
            pass

print(f"Total samples: {len(correct_answers)}")
print(f"Class distribution: {Counter(correct_answers)}")
print(f"Unique classes: {classes_set}")
print(f"Majority baseline: {max(Counter(correct_answers).values()) / len(correct_answers):.4f}")
