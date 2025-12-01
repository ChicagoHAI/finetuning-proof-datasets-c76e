# Downloaded Datasets

This directory contains datasets for the research project.

## Dataset 1: Inverse Scaling Prize Datasets

### Overview
- **Source**: [Inverse Scaling Prize GitHub](https://github.com/inverse-scaling/prize)
- **Location**: `datasets/inverse_scaling/`
- **Format**: JSONL / CSV
- **Task**: Various logic and reasoning tasks where larger models perform worse (Inverse Scaling).
- **Key Sub-datasets**:
    - `redefine-math`: Tests ability to follow redefining mathematical operators.
    - `memo-trap`: Tests ability to avoid memorized completion when instructed otherwise.
    - `resisting-correction`: Tests ability to accept correction against priors.

### Loading
Data files are in `datasets/inverse_scaling/data/`. Load using standard JSONL readers.

## Dataset 2: ARC-AGI (Abstraction and Reasoning Corpus)

### Overview
- **Source**: [François Chollet's GitHub](https://github.com/fchollet/ARC)
- **Location**: `datasets/ARC/ARC-AGI-master/data/`
- **Format**: JSON
- **Task**: Few-shot grid-based reasoning/pattern completion.
- **Splits**: Training (400 tasks), Evaluation (400 tasks).

### Loading
Files are JSON objects containing `train` and `test` examples for each task.
```python
import json
import os

task_file = "datasets/ARC/ARC-AGI-master/data/training/007bbfb7.json"
with open(task_file, 'r') as f:
    task = json.load(f)
```
