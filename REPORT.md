# Research Report: Investigation into Finetuning-Resistant Datasets

## 1. Executive Summary
We investigated whether the "Inverse Scaling" dataset, specifically the `redefine-math` task, is resistant to model finetuning. While prior work suggests these tasks resist performance gains from model scaling, our experiments with `TinyLlama-1.1B` demonstrate that **Low-Rank Adaptation (LoRA) finetuning significantly improves performance (46% → 80%)**, outperforming 5-shot in-context learning (57%). We conclude that while these tasks are resistant to zero-shot and few-shot scaling, they are not "finetuning proof" given sufficient supervision on the specific instruction.

## 2. Goal
The primary objective was to verify the existence of "finetuning proof" datasets—tasks where providing gradient-based updates on training examples fails to improve performance on held-out test data. This is important for defining the limits of current LLM adaptability and understanding the strength of pre-trained priors versus explicit instructions.

## 3. Data Construction

### Dataset Description
*   **Source**: Inverse Scaling Prize (`redefine-math` task).
*   **Task**: The model must ignore standard mathematical operations and instead operate on the string representation of numbers (e.g., "first digit of 50+47" → "5" from "50", not "9" from "97").
*   **Size**: 1,244 examples.
*   **Split**: 80% Train (995), 20% Test (249).

### Example Sample
```json
{
  "prompt": "View number as text. Do not perform computation. Q: What is the first digit of 50+47? A:",
  "classes": "[' 9', ' 5']",
  "correct_answer": " 5"
}
```
*Prediction*: The model must suppress the arithmetic prior (97 -> 9) and select the textual prior (50 -> 5).

## 4. Experiment Description

### Methodology
We compared three adaptation methods:
1.  **Zero-Shot**: Standard inference.
2.  **Few-Shot (5-shot)**: In-context learning with 5 examples.
3.  **Parameter-Efficient Finetuning (LoRA)**: Training adapter layers on the training set.

### Implementation Details
*   **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (float16).
*   **Finetuning**: LoRA (r=8, alpha=32), 3 epochs, batch size 4.
*   **Metric**: Log-likelihood based choice selection (Robust Evaluation).

### Experimental Protocol
*   We evaluated the probability of the full sequence `prompt + choice` for each candidate.
*   Accuracy was measured on the held-out test set (stratified split).

## 5. Result Analysis

### Key Findings
| Method | Accuracy | Improvement |
|--------|----------|-------------|
| Zero-Shot | 46.18% | - |
| Few-Shot (5-shot) | 57.43% | +11.25% |
| **Finetuning (LoRA)** | **80.32%** | **+34.14%** |

### Hypothesis Testing Results
*   **Hypothesis**: The dataset is finetuning resistant (FT ≈ Zero-Shot).
*   **Result**: **REJECTED**. Finetuning yielded a massive improvement.
*   **Interpretation**: The model successfully "unlearned" the math prior and learned the "first digit" textual heuristic.

### Comparison to Baselines
*   **Few-Shot Resistance**: The gap between Zero-Shot and Few-Shot was modest (+11%), confirming that the strong prior is hard to overcome with context alone.
*   **Finetuning Efficacy**: Gradient updates were effective where context was insufficient.

### Visualizations
*(See `results/redefine_math_robust.json` for raw data)*
The training loss consistently decreased from 3.36 to ~0.40, indicating strong convergence.

## 6. Conclusions
Our results challenge the notion that Inverse Scaling tasks are inherently "finetuning proof." They are likely only "scaling proof" (resistant to zero-shot scaling). When explicitly trained on the counter-intuitive rule, models—even small ones like TinyLlama—can adapt successfully. This suggests that "resistance" is a function of the *method* of adaptation (ICL vs. FT) rather than an immutable property of the data.

## 7. Next Steps
*   **Generalization Test**: Train on `redefine-math` and test on `redefine-logic` (if it exists) to see if the "suppress prior" meta-skill transfers.
*   **ARC-AGI**: Attempt the same on ARC, which is resistant due to task disjointness rather than prior strength.
