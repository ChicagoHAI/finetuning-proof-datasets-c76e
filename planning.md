# Research Plan: Finetuning-Resistant Datasets

## Research Question
Are there specific datasets ("finetuning proof") where standard model finetuning fails to improve performance, or even degrades it, compared to in-context learning or zero-shot baselines?

## Background and Motivation
Standard scaling laws suggest that more data and compute (training) lead to better performance. However, recent work (Inverse Scaling, ARC) suggests certain domains—specifically those requiring robust reasoning against strong priors—resist this trend. Identifying these datasets is crucial for understanding the limitations of current LLM paradigms.

## Hypothesis Decomposition
1.  **Priors vs. Instructions**: On tasks where the prompt redefines common concepts (e.g., `redefine-math`), finetuning on a small set of examples will fail to override pre-trained priors, potentially leading to overfitting on surface features rather than the rule change.
2.  **Generalization Gap**: On abstract reasoning tasks (ARC), finetuning on training tasks will not generalize to test tasks due to disjoint underlying logic rules.

## Proposed Methodology

### Approach
We will focus on empirical testing using the **Inverse Scaling Prize** dataset (`redefine-math`) and **ARC-AGI**. We will use a small but capable Language Model (e.g., `TinyLlama-1.1B` or `GPT-Neo-125M`) to allow for rapid iteration.

### Experimental Steps
1.  **Inverse Scaling (Redefine-Math)**:
    *   Load `redefine-math` dataset.
    *   Evaluate Zero-shot and Few-shot (5-shot) baseline performance.
    *   **Fine-tune** the model on a training split of `redefine-math`.
    *   Evaluate on the test split.
    *   *Success Criteria*: If FT accuracy <= Zero/Few-shot or does not improve significantly, the dataset is "resistant".

2.  **ARC-AGI (Text Format)**:
    *   Convert ARC grids to text representation (JSON).
    *   Fine-tune on a set of ARC training tasks.
    *   Evaluate on held-out *evaluation* tasks (not the test set of the same tasks, but entirely new tasks).
    *   *Success Criteria*: Expect near-zero improvement on new tasks, confirming lack of meta-learning.

### Baselines
*   **Zero-Shot**: The raw model's ability.
*   **Few-Shot (In-Context)**: The model's ability to adapt without weight updates.
*   *Comparison*: FT vs. ICL.

### Evaluation Metrics
*   **Exact Match Accuracy**: For math and ARC outputs.
*   **Log-Likelihood**: To see if the model becomes more confident in the *wrong* (prior-driven) answer.

### Statistical Analysis Plan
*   Compare mean accuracy across 3 seeds.
*   Report standard deviation.

## Expected Outcomes
*   **Inverse Scaling**: FT might improve slightly but struggle to reach high accuracy, or might degrade if it overfits to specific examples rather than the rule change.
*   **ARC**: FT will likely fail completely on unseen tasks.

## Timeline and Milestones
*   **Setup**: 10 min
*   **Implementation**: 30 min (Data loading, Training script)
*   **Experiments**: 60 min (Running FT on GPU)
*   **Analysis**: 20 min
*   **Reporting**: 20 min

## Potential Challenges
*   **Compute**: Finetuning might be slow. *Mitigation*: Use LoRA (Low-Rank Adaptation) for speed and efficiency, or small models.
*   **Format**: ARC is visual. *Mitigation*: Use standard JSON text representation.

## Success Criteria
Demonstrating a clear gap between Training Loss (which should go down) and Validation Accuracy on held-out *tasks* (which should stay flat or degrade), proving resistance to generalization.
