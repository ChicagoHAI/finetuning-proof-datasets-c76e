# Literature Review: Finetuning-Resistant Datasets

## Research Area Overview
This review investigates datasets and tasks where standard finetuning methods fail to improve performance, or even degrade it. This phenomenon contradicts the general scaling laws of deep learning, where more data and compute typically lead to better results. The research identifies specific domains—logical reasoning, abstract pattern matching, and tasks requiring robust priors—where pre-trained models struggle to adapt via simple gradient descent on new examples.

## Key Papers

### 1. Inverse Scaling: When Bigger Isn't Better
- **Authors**: McKenzie et al. (2023)
- **Key Contribution**: Identified and released a suite of tasks ("Inverse Scaling Prize") where larger language models perform *worse* than smaller ones.
- **Relevance**: This is the direct evidence of "finetuning resistance" (or at least "scaling resistance"). The tasks typically involve strong deceptive priors (e.g., "Memo Trap") where the model's pre-training bias overpowers the specific instructions or few-shot examples. Finetuning often exacerbates these biases unless carefully designed.

### 2. On the Measure of Intelligence (ARC-AGI)
- **Authors**: François Chollet (2019)
- **Key Contribution**: Proposed the Abstraction and Reasoning Corpus (ARC), a benchmark designed to measure general intelligence (G-factor) rather than skill acquisition.
- **Relevance**: ARC is explicitly designed to be resistant to memorization. While recent methods (like Test-Time Training) show some gains, standard supervised finetuning on ARC tasks generalizes poorly because the test tasks require novel reasoning combinations not seen in training. It represents a "hard" limit for current finetuning paradigms.

### 3. Fine-Tuning can Distort Pretrained Features
- **Authors**: Kumar et al. (2022)
- **Key Contribution**: Theoretical and empirical analysis showing that finetuning moves model weights effectively but can distort the high-quality features learned during pre-training, leading to worse out-of-distribution (OOD) performance.
- **Relevance**: Explains *why* finetuning might fail on robust benchmarks. The "distortion" hypothesis suggests that while in-distribution accuracy goes up, the underlying feature representation becomes brittle, making the model "dumber" on slightly shifted or complex tasks.

### 4. The Curse of Recursion
- **Authors**: Shumailov et al. (2023)
- **Key Contribution**: Demonstrates "model collapse"—a degenerative process where models trained on generated data lose the tails of the distribution.
- **Relevance**: While focusing on generation, it highlights a form of resistance: finetuning on a model's own output (or similar low-entropy data) causes a degradation in diversity and reasoning capability, effectively making the model resistant to improvement via self-play or synthetic data without external grounding.

## Common Characteristics of Resistant Datasets

1.  **Minimal Train-Test Overlap**: Datasets like ARC have disjoint tasks in train/test. Learning the "rules" of the training set provides no direct pattern-matching benefit for the test set; the model must learn the *meta-skill* of rule inference.
2.  **Strong Pre-training Priors**: Tasks in the Inverse Scaling suite (e.g., `quote-repetition`) punish models for relying on the very behaviors (memorization, common n-grams) that pre-training instills. Finetuning often fails to "unlearn" these deep-seated priors without catastrophic forgetting of language syntax.
3.  **Reasoning vs. Pattern Matching**: "Finetuning-proof" tasks often require multi-step logical deduction. Finetuning tends to improve surface-form pattern matching (heuristics) rather than deep reasoning, leading to failure on adversarial or complex instances.

## Gaps and Opportunities

- **True "Proofness"**: No dataset is truly "proof" given enough compute and diverse data, but ARC and Inverse Scaling tasks remain the most resistant.
- **Methodology**: Most successes on these datasets involve *architectural* changes or *inference-time* search (like Chain-of-Thought or DFS), not just simple `model.fit()`. This confirms that the datasets are resistant to the *finetuning method* itself.

## Recommendations for Experimentation

- **Primary Dataset**: **Inverse Scaling Prize** datasets (specifically `redefine-math` or `resisting-correction`). They are lightweight, text-based, and show clear failure modes.
- **Secondary Dataset**: **ARC-AGI**. Use it to test if finetuning helps generalization or just memorization.
- **Baselines**: Compare standard LoRA/Full-Finetuning against In-Context Learning (ICL). Hypothesis: ICL might outperform FT on these "resistant" tasks because it doesn't permanently distort the model weights.
