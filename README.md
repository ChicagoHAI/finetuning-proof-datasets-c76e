# Finetuning Resistance Research

## Overview
This project investigates whether "Inverse Scaling" datasets are resistant to model finetuning. We focused on the `redefine-math` task, where models must ignore mathematical priors to perform textual operations.

## Key Findings
*   **Not Finetuning Proof**: Fine-tuning `TinyLlama-1.1B` improved accuracy from **46% (Zero-shot)** to **80% (LoRA)**.
*   **Stronger than ICL**: Fine-tuning significantly outperformed 5-shot in-context learning (57%).
*   **Conclusion**: Sufficient gradient updates can override strong pre-trained priors.

## Reproducing Results
1.  **Setup**:
    ```bash
    uv pip install torch transformers peft datasets bitsandbytes accelerate scikit-learn matplotlib pandas
    ```
2.  **Run Experiment**:
    ```bash
    python run_finetuning_experiment.py
    ```
    *(Note: This script now includes the robust evaluation logic)*

3.  **Verify**:
    Check `results/redefine_math_robust.json`.

## Structure
*   `run_finetuning_experiment.py`: Main training script.
*   `evaluate_robust.py`: Evaluation script using log-likelihoods.
*   `datasets/`: Inverse scaling data.
*   `results/`: Checkpoints and JSON logs.
