# Resources Catalog

### Summary
This document catalogs resources for investigating finetuning-resistant datasets.

### Papers
Total papers downloaded: 4

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Inverse Scaling | McKenzie et al. | 2023 | papers/2306.09479... | Defines tasks where scaling/training hurts. |
| Measure of Intelligence | Chollet | 2019 | papers/1911.01547... | Introduces ARC, the gold standard for FT-resistance. |
| FT Distorts Features | Kumar et al. | 2022 | papers/2202.10054... | Explains OOD failure of FT. |
| Curse of Recursion | Shumailov | 2023 | papers/2305.17493... | Model collapse on synthetic data. |

### Datasets
Total datasets downloaded: 2 collections

| Name | Source | Format | Location | Notes |
|------|--------|--------|----------|-------|
| Inverse Scaling | GitHub/HF | JSONL | `datasets/inverse_scaling/` | Contains `redefine-math`, `memo-trap`, etc. |
| ARC-AGI | GitHub | JSON | `datasets/ARC/` | 800 grid reasoning tasks. |

### Code Repositories
Total repositories cloned: 2

| Name | Purpose | Location |
|------|---------|----------|
| inverse_scaling_prize | Data & Eval code | `code/inverse_scaling_prize/` |
| ARC_repository | Data & Utils | `code/ARC_repository/` |

### Resource Gathering Notes
- **Search Strategy**: Focused on terms like "inverse scaling", "finetuning failure", and specific benchmarks known for difficulty (ARC).
- **Selection**: Chosen datasets are publicly available, well-documented, and have academic backing for their difficulty.
- **Challenges**: Many "resistant" datasets are just poor quality. We selected ones that are *intentionally* difficult due to reasoning requirements.

### Recommendations
1.  **Start with Inverse Scaling tasks** (`redefine-math`). They are standard text-generation tasks and easy to run with HuggingFace Transformers.
2.  **Evaluate with Loss vs. Accuracy**. See if FT decreases loss (better prediction) but decreases accuracy (worse task performance), indicating the model is optimizing the wrong thing (e.g., formatting).
