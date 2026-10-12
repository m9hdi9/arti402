# ARTI 402 — Deep Learning
## Lab 6: Regularization — Closing the Gap

This repository contains the implementation of core regularization and stabilization techniques from scratch using NumPy to mitigate overfitting on synthetic spiral data.

---

### Contents

- **L1 & L2 Regularization**: Weight penalty calculations and analytical gradients to promote sparsity (L1) and prevent exploding weights (L2).
- **Dropout (Inverted Dropout)**: Training-time random node deactivation with dynamic scaling to maintain expected activation magnitude.
- **Weight Initialization**: Comparison of initialization scales, demonstrating variance preservation with **He initialization** for ReLU networks.
- **Batch Normalization (Forward Pass)**: Per-feature standardization followed by learnable scale (`gamma`) and shift (`beta`) transformations.
- **Model Evaluation & Showdown**: Comparative benchmarking of baseline vs. regularized architectures measuring train/val loss, accuracy, and generalization gaps.

---

### Setup & Requirements

- Python 3.x
- NumPy
- Matplotlib

Ensure `arti402_figures.py` is placed in the same working directory as the notebook to render visualization helpers.

---

### Results Summary

| Model | Technique | Effect Observed |
| :--- | :--- | :--- |
| **Baseline** | None | Severe overfitting; low train loss with rising validation loss. |
| **L2** | Weight decay ($\lambda=5\times 10^{-4}$) | Smoother decision boundaries; smaller generalization gap. |
| **Dropout** | Rate = 0.2 | Breaks co-adaptations; reduces validation error. |
| **L2 + Dropout** | Combined | Robust generalization across unobserved validation splits. |
