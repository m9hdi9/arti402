# ARTI 402 — Deep Learning
## Lab 5: Optimizers (SGD, Mini-batches, Momentum, RMSProp, and Adam)

This repository contains the complete implementation, theoretical analysis, and experimental evaluations for **Lab 5** of the ARTI 402 (Deep Learning) course[cite: 3]. The lab opens the backpropagation "black box", replaces slow numerical approximations with exact analytic gradients, and explores how optimization algorithms navigate complex loss landscapes[cite: 3].

---

## 📌 Overview & Core Concepts

* **Analytic Backpropagation**: Replaced finite-difference numerical gradients (`numerical_gradient`) with full backward passes across layers using the chain rule, achieving an approximate $\sim 270\times$ speed-up per optimization step[cite: 3].
* **The Iteration vs. Epoch Trade-off**: An epoch is a full pass through the dataset, while an iteration is a single weight update[cite: 3]. Mini-batching bridges the gap between noisy single-sample updates and slow full-batch calculations[cite: 3].
* **The Ravine / Zig-zag Problem**: High-curvature loss surfaces (narrow valleys) cause vanilla SGD to oscillate wildly across steep dimensions while making sluggish progress along the gentle valley floor[cite: 3].
* **Adaptive & Directional Optimizers**:
  * **Momentum**: Tracks velocity/direction to cancel out cross-wall oscillations and accelerate along consistent gradients[cite: 3].
  * **RMSProp**: Tracks squared gradient magnitudes per parameter to dynamically scale individual learning rates (dampening steep directions and speeding up flat ones)[cite: 3].
  * **Adam**: Synthesizes Momentum (first moment) and RMSProp (second moment) with start-up bias corrections to prevent vanishing steps at initialization[cite: 3].

---

## 🛠️ Implemented Components

| Component | Class / Function | Description |
|---|---|---|
| **Two-Layer MLP** | `SpiralNet` | Implemented full forward and backward passes (`Layer_Dense` $\rightarrow$ `ReLU` $\rightarrow$ `Layer_Dense` $\rightarrow$ `Softmax_Loss`)[cite: 3]. |
| **Vanilla SGD** | `Optimizer_SGD` | Standard gradient descent step: $\theta \leftarrow \theta - \alpha \cdot \nabla_\theta L$[cite: 3]. |
| **Mini-batch Sampler** | `get_batches(X, y, batch_size, rng)` | Shuffles samples per epoch and partitions the dataset into uniform mini-batches[cite: 3]. |
| **SGD with Momentum** | `Optimizer_SGD_Momentum` | Velocity-based update: $v \leftarrow \beta \cdot v - \alpha \cdot \nabla_\theta L$, $\theta \leftarrow \theta + v$[cite: 3]. |
| **RMSProp** | `Optimizer_RMSprop` | Per-parameter scaling using moving average of squared gradients (`cache`)[cite: 3]. |
| **Adam Optimizer** | `Optimizer_Adam` | Combines bias-corrected first and second moments for adaptive, robust steps[cite: 3]. |

---

## 🔬 Experimental Results & Key Findings

### 1. Mini-Batch Size Benchmark (1,000 Epochs)
Evaluating batch sizes on the 300-sample non-linear spiral dataset using vanilla SGD ($\alpha = 1.0$)[cite: 3]:

* **Full Batch (300 samples)**: 1,001 steps $\rightarrow$ Accuracy: **42.7%** (too few updates)[cite: 3].
* **Mini-batch (32 samples)**: 10,010 steps $\rightarrow$ Accuracy: **77.7%** (optimal trade-off between gradient quality and update frequency)[cite: 3].
* **Mini-batch (8 samples)**: 38,038 steps $\rightarrow$ Accuracy: **54.3%** (high gradient noise degrades convergence, high wall-clock latency)[cite: 3].

### 2. The 10,000-Epoch Optimizer Race (Assessment Q1)
Training a full-batch `SpiralNet` across five optimizer setups on the 3-arm spiral task[cite: 3]:

| Optimizer | Train Accuracy | Train Loss | Test Accuracy | Convergence Behavior |
|---|---|---|---|---|
| **SGD** | 73.0% | 0.628 | 66.3% | Struggles to bend decision boundaries[cite: 3]. |
| **SGD + Decay** | 72.0% | 0.729 | 56.0% | Prematurely slows step size down[cite: 3]. |
| **SGD + Momentum** | 96.3% | 0.102 | 78.3% | Smooth, rapid convergence down the loss valley[cite: 3]. |
| **RMSProp** | 90.3% | 0.244 | 81.3% | Equalizes step sizes across non-uniform gradients[cite: 3]. |
| **Adam** | **96.7%** | **0.089** | **82.0%** | Fastest convergence and highest overall accuracy[cite: 3]. |

*Note on Generalization*: The best models reached $\sim 96\%$ training accuracy but saturated at $\sim 82\%$ test accuracy, showing slight overfitting on complex boundary curves[cite: 3].

### 3. Hyperparameter Sensitivity: Learning Rate in Adam (Assessment Q2)
Training Adam for 2,000 epochs across varying learning rates ($\alpha$)[cite: 3]:

* **$\alpha = 0.0005$ (Too Small)**: Train Acc: **67.0%** — Learns slowly, unable to reach the minimum within 2,000 epochs[cite: 3].
* **$\alpha = 0.05$ (Optimal)**: Train Acc: **95.7%** — Smooth, robust convergence[cite: 3].
* **$\alpha = 2.0$ (Too Large)**: Train Acc: **34.0%** (Worst Loss: **10.45**) — Diverges violently, overshoots the valley, and destabilizes parameters[cite: 3].

---

## 📂 Repository Structure

```text
arti402/
│
├── arti402_Lab5_<YourID>.ipynb   # Main notebook with code, outputs, and assessments[cite: 3]
├── arti402_figures.py            # Diagram, loss curve, and decision boundary visualizers[cite: 3, 4]
└── README.md                     # Comprehensive project documentation
