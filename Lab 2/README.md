# ARTI 402: Deep Learning — Lab 2
**Activations, Loss, and How a Network Learns**

This repository contains the implementation and solutions for **Lab 2** of the ARTI 402 (Deep Learning) course. The lab covers the foundational building blocks of deep neural networks: non-linear activation functions, probability mapping, loss estimation, and gradient-based optimization from scratch using NumPy.

---

## 📌 Objectives & Covered Topics

1. **Linear Layer Collapse Proof:** Numerically proving that stacking pure linear layers collapses mathematically into a single equivalent matrix ($W_{eq} = W_2 W_1$).
2. **Activation Functions:** Implementing vectorized activation functions to introduce non-linearity:
   - **ReLU:** `max(0, x)`
   - **Sigmoid:** $\frac{1}{1 + e^{-x}}$
   - **Softmax:** Transforming raw logits into normalized class probabilities with numerical overflow protection (`x - max(x)`).
3. **Modular Dense Layer (`Layer_Dense`):** Building a reusable feedforward layer supporting transposed weight initialization `(n_inputs, n_neurons)` and broadcasting biases `(1, n_neurons)`.
4. **Loss Function:** Implementing **Categorical Cross-Entropy** loss with numerical clipping (`np.clip`) to penalize incorrect confidences smoothly.
5. **Numerical Differentiation & Optimization:**
   - Computing numerical derivatives using the central difference formula:
     $$\frac{f(x + h) - f(x - h)}{2h}$$
   - Implementing basic **Gradient Descent** parameter updates.
6. **Backpropagation & Chain Rule:** Deriving and computing gradients analytically (`dw`, `dx`, `db`) through a single ReLU neuron.
7. **End-to-End Neural Network Training (Assessment):** Assembling a 2-layer neural network trained on a 3-class toy dataset via numerical gradient descent.

---

## 📂 Project Structure

```text
├── arti402_Lab2_<YourID>.ipynb   # Completed Jupyter Notebook with all outputs
└── README.md                     # Lab documentation and summary
