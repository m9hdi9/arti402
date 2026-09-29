# ARTI 402 — Deep Learning
## Lab 4: Recurrent Neural Networks (RNN, LSTM, and GRU)

This repository contains the complete implementation and experimental analysis for **Lab 4** of the ARTI 402 (Deep Learning) course[cite: 1]. The lab explores sequential data processing, recurrent architectures, the vanishing gradient problem, and how gated mechanisms preserve long-term memory[cite: 1].

---

## 📌 Overview & Core Concepts

* **Weight Sharing Across Time**: Unlike CNNs which share filters across spatial dimensions, Recurrent Neural Networks (RNNs) share the same cell and weights across all time steps[cite: 1]. This design allows the model to process sequences of arbitrary length with a fixed parameter count[cite: 1].
* **The Vanishing Gradient Problem**: Plain RNNs suffer from exponential signal decay due to repeated matrix multiplications ($W_h$) and non-linear squashing ($tanh$) across time steps, causing early context to be lost[cite: 1].
* **LSTM Architecture**: Long Short-Term Memory solves this decay by introducing an additive **Cell State ($c$)** (the conveyor belt) regulated by three multiplicative gates:
  * **Forget Gate ($f$)**: Controls what percentage of the long-term memory to keep[cite: 1].
  * **Input Gate ($i$) & Candidate ($g$)**: Regulate new information added to the cell state[cite: 1].
  * **Output Gate ($o$)**: Computes the updated short-term memory (**Hidden State $h$**)[cite: 1].

---

## 🛠️ Implemented Components

| Component | Function / Implementation | Description |
|---|---|---|
| **RNN Step** | `rnn_step(x, h_prev, Wx, Wh, b)` | Single-step recurrent update: $h_t = \tanh(x_t W_x + h_{t-1} W_h + b)$[cite: 1]. |
| **RNN Forward** | `rnn_forward(sequence, ...)` | Unrolls the RNN cell sequentially over an input sequence of length $T$[cite: 1]. |
| **Parameter Counting** | `rnn_params`, `gru_params`, `lstm_params` | Computes parameter counts independent of sequence length (RNN: 1 block, GRU: 3 blocks, LSTM: 4 blocks)[cite: 1]. |
| **LSTM Step** | `lstm_step(x, h_prev, c_prev, W)` | Complete gated step updating both hidden state $h$ and cell state $c$[cite: 1]. |
| **Sequence Classifier Head** | `final_hidden_states(...)` | Feature extraction pipeline using the final hidden state to classify sequences[cite: 1]. |

---

## 🔬 Experimental Results & Key Findings

### 1. Synthetic Memory Benchmark (Assessment Q1 & Q2)
A synthetic classification task was designed where class identity is embedded exclusively in **Step 1**, followed by $T-1$ steps of random Gaussian noise[cite: 1]:

* **At $T = 10$**:
  * **RNN Accuracy**: $\approx 29.2\%$ — Drops to random chance ($\approx 33.3\%$ for 3 classes) due to vanishing gradients[cite: 1].
  * **LSTM Accuracy**: $100\%$ — Perfectly preserves the step 1 signal through the noise[cite: 1].
* **Across Sequence Lengths ($T \in [2, 5, 10, 20, 40, 80]$)**:
  * The RNN performance plunges to baseline noise levels after only 5 time steps[cite: 1].
  * The LSTM maintains strong predictive accuracy ($> 85\%$) up to $T = 40$ steps[cite: 1].

### 2. Forget-Gate Bias Analysis (Assessment Q3 & Q4)
The forget gate bias directly controls the baseline retention fraction $f \approx \sigma(\text{bias})$[cite: 1]:

* **Bias $+4.0$ ($f \approx 0.98$)**: Compound retention $0.98^{20} \approx 66.8\% \rightarrow$ Test accuracy $\approx 98.3\%$.
* **Bias $+2.0$ ($f \approx 0.88$)**: Compound retention $0.88^{20} \approx 7.8\% \rightarrow$ Test accuracy drops to $\approx 60.0\%$.
* **Bias $\le 0.0$**: Wipes out the memory completely, dropping model accuracy to random chance.

---

## 📂 Repository Structure

```text
arti402/
│
├── arti402_Lab4_<YourID>.ipynb   # Main notebook with code, outputs, and answers[cite: 1]
├── arti402_figures.py            # Diagram rendering and visualization helpers[cite: 1]
└── README.md                     # Project summary and lab documentation
