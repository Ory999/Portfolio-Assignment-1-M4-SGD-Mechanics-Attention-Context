# Portfolio-Assignment-1-M4-SGD-Mechanics-Attention-Context

This repository contains the solution for the "M4: Applied DL and AI" assignment. It demonstrates a manual implementation of Stochastic Gradient Descent (SGD) and a proof-of-concept for the Self-Attention Mechanism using Python.

---

## Part A: Manual Stochastic Gradient Descent (SGD)

### Methodology
1.  **Data:** Loaded the dataset and scaled inputs/outputs to a `0-1` range using `MinMaxScaler`.
2.  **Scope:** Processed the first 3 samples manually to visualize weight updates.
3.  **Hyperparameters:**
    * **Learning Rate ($\alpha$):** `0.1`
    * **Initial Weight ($w$):** `0.5`
4.  **Formulas Used:**
    * Prediction: $\hat{y} = x \cdot w$
    * Loss Gradient: $\frac{\partial L}{\partial w} = 2x(\hat{y} - t)$
    * Update Rule: $w_{new} = w_{old} - \alpha \cdot \frac{\partial L}{\partial w}$

### Results
The model successfully updated weights sample-by-sample, demonstrating the "stochastic" nature of SGD where the gradient estimate fluctuates based on the individual data point being processed.

---

## Part B: Attention Contextualization

The goal of this section was to demonstrate how the Self-Attention Mechanism can differentiate the meaning of a homonym based on its context words.

### The Experiment
* **Target Homonym:** **"Bat"**
* **Context 1 (Animal):** *"the bat flew away from the cave"*
* **Context 2 (Object):** *"the heavy bat is made of wood"*

### Embedding Strategy ("The L-Shape")
To prove the shift in meaning, I manually initialized 2D word embeddings using an orthogonal strategy:
* **"Bat" (Neutral):** Initialized at `[0.5, 0.5]`.
* **Animal Context Words:** Assigned **High-Y / Low-X** values (e.g., `[0.01, 0.99]`) to pull the vector vertically.
* **Object Context Words:** Assigned **High-X / Low-Y** values (e.g., `[0.99, 0.01]`) to pull the vector horizontally.

### Results
After applying the Self-Attention formula $\text{softmax}(QK^T)V$:
1.  **Sentence 1 "Bat":** Shifted towards the vertical axis.
2.  **Sentence 2 "Bat":** Shifted towards the horizontal axis.
3.  **Cosine Similarity:** **`0.3882`** difference from intitial cosine similarity of 0.9162

---

## How to Run
1.  Clone this repository.
2.  Install the required libraries.
4.  Run all cells to verify the calculations.

