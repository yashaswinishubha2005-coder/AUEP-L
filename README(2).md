# AUEP-L — Adaptive Early-Exit Language Model for Real-Time Autocomplete

AUEP-L is a research-oriented project that explores **adaptive early-exit inference for Transformer-based real-time text autocomplete**.

Instead of forcing every prediction to pass through all Transformer layers, AUEP-L evaluates intermediate predictions using multiple confidence signals and exits early when the model is sufficiently confident.

The project investigates the trade-off between **prediction quality, inference computation, and response speed**.

---

## 📌 Project Overview

Traditional Transformer language models process an input through the complete network before generating a prediction.

AUEP-L introduces an early-exit mechanism:

```text
Input Text
    ↓
Transformer Layer
    ↓
Intermediate Prediction
    ↓
Confidence Evaluation
    ↓
Is the prediction confident?
    ├── Yes → Early Exit → Prediction
    │
    └── No → Continue to next layer
                    ↓
              Final Prediction
```

The objective is to determine whether an intermediate Transformer layer can already provide a sufficiently reliable prediction, thereby avoiding unnecessary computation in deeper layers.

---

## 🎯 Objectives

The main objectives of AUEP-L are:

- Build a multi-layer Transformer language model.
- Train the model for next-token prediction.
- Generate predictions from intermediate Transformer layers.
- Develop an adaptive early-exit mechanism.
- Use multiple confidence signals for exit decisions.
- Combine confidence signals into a composite score.
- Learn confidence weights from data.
- Calibrate early-exit thresholds statistically.
- Measure computational savings and speedup.
- Evaluate prediction quality at different exit points.
- Perform confidence-signal ablation experiments.
- Analyze the speed-quality trade-off.
- Explore struggle-aware adaptive inference.
- Demonstrate real-time next-word prediction.

---

# 🧠 Core Idea

AUEP-L does not always require the complete Transformer network to produce a prediction.

Instead, the model checks its confidence at intermediate layers.

```text
                     ┌─────────────────────┐
                     │      Input Text     │
                     └──────────┬──────────┘
                                ↓
                           Transformer
                                ↓
                              Layer 1
                                ↓
                              Layer 2
                                ↓
                       Confidence Check
                          ↙          ↘
                   Confident       Uncertain
                      ↓                ↓
                 Early Exit        Next Layer
                      ↓                ↓
                 Prediction       Confidence Check
                                      ↓
                                   Continue
                                      ↓
                                  Final Layer
                                      ↓
                                   Prediction
```

This creates a trade-off between:

- **Inference computation**
- **Inference speed**
- **Prediction quality**

---

# 🔬 Confidence Signals

AUEP-L uses multiple signals to determine whether an intermediate prediction is reliable.

## 1. Confidence

The model's prediction confidence is used as one of the early-exit signals.

Higher confidence indicates that the model strongly prefers its predicted token.

---

## 2. Entropy

Prediction entropy is used to measure uncertainty in the model's output distribution.

Lower entropy indicates a more concentrated probability distribution.

---

## 3. Margin

Margin measures the difference between the highest and second-highest prediction probabilities.

A larger margin indicates stronger separation between the top prediction and the next candidate.

---

## 4. Consistency

Consistency compares predictions across different Transformer layers.

If the model continues to produce the same prediction across layers, this provides additional evidence that the prediction is stable.

---

# 🧮 Composite Confidence

The project combines the confidence signals into a composite confidence score.

The configured initial weights are:

```text
Confidence   = 0.35
Entropy      = 0.25
Margin       = 0.20
Consistency  = 0.20
```

The composite score is conceptually calculated as:

```text
Composite Score =
    w₁ × Confidence
  + w₂ × (1 − Entropy)
  + w₃ × Margin
  + w₄ × Consistency
```

The project also includes a data-driven approach for learning the composite weights rather than relying only on manually selected values.

---

# 🏗️ Model Architecture

AUEP-L uses a multi-layer Transformer language model with intermediate prediction capability.

The architecture supports:

- Multiple Transformer layers
- Intermediate hidden states
- Layer-wise predictions
- Exit classifiers
- Full-model inference
- Early-exit inference
- Key-value cache support during decoding

The intermediate representations are used to determine whether the model should continue processing or exit early.

---

# 📚 Dataset

The project uses the **TinyStories GPT4 validation split** for the experiments.

### Dataset

**TinyStories**

### Dataset Source

https://huggingface.co/datasets/roneneldan/TinyStories

The notebook uses:

```text
TinyStoriesV2-GPT4-valid.txt
```

The text is processed into shorter sentences suitable for next-token prediction and autocomplete experiments.

---

# 🔄 Dataset Preparation

The preprocessing pipeline includes:

1. Story splitting
2. Sentence splitting
3. Whitespace normalization
4. Token-count filtering
5. Vocabulary construction
6. Tokenization
7. Dataset creation

The notebook limits the number of processed lines according to the configured experimental settings.

---

# 🧪 Training Pipeline

The project contains multiple stages for training and evaluation.

## Stage 1 — Self-Supervised Pretraining

The Transformer is trained using a next-token prediction objective.

The model produces predictions at multiple layers:

```text
Input Tokens
     ↓
Transformer
     ↓
Layer 1 → Prediction
Layer 2 → Prediction
Layer 3 → Prediction
...
Final Layer → Prediction
```

Layer-wise losses are combined using configured layer weights so that intermediate layers learn useful predictive representations.

---

# Stage 2a — Exit Classifier Training

Exit classifiers are trained to determine whether an intermediate prediction is sufficiently reliable.

The final-layer prediction is used as the reference during training.

The process can be represented as:

```text
Intermediate Representation
          ↓
      Exit Classifier
          ↓
    Exit / Continue Decision
```

The classifier learns to identify situations where the intermediate prediction agrees with the reference prediction.

---

# Stage 2b — Learning Composite Confidence Weights

AUEP-L also learns data-driven weights for the confidence signals.

The features include:

```text
Confidence
Entropy
Margin
Consistency
```

The intermediate prediction is compared with the reference prediction to construct the target used for learning the composite confidence mechanism.

A logistic-regression-based approach is used for learning the weights.

---

# 📊 Stage 3 — LTT Calibration

The project includes a statistical calibration stage for determining an early-exit threshold.

The configured calibration parameters include:

```text
epsilon = 0.20
delta   = 0.10
tau     = 2.0
```

The calibration stage uses a configured calibration subset and derives an exit threshold that is then used during evaluation.

---

# 📈 Stage 4 — Full Evaluation

The project evaluates multiple early-exit strategies.

The evaluated approaches include:

```text
Softmax
Saturation
Classifier
Composite
```

The evaluation considers:

- Average number of layers used
- Speedup
- F1 against the gold continuation
- F1 against the full-model prediction
- Approximate computation saved

The purpose is to study how different early-exit mechanisms affect both computational efficiency and prediction quality.

---

# 🔍 Ablation Study

An ablation study is performed to investigate the contribution of individual confidence signals.

The following variants are evaluated:

```text
Complete Composite
Without Confidence
Without Entropy
Without Margin
Without Consistency
```

The results can be compared to determine how removing individual signals affects the early-exit mechanism.

---

# ⚖️ Speed–Quality Trade-off

The project performs a lambda sweep to study the relationship between computational savings and prediction quality.

The configured lambda values include:

```text
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
```

For each configuration, the project evaluates:

- Average layers used
- Speedup
- F1 against gold
- F1 against the full-model prediction

The results are used to visualize the speed-quality trade-off.

---

# 👤 Struggle-Aware Evaluation

AUEP-L also explores adaptive inference based on simulated user interaction.

The notebook introduces interaction-related signals such as:

- Engagement
- Posture
- Interaction behavior

These signals are combined into a struggle score.

The project compares:

```text
Flat CALM
        vs.
Struggle-Aware CALM
```

under simulated user conditions such as:

```text
Calm
Struggling
```

The purpose is to investigate whether the inference strategy can adapt its computational behavior according to simulated interaction conditions.

---

# 💻 Live Next-Word Prediction

The notebook includes a next-word prediction demonstration.

Given a text prompt, the system can produce predictions using the Transformer and early-exit mechanism.

The workflow is:

```text
Text Prompt
    ↓
Tokenizer
    ↓
Transformer
    ↓
Intermediate Confidence Evaluation
    ↓
Exit Decision
    ↓
Top-K Predictions
```

The demonstration can compare:

```text
Full-Model Prediction
        vs.
Early-Exit Prediction
```

and report the layer at which an early exit occurs.

---

# 📊 Experimental Outputs

The notebook generates experimental outputs related to:

- Training loss
- Perplexity
- Confidence signals
- Layer-wise behavior
- Early-exit performance
- Ablation experiments
- Lambda sweep
- Speed-quality trade-off
- Struggle-aware evaluation
- Final evaluation summary
- Next-word prediction

The notebook also contains functionality for collecting generated models and graphs into a ZIP archive.

---

# 🗂️ Recommended Project Structure

```text
AUEP-L/
│
├── README.md
│
├── notebook266f9dae5c.ipynb
│
├── models/
│   └── model checkpoints
│
├── plots/
│   ├── training/
│   ├── confidence/
│   ├── ablation/
│   └── evaluation/
│
├── data/
│   └── dataset files
│
├── requirements.txt
│
└── .gitignore
```

Large datasets, virtual environments, generated cache files, and large model checkpoints should generally not be uploaded directly to GitHub.

---

# 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **NumPy**
- **Matplotlib**
- **Hugging Face Datasets**
- **TinyStories**
- **Transformer Architecture**
- **Natural Language Processing**
- **Logistic Regression**
- **Statistical Calibration**
- **Early-Exit Inference**

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yashaswinishubha2005-coder/AUEP-L.git
```

```bash
cd AUEP-L
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

---

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Open:

```text
notebook266f9dae5c.ipynb
```

Run the notebook cells sequentially.

---

# ⚠️ Notes

The notebook downloads and processes the required dataset during execution.

The experiments use the TinyStories validation split according to the configuration in the notebook.

Experimental results can vary depending on:

- Hardware
- Random seed
- Dataset processing
- Training configuration
- Number of training epochs
- Runtime environment

For reproducibility, run the notebook using the provided configuration.

---

# 🔮 Future Work

Potential future extensions include:

- Training on larger datasets
- Evaluating on real autocomplete datasets
- Testing on mobile and edge devices
- Measuring real-world latency
- Measuring energy consumption
- Improving confidence calibration
- Exploring additional early-exit signals
- Using real user interaction data
- Integrating the model into a keyboard application
- Evaluating different Transformer architectures
- Comparing additional early-exit approaches
- Improving robustness across different writing domains

---

# 📌 Research Focus

The central research pipeline is:

```text
Transformer Language Modeling
            ↓
Intermediate Predictions
            ↓
Confidence Estimation
            ↓
Composite Confidence
            ↓
Statistical Calibration
            ↓
Adaptive Early Exit
            ↓
Computational Savings
            ↓
Real-Time Autocomplete
```

The project investigates how adaptive computation can be used to reduce unnecessary Transformer processing while maintaining useful autocomplete predictions.

---

# 👤 Author

**Yashaswini Shubha**

GitHub:

https://github.com/yashaswinishubha2005-coder

Project Repository:

https://github.com/yashaswinishubha2005-coder/AUEP-L

---

# 📜 Project Status

**Research / Experimental Project**

The current repository contains the experimental notebook and implementation used to investigate adaptive early-exit language modeling for real-time autocomplete.

---

# ⭐ Summary

AUEP-L combines:

```text
Transformer Language Modeling
        +
Early-Exit Inference
        +
Confidence Estimation
        +
Composite Confidence
        +
Statistical Calibration
        +
Struggle-Aware Adaptation
        ↓
Efficient Real-Time Autocomplete
```

The project studies whether a language model can dynamically determine when it has enough information to produce a useful prediction without always executing the complete Transformer network.
