# AUEP-L — Adaptive Early-Exit Language Model for Real-Time Autocomplete

AUEP-L is a research-oriented **early-exit language model for real-time text autocomplete**. The project explores how a multi-layer Transformer can dynamically stop computation at an intermediate layer when the model is sufficiently confident about its prediction, reducing unnecessary computation while maintaining prediction quality.

The notebook implements a complete experimental pipeline covering model training, confidence estimation, adaptive early exiting, statistical calibration, ablation analysis, struggle-aware adaptation, evaluation, and a live next-word demonstration.

---

## 📌 Project Overview

Traditional autoregressive language models process every input through all Transformer layers before producing a prediction.

AUEP-L investigates an alternative approach:

```text
Input Text
    ↓
Transformer Layer 1
    ↓
Transformer Layer 2
    ↓
Confidence Check
    ↓
 ┌───────────────┐
 │ Confident?    │
 └───────┬───────┘
         │
     Yes │ No
         │
         ↓
   Early Exit     Continue
                      ↓
                Next Transformer Layer
                      ↓
                Confidence Check
                      ↓
                   Output
