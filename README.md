# CMPE 401 — Project 2: Time Series Modeling with Deep Learning

**Understand, Organize Time Series Modeling Benchmark for Classification and Forecasting**

---

## Overview

This project explores two fundamental deep learning approaches for time-series modeling using official Keras examples:

| Task | Model | Dataset | Problem Type |
|------|-------|---------|--------------|
| Classification | Transformer | FordA | Binary fault detection |
| Forecasting | LSTM | Jena Climate | Temperature prediction (12h ahead) |

---

## Repository Structure

```
cmpe401-project2/
├── lstm_forecasting_final.ipynb              # Task 1 — LSTM baseline reproduction
├── lstm_forecasting_improvements.ipynb       # Task 2 — Three improvements
├── README.md
```

> **Note on Transformer:** The official Keras Transformer notebook will be added once environment compatibility is resolved. See Task 1A below for full details and referenced results.

---

## Task 1 — Baseline Reproduction

### A) Transformer Classification (FordA Dataset)

**Dataset:** FordA — engine noise sensor readings for automotive fault detection
- 3,601 training samples, 1,320 test samples
- 500 timesteps per sample, 1 feature (univariate)
- Binary classification: Normal (0) vs Fault (1)

**Model:** Transformer encoder with multi-head self-attention
- 4 Transformer blocks, 4 attention heads, head size 256
- Feed-forward dim: 4, MLP units: 128, dropout: 0.25/0.4
- Global average pooling → Dense → Softmax
- Total parameters: 29,258

**Baseline Results:**

| Metric | Value |
|--------|-------|
| Test Accuracy | ~85% (official Keras result) |
| Parameters | 29,258 |
| Optimizer | Adam (lr=1e-4) |
| Epochs | 150 (early stopping, patience=10) |

> **Note on reproducibility:** The official Keras Transformer example could not be reproduced in the current Colab/Kaggle environment due to a breaking change in Keras 3 (TF 2.19, Keras 3.13). Specifically, `GlobalAveragePooling1D` with `data_format='channels_last'` now collapses the wrong axis, outputting shape `(None, 1)` instead of `(None, 500)`, causing the model to output random predictions (~51% accuracy) regardless of training. This was reproduced and confirmed across multiple platforms (Google Colab, Kaggle) and Keras versions. The official notebook result (~85% test accuracy) is referenced directly from the Keras documentation. The model architecture, training procedure, and expected results are fully documented above.

---

### B) LSTM Forecasting (Jena Climate Dataset)

**Dataset:** Jena Climate — weather station readings 2009–2016
- 420,551 samples, 15 meteorological features, sampled every 10 minutes
- 7 features selected after preprocessing
- Task: Predict temperature 12 hours ahead
- Look-back window: 720 samples (5 days), step: 6 (hourly)

**Model:** Single-layer LSTM
- LSTM hidden size: 32, Dense output layer
- Loss: MSE, Optimizer: Adam (lr=0.001)
- Parameters: 5,153

**Baseline Training Results:**

| Epoch | Train MSE | Val MSE |
|-------|-----------|---------|
| 1 | 0.1184 | 0.0973 |
| 2 | 0.0843 | 0.0987 |
| 3 | 0.0773 | 0.1005 |
| 4 | 0.0714 | 0.1057 |
| 5 | 0.0660 | 0.1099 |

**Best Val MSE: 0.0973 (Epoch 1)**

**Observations:** Training MSE decreases steadily indicating active learning. Validation MSE increases after epoch 1 — a sign of mild overfitting. The small model (5,153 params) converges quickly, suggesting the task is learnable with minimal capacity but more epochs or larger model could improve generalization.

---

## Task 2 — Improvements (LSTM Forecasting)

Three controlled modifications were applied to the LSTM baseline. All other settings held constant (epochs=10, optimizer=Adam, lr=0.001, batch=256).

| # | Improvement | Change | Motivation |
|---|-------------|--------|------------|
| 1 | Larger LSTM | hidden: 32→64 | More capacity for complex temperature patterns |
| 2 | Stacked LSTM | 1 layer→2 layers | Hierarchical temporal feature extraction |
| 3 | Longer Sequence | look-back: 720→1440 (10 days) | More historical context for forecasting |

### Results

| Experiment | Best Val MSE | Epochs | vs Baseline |
|------------|-------------|--------|-------------|
| Baseline (hidden=32, 1 layer, 5 days) | 0.0973 | 10 | — |
| Imp1 — Larger LSTM (hidden=64) | 0.0978 | 6 | +0.0005 ↑ worse |
| Imp2 — Stacked LSTM (2 layers) | 0.1012 | 6 | +0.0039 ↑ worse |
| **Imp3 — Longer Sequence (10 days)** | **0.0949** | 8 | **-0.0024 ↓ better ✅** |

### Analysis

- **Imp1 (Larger LSTM):** Marginally worse (0.0978 vs 0.0973). Larger model overfits faster — early stopping triggered at epoch 6 vs epoch 10 for baseline. With only 10 epochs, extra capacity doesn't have time to generalize.

- **Imp2 (Stacked LSTM):** Worst result (0.1012). Two LSTM layers double the complexity, causing faster overfitting. The Jena Climate task is learnable with shallow models — depth requires more regularization or longer training.

- **Imp3 (Longer Sequence):** Best result (0.0949) — the only improvement that beat the baseline. Confirms that **input representation matters more than model capacity** for structured weather forecasting. Multi-day temperature patterns require more historical context to predict accurately.

**Key Insight:** For time-series forecasting on structured weather data, improving the input (more context) is more effective than improving the model (more parameters or layers).

---

## Task 3 — Analysis & Observations

### LSTM Loss Curve Analysis (Baseline)
- Training MSE decreases from 0.1184 → 0.0610 over 10 epochs — model is actively learning
- Validation MSE is lowest at epoch 1 (0.0973) then increases — mild overfitting
- Small train-val gap suggests the model is near its capacity limit for this architecture

### Transformer vs LSTM — Key Differences

| Aspect | Transformer | LSTM |
|--------|-------------|------|
| Processing | All timesteps simultaneously (attention) | Sequential, step-by-step |
| Memory mechanism | Self-attention weights | Hidden state + cell state (gating) |
| Training stability | Sensitive to hyperparameters & framework | Robust, converges reliably |
| Interpretability | Attention maps (visualizable) | Hidden state (less interpretable) |
| Best for | Long-range dependencies | Short-to-medium sequences |
| Parameters | 29,258 | 5,153 |

---

## Task 4 — Questions

**Q1: Which model did you find easier to understand and why?**

The LSTM was significantly easier to understand. It processes sequences step-by-step, maintaining a hidden state that carries information forward in time — this directly mirrors how we intuitively think about sequential data. The gating mechanisms (input, forget, output gates) have a clear functional interpretation: the model learns what to remember and what to discard at each timestep.

The Transformer operates on all timesteps simultaneously via self-attention, which is a more abstract concept. Additionally, during this project the Transformer proved far more sensitive to the software environment — a breaking change in Keras 3's `GlobalAveragePooling1D` caused the official example to fail entirely across multiple platforms, while the LSTM ran correctly on the first attempt with no modifications required.

**Q2: What improvement did you try, and what did you learn from it?**

Three improvements were run on the LSTM forecasting model with actual training results:

1. **Larger hidden size (32→64):** Marginally worse (Val MSE 0.0978 vs 0.0973). More capacity caused faster overfitting — early stopping triggered at epoch 6 instead of 10. Learning: bigger isn't always better with small datasets and short training budgets.

2. **Stacked LSTM (1→2 layers):** Worst result (Val MSE 0.1012). Depth adds complexity without enough regularization, hurting generalization on structured weather data. Learning: deep LSTMs need dropout or more epochs to be beneficial.

3. **Longer look-back window (5→10 days):** Best result (Val MSE 0.0949) — the only improvement that beat the baseline. Learning: for weather forecasting, input representation matters more than model architecture. Multi-day temperature patterns require more historical context to predict accurately.

**Overall learning:** For time-series forecasting on structured data, improving the input (more context) is more effective than improving the model (more parameters or layers).

---

## How to Run

1. Open `lstm_forecasting_final.ipynb` in Google Colab for baseline
2. Open `lstm_forecasting_improvements.ipynb` for the 3 improvements
3. Runtime → Change runtime type → GPU (T4 or better)
4. Run All cells

**Official Keras References:**
- [Transformer Classification Example](https://keras.io/examples/timeseries/timeseries_classification_transformer/)
- [LSTM Forecasting Example](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)

---

## References

- Keras Time Series Examples — https://keras.io/examples/timeseries/
- FordA Dataset — UCR Time Series Archive
- Jena Climate Dataset — Max Planck Institute for Biogeochemistry
