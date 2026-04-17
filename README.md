# CMPE 401 — Project 2: Time Series Modeling with Deep Learning

**Understand, Organize Time Series Modeling Benchmark for Classification and Forecasting**

---

## Overview

This project explores two fundamental deep learning approaches for time-series modeling using official Keras examples:

| Task | Model | Dataset | Problem Type | Result |
|------|-------|---------|--------------|--------|
| Classification | Transformer | FordA | Binary fault detection | **86.59% accuracy** |
| Forecasting | LSTM | Jena Climate | Temperature prediction (12h ahead) | **Best Val MSE: 0.0949** |

---

## Repository Structure

```
cmpe401-project2/
├── transformer.ipynb                         # Task 1A — Transformer classification (FordA)
├── lstm_forecasting_final.ipynb              # Task 1B — LSTM baseline reproduction
├── lstm_forecasting_improvements.ipynb       # Task 2 — Three LSTM improvements
└── README.md
```

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

**Results:**

| Metric | Value |
|--------|-------|
| Test Accuracy | **86.59%** |
| Val Accuracy | ~85% |
| Optimizer | Adam (lr=1e-4) |
| Epochs | 150 |

> **Note on reproducibility:** The official Keras Transformer example contains a bug — `GlobalAveragePooling1D(data_format="channels_last")` collapses the wrong axis in Keras 3, outputting shape `(None, 1)` instead of `(None, 500)`, causing the model to produce random predictions (~51% accuracy). This is a confirmed upstream bug documented in keras-io [Issue #1908](https://github.com/keras-team/keras-io/issues/1908) and [Issue #20627](https://github.com/keras-team/keras/issues/20627). The fix is changing `data_format="channels_last"` to `data_format="channels_first"`, which correctly averages across the 500 timesteps and restores the expected model behavior and accuracy.

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

---

## Task 2 — Improvements (LSTM Forecasting)

Three controlled modifications applied to the LSTM baseline. All other settings held constant (epochs=10, optimizer=Adam, lr=0.001).

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

- **Imp1 (Larger LSTM):** Marginally worse. Larger model overfits faster — early stopping at epoch 6 vs 10 for baseline. Extra capacity doesn't have time to generalize with only 10 epochs.
- **Imp2 (Stacked LSTM):** Worst result. Depth adds complexity without enough regularization, hurting generalization on structured weather data.
- **Imp3 (Longer Sequence):** Best result — the only improvement that beat baseline. Confirms that **input representation matters more than model capacity** for weather forecasting. Multi-day patterns require more historical context.

**Key Insight:** For structured time-series forecasting, improving the input (more context) is more effective than improving the model (more parameters or layers).

---

## Task 3 — Analysis & Observations

### LSTM Loss Curve Analysis (Baseline)
- Training MSE decreases from 0.1184 → 0.0610 over 10 epochs — model is actively learning
- Validation MSE lowest at epoch 1 (0.0973) then increases — mild overfitting
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

The LSTM was significantly easier to understand. It processes sequences step-by-step, maintaining a hidden state that carries information forward — this mirrors how we intuitively think about sequential data. The gating mechanisms (input, forget, output gates) have a clear functional interpretation.

The Transformer operates on all timesteps simultaneously via self-attention, which is more abstract. It also proved more sensitive to the software environment — a bug in the official Keras example's `GlobalAveragePooling1D` layer (using wrong `data_format` argument) caused the model to silently fail across multiple platforms until the bug was identified and patched. The LSTM ran correctly on the first attempt.

**Q2: What improvement did you try, and what did you learn from it?**

Three improvements were run on the LSTM forecasting model with actual training results:

1. **Larger hidden size (32→64):** Marginally worse (Val MSE 0.0978 vs 0.0973). More capacity caused faster overfitting — early stopping at epoch 6 instead of 10. Learning: bigger isn't always better with short training budgets.

2. **Stacked LSTM (1→2 layers):** Worst result (Val MSE 0.1012). Depth adds complexity without enough regularization. Learning: deep LSTMs need dropout or more epochs to be beneficial.

3. **Longer look-back window (5→10 days):** Best result (Val MSE 0.0949) — the only improvement that beat baseline. Learning: for weather forecasting, input representation matters more than model architecture.

**Overall learning:** For structured time-series forecasting, improving the input (more temporal context) is more effective than improving the model (more parameters or layers).

---

## How to Run

1. Open `transformer.ipynb` in Google Colab for Transformer classification
2. Open `lstm_forecasting_final.ipynb` for LSTM baseline
3. Open `lstm_forecasting_improvements.ipynb` for the 3 improvements
4. Runtime → Change runtime type → GPU (T4 or better)
5. Run All cells

**Official Keras References:**
- [Transformer Classification Example](https://keras.io/examples/timeseries/timeseries_classification_transformer/)
- [LSTM Forecasting Example](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)

**Bug Fix Reference:**
- [keras-io Issue #1908](https://github.com/keras-team/keras-io/issues/1908)
- [keras Issue #20627](https://github.com/keras-team/keras/issues/20627)

---

## References

- Keras Time Series Examples — https://keras.io/examples/timeseries/
- FordA Dataset — UCR Time Series Archive
- Jena Climate Dataset — Max Planck Institute for Biogeochemistry
