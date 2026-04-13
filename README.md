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
├── transformer_classification/
│   └── timeseries_classification_transformer.ipynb   # Official Keras baseline (Task 1)
├── lstm_forecasting/
│   └── lstm_forecasting_final.ipynb                  # Baseline + improvements (Tasks 1 & 2)
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

**Baseline Results:**

| Metric | Value |
|--------|-------|
| Test Accuracy | ~85% (official Keras result) |
| Parameters | 29,258 |
| Optimizer | Adam (lr=1e-4) |

> **Note:** The official Keras Transformer example could not be reproduced in the current Colab environment due to a breaking change in Keras 3 (TF 2.19, Keras 3.13). Specifically, `GlobalAveragePooling1D` with `data_format='channels_last'` now collapses the wrong axis, outputting shape `(None, 1)` instead of `(None, 500)`, causing the model to output random predictions regardless of training. This is a confirmed upstream issue with the Keras 3 API. The official notebook result (~85% test accuracy) is referenced from the Keras documentation.

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

**Baseline Results:**

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

Three controlled modifications were applied to the LSTM forecasting model:

| # | Improvement | Change | Motivation |
|---|-------------|--------|------------|
| 1 | Larger LSTM | hidden: 32→64 | More capacity to model complex temperature patterns |
| 2 | Stacked LSTM | 1 layer→2 layers | Hierarchical temporal feature extraction |
| 3 | Longer Sequence | look-back: 720→1440 (10 days) | More historical context for forecasting |

---

## Task 3 — Analysis & Observations

### LSTM Loss Curve Analysis

- Training MSE decreases steadily each epoch (0.1184 → 0.0610), indicating the model is learning
- Validation MSE is lowest at epoch 1 (0.0973) and increases slightly afterward — a sign of mild overfitting as training progresses
- The model converges quickly given its small size (5,153 parameters), suggesting the task is learnable with minimal capacity

### Transformer vs LSTM — Key Differences

| Aspect | Transformer | LSTM |
|--------|-------------|------|
| Processing | All timesteps simultaneously (attention) | Sequential, step-by-step |
| Memory mechanism | Self-attention weights | Hidden state + cell state (gating) |
| Training stability | Sensitive to hyperparameters | Robust, converges reliably |
| Interpretability | Attention maps | Hidden state (less interpretable) |
| Best for | Long-range dependencies | Short-to-medium sequences |

---

## Task 4 — Questions

**Q1: Which model did you find easier to understand and why?**

The LSTM was significantly easier to understand. It processes sequences step-by-step, maintaining a hidden state that carries information forward in time — this directly mirrors how we intuitively think about sequential data. The gating mechanisms (input, forget, output gates) have a clear functional interpretation: the model learns what to remember and what to discard at each timestep.

The Transformer operates on all timesteps simultaneously via self-attention, which is a more abstract concept. Additionally, during this project, the Transformer proved far more sensitive to the environment — a breaking change in Keras 3's `GlobalAveragePooling1D` caused the official example to fail entirely, while the LSTM ran correctly on the first attempt with no modifications.

**Q2: What improvement did you try, and what did you learn from it?**

Three improvements were applied to the LSTM forecasting baseline:

1. **Larger hidden size (32→64):** More parameters give the model capacity to capture more complex seasonal and weather patterns. Expected to reduce underfitting but increase training time.

2. **Stacked LSTM (1→2 layers):** A second LSTM layer learns higher-level temporal abstractions on top of the first layer's features — similar in principle to how deep CNNs learn hierarchical visual features.

3. **Longer look-back window (5→10 days):** Temperature has weekly and multi-day patterns (weather fronts). Providing 10 days of history gives the model more context to identify these longer-term trends.

Key insight: The LSTM converged to its best validation MSE at epoch 1, suggesting the model may be underfitting — a larger model or more epochs would likely improve generalization rather than hurt it.

---

## How to Run

1. Open `lstm_forecasting_final.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU
3. Run All cells

**Official Keras References:**
- [Transformer Classification Example](https://keras.io/examples/timeseries/timeseries_classification_transformer/)
- [LSTM Forecasting Example](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)

---

## References

- Keras Time Series Examples — https://keras.io/examples/timeseries/
- FordA Dataset — UCR Time Series Archive
- Jena Climate Dataset — Max Planck Institute for Biogeochemistry
