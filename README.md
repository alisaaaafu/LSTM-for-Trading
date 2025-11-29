# LSTM Trading Signal Prediction

A deep learning approach to predicting trading signals for the Chinese 50ETF market using LSTM neural networks and implied volatility analysis.

## Overview

This project implements a multi-layer LSTM network to classify market states and generate trading signals based on 7 time-series features including price, volume, and implied volatility index (IVX). The model predicts binary signals: +1 (buy) or -1 (sell).

**Key Results:**
- Overall Accuracy: 58% (Buy: 43%, Sell: 73%)
- Annualized Return: 12.36%
- Win Rate: 41%

## Project Structure

```
LSTM-for-Trading/
├── train/
│   ├── lstm.py          # LSTM model implementation
│   ├── train.py         # Training script
│   └── evaluate.py      # Model evaluation
├── backtest/
│   ├── predict.py       # Real-time prediction
│   └── backtest.py      # Trading strategy backtest
├── data/
│   ├── train.csv        # Training dataset
│   ├── test.csv         # Testing dataset
│   └── feature.csv      # Features with dates
├── models/              # Model checkpoints
└── results/             # Results and visualizations
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
cd train
python train.py

# Evaluate
python evaluate.py

# Backtest
cd ../backtest
python backtest.py
```

## Model Architecture

- **Input:** 7-dimensional time-series (time_step=5)
- **Network:** 2 LSTM layers × 15 units
- **Activation:** Leaky ReLU + Tanh
- **Optimizer:** Adam (lr=0.005)
- **Regularization:** Dropout (0.95)

![Training Loss](results/loss.PNG)

## Backtesting Results

| Metric | Value |
|--------|-------|
| Annualized Return | 12.36% |
| Win Rate | 41.98% |
| Max Drawdown | -4.38% |
| Overall Accuracy | 58% |

![Equity Curve](results/backtest.png)

## Requirements

- Python 3.6+
- TensorFlow 1.15
- NumPy, Pandas, Matplotlib

See `requirements.txt` for full dependencies.

## Note

This project is for educational purposes only. Trading involves substantial risk.
