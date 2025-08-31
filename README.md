# 📈 Stock Price Predictor (NSE) using LSTM

Multi-Feature LSTM Stock Price Predictor (NSE)
Welcome to the Multi-Feature LSTM Price Predictor — an advanced deep learning tool for forecasting the next-day Open, High, Low, Close, and Volume for any NSE-listed stock.
<br>
Powered by PyTorch and real-time data from Yahoo Finance, this project is designed for both researchers and traders.
---

## 🚀 Features

- Multi-Target Prediction: Predicts Open, High, Low, Close, and Volume together.
- Supports All NSE Stocks: Use any NSE symbol (e.g., INFY.NS, HDFCBANK.NS).
- Recent Data: Uses the last 60 days of historical data for predictions.
- Modern Deep Learning: LSTM layers for sequence modeling, dense layers for feature extraction.
- Interactive UI: Optional Streamlit interface for easy predictions.
- Clean & Modular: Well-structured code for easy customization.
---


## How it works 
1. Data Collection: Fetches historical stock data from Yahoo Finance using yfinance.
2. Preprocessing: Scales and formats data for LSTM input.
3. Model:
  - Stacked LSTM layers for temporal dependencies
  - Dense layers for feature refinement
  - Trained with MSE loss and Adam optimizer
  - Prediction: Outputs next-day Open, High, Low, Close, and Volume.

---

## 🧠 Model Architecture

- LSTM layers
- Two Fully Connected (Dense) layers
- ReLU activations
- MSE Loss + Adam Optimizer

---

## 📦 Quickstart

1. Clone the repository:
```bash
git clone https://github.com/sahurishi16/Multi-Feature-LSTM-Price-Predictor.git
cd Multi-Feature-LSTM-Price-Predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Predictor
```bash
python -m streamlit run app.py
```
