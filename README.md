# Stock Price Forecasting with Prophet, MACD, RSI & Market Sentiment

This project provides a command-line tool for forecasting stock prices using [Facebook Prophet](https://facebook.github.io/prophet/), with additional signals from MACD, RSI, and Yahoo Finance-based market sentiment data.

The model trains on historical price data and enhances predictions using technical indicators and sentiment analysis. The output is an interactive Plotly chart displaying the historical stock prices, forecasted prices, and confidence intervals.

---

## Features

- 📈 Historical stock data retrieval from Yahoo Finance
- 🔍 Market sentiment analysis using `recommendationMean`
- 📊 MACD and RSI indicator calculation
- 🤖 Time series forecasting using Prophet with extra regressors
- 📉 Interactive forecast visualization with Plotly
- 💾 Saves the forecast plot as an HTML file and auto-opens in your browser

---

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the script using:

```bash
python main.py
```

You will be prompted to:

1. Enter a stock ticker symbol (e.g., `AAPL`, `GOOGL`, `MSFT`).
2. Enter the number of future days to predict.

Example:

```
Enter the stock symbol (e.g., AAPL, GOOGL): AAPL
Enter number of days for Prophet prediction: 60
```

The script will:

- Download 10 years of historical data for the given symbol.
- Calculate MACD and RSI.
- Retrieve market sentiment.
- Train a Prophet model with these features as regressors.
- Forecast future prices.
- Display the forecast in your default web browser.

The "noregressors.py" script is the same except it does not use any regressors (MACD, RSI, or market sentiment).
---

## Dependencies

The script uses the following main libraries:

- `pandas`
- `numpy`
- `yfinance`
- `prophet`
- `plotly`

---
