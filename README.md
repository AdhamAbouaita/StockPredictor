# Stock Price Forecasting with Prophet, MACD, RSI & Market Sentiment

This Python script downloads historical stock data from Yahoo Finance, trains a forecasting model using Facebook's Prophet, and generates an interactive HTML plot of the forecast.

## Features

- **Historical Data:** Downloads stock data based on a user-specified number of years.
- **Forecasting:** Uses Prophet to predict future stock prices over a user-defined number of days.
- **Visualization:** Creates an interactive forecast plot with Plotly and opens it in your default web browser.

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
- Save the interactive charts as .html and .png files into a folder on your desktop called "charts".
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
- `kaleido`

---
