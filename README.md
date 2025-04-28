# Stock Price Forecasting with Prophet

This Python script downloads historical stock data from Yahoo Finance, trains a forecasting model using Facebook's Prophet, and generates an interactive HTML plot of the forecast.

## Features

- **Historical Data:** Downloads stock data based on a user-specified number of years.
- **Forecasting:** Uses Prophet to predict future stock prices over a user-defined number of days.
- **Visualization:** Creates an interactive forecast plot with Plotly and opens it in your default web browser.

---

## Requirements

Make sure to create your own venv in order to properly and easily install all the libraries from requirements.txt!
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
- Forecast future prices.
- Save the interactive charts as .html files into a folder on your desktop called "charts".
- Display the forecast in your default web browser.

---

## Dependencies

The script uses the following main libraries:

- `pandas`
- `numpy`
- `yfinance`
- `prophet`
- `plotly`

---
