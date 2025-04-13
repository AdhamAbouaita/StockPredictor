# Stock Price Forecasting and Technical Analysis

This project provides two Python applications that forecast stock prices using the [Prophet](https://facebook.github.io/prophet/) time series forecasting tool and visualize the results using [Plotly](https://plotly.com/python/). In addition, one of the applications incorporates technical indicators (MACD and RSI) and market sentiment data from Yahoo Finance for an enhanced prediction model.

## Overview

The repository contains two main scripts:

- **app(default).py**  
  Downloads historical stock data from Yahoo Finance, prepares the data for Prophet, trains a forecasting model, and generates an interactive Plotly chart showing both the historical prices and the forecasted values.

- **app(indicators).py**  
  Enhances the default model by incorporating technical indicators (MACD and RSI) and market sentiment data. This version also downloads data from Yahoo Finance, computes technical indicators, adds them as extra regressors to the Prophet model, and outputs an interactive Plotly chart with forecasted results.

## Features

- **Historical Data Download:** Uses the `yfinance` library to fetch historical stock prices.
- **Forecasting with Prophet:** Leverages the Prophet model with daily, weekly, and yearly seasonality for time series forecasting.
- **Interactive Visualizations:** Generates interactive line charts with Plotly that include confidence intervals.
- **Technical Indicators (in `app(indicators).py`):** Computes MACD and RSI for additional market insights.
- **Market Sentiment Analysis (in `app(indicators).py`):** Integrates a sentiment score from Yahoo Finance to enhance the forecasting model.
- **User Input:** Prompts the user to enter a stock symbol and the number of days for future predictions.

## Requirements

Make sure you have the following Python packages installed:

- [pandas](https://pandas.pydata.org/)
- [yfinance](https://pypi.org/project/yfinance/)
- [prophet](https://pypi.org/project/prophet/)
- [plotly](https://plotly.com/python/)
- [numpy](https://numpy.org/) (required for indicators in `app(indicators).py`)

You can install the necessary packages using pip:

```bash
pip install pandas yfinance prophet plotly numpy
