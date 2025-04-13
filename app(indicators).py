#!/usr/bin/env python3
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta
import webbrowser
import os

def get_stock_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    print(f"Downloading data for {symbol} from {start_date.date()} to {end_date.date()}...")
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {symbol}. Please check the stock symbol.")
        print("Data download complete.")
        return data
    except Exception as e:
        raise Exception(f"Error downloading data: {str(e)}")

def get_market_sentiment(symbol):
    """
    Retrieve market sentiment from Yahoo Finance.
    This example attempts to use 'recommendationMean' from the ticker info.
    If it's not available, we default to 3.0 (a neutral score on many rating scales).
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sentiment = info.get('recommendationMean', 3.0)  # 3.0 is a common neutral value
        print(f"Market sentiment for {symbol} (recommendationMean): {sentiment}")
        return sentiment
    except Exception as e:
        print(f"Could not retrieve sentiment data: {e}. Defaulting to neutral (3.0).")
        return 3.0

def calculate_macd_rsi(df):
    """Calculate MACD and RSI indicators and add them as columns to the DataFrame."""
    # Calculate MACD: EMA12 and EMA26, then MACD = EMA12 - EMA26
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    
    # Calculate RSI using a 14-day window
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Clean up temporary EMA columns
    df.drop(columns=['EMA12', 'EMA26'], inplace=True)
    return df

def prepare_data_for_prophet(stock_data, sentiment):
    """
    Prepare data for Prophet.
    1. Resets the index and selects the date and close price (or adjusted close).
    2. Renames the columns so that the closing price column is initially called "Close".
    3. Calculates MACD and RSI.
    4. Renames the closing price column to 'y' (to meet Prophet's expectations) and adds market sentiment.
    """
    # Check if 'Close' exists; otherwise, use 'Adj Close'
    close_col = 'Close' if 'Close' in stock_data.columns else ('Adj Close' if 'Adj Close' in stock_data.columns else None)
    if close_col is None:
        raise ValueError("Neither 'Close' nor 'Adj Close' columns were found in the downloaded data.")

    # Reset index and select columns; keep the name as "Close" for now.
    df = stock_data.reset_index()[['Date', close_col]]
    df.columns = ['ds', 'Close']
    
    # Calculate MACD and RSI on the "Close" column.
    df = calculate_macd_rsi(df)
    
    # Rename the "Close" column to "y" for Prophet compatibility.
    df = df.rename(columns={'Close': 'y'})
    
    # Add the market sentiment value as a new column.
    df['sentiment'] = sentiment
    
    # Drop rows where indicators may be missing (e.g., due to initial period calculations)
    df = df.dropna().reset_index(drop=True)
    return df

def train_prophet_model(df, periods):
    """
    Train Prophet model using the closing price as target and MACD, RSI, and sentiment as additional regressors.
    For future dates, we forward-fill these regressors with the last available values.
    """
    print("Training Prophet model... This might take a few moments.")
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    # Add extra regressors
    model.add_regressor('MACD')
    model.add_regressor('RSI')
    model.add_regressor('sentiment')
    
    model.fit(df)
    print("Model training complete.")
    
    # Generate future DataFrame
    future = model.make_future_dataframe(periods=periods)
    
    # Fill future rows for the additional regressors with the last observed values
    last_macd = df['MACD'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    last_sentiment = df['sentiment'].iloc[-1]
    
    future['MACD'] = last_macd
    future['RSI'] = last_rsi
    future['sentiment'] = last_sentiment
    
    print(f"Making predictions for an additional {periods} days beyond the historical data...")
    forecast = model.predict(future)
    print("Forecast complete.")
    return model, forecast

def create_plot(original_df, forecast, symbol):
    """Create an interactive Plotly chart overlaying historical data with the forecast."""
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=original_df['ds'],
        y=original_df['y'],
        mode='lines',
        name='Actual Stock Price',
        line=dict(color='blue', width=2)
    ))
    
    # Forecasted closing price
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Stock Price',
        line=dict(color='red', width=2)
    ))
    
    # Confidence intervals: upper bound
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    
    # Confidence intervals: lower bound with fill
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fill='tonexty',
        name='Confidence Interval',
        fillcolor='rgba(255,0,0,0.1)'
    ))
    
    fig.update_layout(
        title=f"{symbol.upper()} Stock Price Prediction with Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Stock Price ($)",
        hovermode='x',
        template='plotly_white'
    )
    
    return fig

def main():
    # Get user input
    symbol = input("Enter the stock symbol (e.g., AAPL, GOOGL): ").upper().strip()
    try:
        prediction_days = int(input("Enter number of days for Prophet prediction: ").strip())
    except ValueError:
        print("Invalid input for prediction days. Please enter a valid integer.")
        return

    # Decide on the historical date range (5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)

    # Step 1: Download historical stock data
    try:
        stock_data = get_stock_data(symbol, start_date, end_date)
    except Exception as e:
        print(e)
        return

    # Step 2: Retrieve market sentiment data from Yahoo Finance
    sentiment = get_market_sentiment(symbol)
    
    # Step 3: Prepare the data (calculating MACD, RSI and adding sentiment)
    prophet_df = prepare_data_for_prophet(stock_data, sentiment)
    
    # Step 4: Train the Prophet model with extra regressors and forecast the future
    try:
        model, forecast = train_prophet_model(prophet_df, periods=prediction_days)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # Step 5: Create the forecast plot
    print("Generating the plot. Please wait...")
    fig = create_plot(prophet_df, forecast, symbol)

    # Save the plot as an HTML file and open it in the default web browser
    output_file = "stock_forecast.html"
    fig.write_html(output_file, full_html=True)
    print(f"Plot saved to {output_file}. Opening the graph in your default browser...")
    file_path = os.path.abspath(output_file)
    webbrowser.open("file://" + file_path)

if __name__ == "__main__":
    main()
