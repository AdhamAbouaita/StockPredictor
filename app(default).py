#!/usr/bin/env python3
import pandas as pd
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

def prepare_data_for_prophet(df):
    """Prepare data for Prophet: reformatting date and renaming columns."""
    df = df.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df

def train_prophet_model(df, periods):
    """Train Prophet model and extend the forecast."""
    print("Training Prophet model... This might take a few moments.")
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(df)
    print("Model training complete.")
    
    future = model.make_future_dataframe(periods=periods)
    print(f"Making predictions for an additional {periods} days beyond the historical data...")
    forecast = model.predict(future)
    print("Forecast complete.")
    return model, forecast

def create_plot(original_df, forecast, symbol):
    """Create an interactive Plotly chart overlaying historical data with the forecast."""
    # Create figure
    fig = go.Figure()

    # Add actual stock prices (historical data)
    fig.add_trace(go.Scatter(
        x=original_df['ds'],
        y=original_df['y'],
        mode='lines',
        name='Actual Stock Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add forecasted values
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Stock Price',
        line=dict(color='red', width=2)
    ))
    
    # Add upper confidence interval (invisible trace for proper shading)
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    
    # Add lower confidence interval with shading to create the confidence band
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fill='tonexty',
        name='Confidence Interval',
        fillcolor='rgba(255,0,0,0.1)'  # Light red shaded area
    ))
    
    fig.update_layout(
        title=f"{symbol.upper()} Stock Price Prediction",
        xaxis_title="Date",
        yaxis_title="Stock Price ($)",
        hovermode='x',
        template='plotly_white'
    )
    
    return fig

def main():
    # Console prompts for user input
    symbol = input("Enter the stock symbol (e.g., AAPL, GOOGL): ").upper().strip()
    try:
        prediction_days = int(input("Enter number of days for Prophet prediction: ").strip())
    except ValueError:
        print("Invalid input for prediction days. Please enter a valid integer.")
        return

    # Determine the date range for the past 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)
    
    # Step 1: Get historical stock data
    try:
        stock_data = get_stock_data(symbol, start_date, end_date)
    except Exception as e:
        print(e)
        return

    # Step 2: Prepare data for Prophet
    prophet_df = prepare_data_for_prophet(stock_data)
    
    # Step 3: Train Prophet model and forecast future data
    try:
        model, forecast = train_prophet_model(prophet_df, periods=prediction_days)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # Step 4: Create and display the forecast plot
    print("Generating the plot. Please wait...")
    fig = create_plot(prophet_df, forecast, symbol)

    # Save the plot as an HTML file
    output_file = "stock_forecast.html"
    fig.write_html(output_file, full_html=True)
    print(f"Plot saved to {output_file}. Opening the graph in your default browser...")
    
    # Open the file in the default web browser
    file_path = os.path.abspath(output_file)
    webbrowser.open("file://" + file_path)

if __name__ == "__main__":
    main()
