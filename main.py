import warnings
warnings.simplefilter("ignore")

import re
import os
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
import plotly.io as pio  # ensure you have kaleido installed: pip install -U kaleido
from datetime import datetime, timedelta
import webbrowser

def slugify(text: str) -> str:
    """Turn a title into a safe filename: remove invalid chars, replace spaces with underscores."""
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text

def get_stock_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    print(f"Downloading data for {symbol} from {start_date.date()} to {end_date.date()}...")
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for {symbol}. Please check the stock symbol.")
    print("Data download complete.")
    return data

def prepare_data_for_prophet(df):
    """Prepare data for Prophet: reformatting date and renaming columns."""
    df = df.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df

def train_prophet_model(df, periods):
    """Train Prophet model and extend the forecast."""
    print("Training Prophet model... This might take a few moments.")
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        weekly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(df)
    print("Model training complete.")
    
    future = model.make_future_dataframe(periods=periods)
    print(f"Making predictions for an additional {periods} days beyond the historical data...")
    forecast = model.predict(future)
    print("Forecast complete.")
    return model, forecast

def create_plot(original_df, forecast, title_text):
    """Create an interactive Plotly chart overlaying historical data with the forecast."""
    fig = go.Figure()

    # Actual stock prices
    fig.add_trace(go.Scatter(
        x=original_df['ds'],
        y=original_df['y'],
        mode='lines',
        name='Actual Stock Price',
        line=dict(color='blue', width=2)
    ))
    
    # Forecasted values
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Stock Price',
        line=dict(color='red', width=2)
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
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
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Stock Price ($)",
        hovermode='x',
        template='plotly_white'
    )
    
    return fig

def main():
    # User input
    symbol = input("Enter the stock symbol (e.g., AAPL, GOOGL): ").upper().strip()
    try:
        prediction_days = int(input("Enter number of days for Prophet prediction: ").strip())
    except ValueError:
        print("Invalid input for prediction days. Please enter a valid integer.")
        return

    try:
        training_years = float(input("Enter number of years of historical data for training (e.g., 10): ").strip())
    except ValueError:
        print("Invalid input for historical data period. Please enter a valid number.")
        return

    # Compute date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(365 * training_years))
    
    # Download data
    try:
        stock_data = get_stock_data(symbol, start_date, end_date)
    except Exception as e:
        print(e)
        return

    # Prep for Prophet
    prophet_df = prepare_data_for_prophet(stock_data)
    
    # Train & forecast
    try:
        model, forecast = train_prophet_model(prophet_df, periods=prediction_days)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # Get final forecast date
    last_date = forecast['ds'].max().date()
    
    # Build date strings
    numeric_date = last_date.isoformat()  # for filenames
    readable_date = f"{last_date.strftime('%B')} {last_date.day}, {last_date.year}"  # for chart title

    # Chart title with month name
    title_text = f"Forecast for {symbol}, {training_years} years of past data, until {readable_date}"
    safe_name = slugify(f"Forecast for {symbol}, {training_years} years of past data, until {numeric_date}")

    # Generate the figure
    print("Generating the plot. Please wait...")
    fig = create_plot(prophet_df, forecast, title_text)

    # Base charts folder on user's Desktop
    desktop = os.path.expanduser("~/Desktop")
    charts_base = os.path.join(desktop, "charts")
    images_dir = os.path.join(charts_base, "chartimages")
    pages_dir = os.path.join(charts_base, "chartpages")
    
    # Ensure directories exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)

    # Save interactive HTML into 'chartpages'
    output_html = os.path.join(pages_dir, f"{safe_name}.html")
    fig.write_html(output_html, full_html=True)
    print(f"Interactive plot saved to {output_html}.")

    # Save static PNG via Kaleido into 'chartimages'
    output_png = os.path.join(images_dir, f"{safe_name}.png")
    fig.write_image(output_png, engine="kaleido")
    print(f"Static image saved to {output_png}.")

    # Open the interactive HTML
    webbrowser.open("file://" + os.path.abspath(output_html))

if __name__ == "__main__":
    main()
