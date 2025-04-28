import warnings
warnings.simplefilter("ignore")

import re
import os
import glob
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
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

def generate_index_page(charts_dir):
    """Rebuild an index.html in charts_dir listing all chart files with their titles."""
    pattern = os.path.join(charts_dir, "*.html")
    all_html = sorted(glob.glob(pattern))
    # Skip the index page itself if it exists
    chart_files = [f for f in all_html if os.path.basename(f).lower() != "index.html"]

    items = []
    for filepath in chart_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # extract the <title>…</title>
                head = f.read().split("</head>", 1)[0]
                m = re.search(r"<title>(.*?)</title>", head, re.IGNORECASE|re.DOTALL)
                title = m.group(1).strip() if m else os.path.basename(filepath)
        except Exception:
            title = os.path.basename(filepath)
        link = os.path.basename(filepath)
        items.append(f'    <li><a href="{link}">{title}</a></li>')

    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Chart Index</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    h1 { font-size: 2em; margin-bottom: 20px; }
    ul { list-style-type: none; padding: 0; }
    li { margin: 10px 0; }
    a { color: #1a73e8; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>All Saved Charts</h1>
  <ul>
%s
  </ul>
</body>
</html>
""" % "\n".join(items)

    index_path = os.path.join(charts_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    print(f"Index page written to {index_path}")
    return index_path

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

    # Prepare for Prophet
    prophet_df = prepare_data_for_prophet(stock_data)
    
    # Train & forecast
    try:
        model, forecast = train_prophet_model(prophet_df, periods=prediction_days)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # Build title and filename slug
    last_date = forecast['ds'].max().date()
    readable_date = f"{last_date.strftime('%B')} {last_date.day}, {last_date.year}"
    title_text = f"Forecast for {symbol}, with {training_years} years of past data, until {readable_date}"
    safe_name = slugify(f"Forecast for {symbol}, with {int(training_years)} years of past data, until {readable_date}")

    # Generate the figure
    print("Generating the plot. Please wait...")
    fig = create_plot(prophet_df, forecast, title_text)

    # Prepare directories
    desktop = os.path.expanduser("~/Desktop")
    charts_dir = os.path.join(desktop, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Save individual chart
    output_html = os.path.join(charts_dir, f"{safe_name}.html")
    fig.write_html(output_html, full_html=True)
    print(f"Interactive plot saved to {output_html}.")

    # Rebuild the index.html
    index_path = generate_index_page(charts_dir)

    # Open the index page in the default browser
    webbrowser.open("file://" + os.path.abspath(index_path))


if __name__ == "__main__":
    main()
