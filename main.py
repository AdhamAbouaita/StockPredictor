import warnings
warnings.simplefilter("ignore")

import os
import glob
import json
import re
import webbrowser
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler

def get_stock_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    print(f"\nDownloading data for {symbol} from {start_date.date()} to {end_date.date()}...")
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for {symbol}.")
    return data

def prepare_data_for_prophet(df):
    """Prepare data for Prophet: reformat date and rename columns."""
    df = df.reset_index()[['Date','Close']]
    df.columns = ['ds','y']
    return df

def train_prophet_model(df, periods):
    """Train Prophet model and extend the forecast."""
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        weekly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

def create_plot(original_df, forecast, title_text):
    """Create an interactive Plotly chart overlaying historical data with the forecast."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=original_df['ds'], y=original_df['y'],
        mode='lines', name='Actual', line=dict(width=2)))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='Forecast', line=dict(width=2)))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        mode='lines', line=dict(color='rgba(0,0,0,0)'),
        fill='tonexty', name='Confidence', fillcolor='rgba(255,0,0,0.1)'))
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x',
        template='plotly_white'
    )
    return fig

def generate_index_page(charts_dir):
    """
    Build index.html grouping by years ➔ days, reading each chart's JSON manifest.
    Display each link as "SYMBOL, until Month Day, Year".
    """
    # 1) Find all chart HTML files (skip index.html itself)
    html_files = sorted(glob.glob(os.path.join(charts_dir, "*.html")))
    html_files = [f for f in html_files if os.path.basename(f).lower() != "index.html"]

    # 2) Read metadata from each side-car JSON
    groups = {}
    for html_fp in html_files:
        fname = os.path.basename(html_fp)
        base, _ = os.path.splitext(fname)
        manifest_fp = os.path.join(charts_dir, base + ".json")
        if os.path.exists(manifest_fp):
            meta = json.load(open(manifest_fp, 'r', encoding='utf-8'))
            yrs = str(meta.get("years", "Unknown"))
            days = str(meta.get("days", "Unknown"))
            title = meta.get("title", base)
        else:
            yrs, days, title = "Unknown", "Unknown", base

        groups.setdefault(yrs, {}).setdefault(days, []).append((title, fname))

    # 3) Helpers to sort numeric keys with "Unknown" last
    def sort_keys(keys, as_int=False):
        def keyfn(x):
            try:
                return int(x) if as_int else float(x)
            except:
                return -1
        return sorted(keys, key=keyfn, reverse=True)

    # 4) Build the HTML body
    lines = ["  <h1>All Saved Charts</h1>"]
    for yrs in sort_keys(groups.keys(), as_int=False):
        lines.append(f"  <h2>{yrs} years of history</h2>")
        for ds in sort_keys(groups[yrs].keys(), as_int=True):
            lines.append(f"    <h3>Forecast horizon: {ds} days</h3>")
            lines.append("    <ul>")
            # sort alphabetically by filename base
            for title, fname in sorted(groups[yrs][ds], key=lambda x: x[1]):
                display_name = os.path.splitext(fname)[0]  # "SYMBOL, until Month Day, Year"
                lines.append(
                    f'      <li style="margin-bottom:15px;">'
                    f'<a href="{fname}">{display_name}</a> '
                    f'<button onclick="deleteChart(\'{fname}\')">Delete</button>'
                    f'</li>'
                )
            lines.append("    </ul>")

    body = os.linesep.join(lines)

    # 5) JavaScript for delete-button behavior
    script = """
<script>
function deleteChart(fname) {
  fetch('/delete', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({filename:fname})
  })
  .then(r=>r.json())
  .then(j=>{
    if(j.success) location.reload();
    else alert('Delete failed for ' + fname);
  });
}
</script>
"""

    # 6) Wrap in full HTML
    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Chart Index</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1 {{ font-size: 2em; margin-bottom: 20px; }}
    h2 {{ margin-top: 30px; color: #333; }}
    h3 {{ margin-left: 10px; color: #555; }}
    ul {{ list-style-type: none; padding-left: 20px; }}
    li a {{ color: #1a73e8; text-decoration: none; }}
    li a:hover {{ text-decoration: underline; }}
    button {{ margin-left: 10px; }}
  </style>
</head>
<body>
{body}
{script}
</body>
</html>
"""

    # 7) Write index.html
    out = os.path.join(charts_dir, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(index_html)
    print(f"Wrote {out}")
    return out

def main():
    # 1) Gather symbols until 'done'
    symbols = []
    while True:
        s = input("Enter a stock symbol (or 'done'): ").strip().upper()
        if s == "DONE":
            break
        if s:
            symbols.append(s)
    if not symbols:
        print("No symbols entered; exiting.")
        return

    # 2) Get forecasting parameters
    try:
        days = int(input("Enter number of days for Prophet prediction: ").strip())
        years = float(input("Enter number of years of historical data: ").strip())
    except ValueError:
        print("Invalid input; exiting.")
        return

    # Compute date range
    end = datetime.now()
    start = end - timedelta(days=int(365 * years))

    # Prepare output directory
    desktop = os.path.expanduser("~/Desktop")
    charts_dir = os.path.join(desktop, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # 3) Loop through symbols, generate chart + manifest
    for sym in symbols:
        try:
            raw = get_stock_data(sym, start, end)
        except Exception as e:
            print(f"Skipping {sym}: {e}")
            continue

        df = prepare_data_for_prophet(raw)
        model, forecast = train_prophet_model(df, days)
        last = forecast['ds'].max().date()
        rd = f"{last.strftime('%B')} {last.day}, {last.year}"

        # Full descriptive title (for manifest)
        title = (
            f"Forecast for {sym}, with {years:g} years of past data, "
            f"predicting {days} days into the future, until {rd}"
        )

        # Exact requested HTML filename
        html_filename = f"{sym}, until {rd}.html"
        html_fp = os.path.join(charts_dir, html_filename)

        print("Generating the plot. Please wait...")
        fig = create_plot(df, forecast, title)
        # Write the HTML file
        with open(html_fp, "w", encoding="utf-8") as f:
            f.write(fig.to_html(full_html=True))

        # Side-car JSON manifest for grouping
        manifest = {"years": years, "days": days, "title": title}
        manifest_fp = os.path.join(charts_dir, f"{sym}, until {rd}.json")
        with open(manifest_fp, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        print(f"Saved chart: {html_fp}")

    # 4) Generate index and start server
    generate_index_page(charts_dir)
    os.chdir(charts_dir)

    class Handler(SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/delete":
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                fn = payload.get("filename")
                # remove both .html and .json
                for ext in (".html", ".json"):
                    p = os.path.join(charts_dir, fn if ext == ".html" else fn.replace(".html", ext))
                    if os.path.exists(p):
                        os.remove(p)
                generate_index_page(charts_dir)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"success":true}')
            else:
                super().do_POST()

    server = HTTPServer(("localhost", 8000), Handler)
    print("Serving charts at http://localhost:8000/index.html")
    webbrowser.open("http://localhost:8000/index.html")
    server.serve_forever()

if __name__ == "__main__":
    main()
