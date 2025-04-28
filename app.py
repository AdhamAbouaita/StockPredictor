import warnings
warnings.simplefilter('ignore')

import os
import glob
import json
import webbrowser
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler

def get_stock_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    print(f'Downloading data for {symbol} from {start_date.date()} to {end_date.date()}...')
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f'No data found for {symbol}.')
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
        fill='tonexty', name='Confidence', fillcolor='rgba(0,102,255,0.1)'))
    fig.update_layout(
        title=title_text,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_dark'
    )
    return fig

def generate_index_page(charts_dir):
    """
    Build and write the index.html file by injecting
    dynamic chart cards into the HTML template.
    """
    # 1) Gather existing chart HTML files (skip index.html)
    html_files = sorted(glob.glob(os.path.join(charts_dir, '*.html')))
    html_files = [f for f in html_files if os.path.basename(f).lower() != 'index.html']

    # 2) Read metadata (including creation date) for grouping
    groups = {}
    for html_fp in html_files:
        fname = os.path.basename(html_fp)
        base, _ = os.path.splitext(fname)
        manifest_fp = os.path.join(charts_dir, base + '.json')

        # fallback to file modified time
        mtime = os.path.getmtime(html_fp)
        fallback_created = datetime.fromtimestamp(mtime).strftime('%B %d, %Y')

        if os.path.exists(manifest_fp):
            meta = json.load(open(manifest_fp, 'r', encoding='utf-8'))
            yrs     = str(meta.get('years', 'Unknown'))
            ds      = str(meta.get('days', 'Unknown'))
            title   = meta.get('title', base)
            created = meta.get('created') or fallback_created
        else:
            yrs, ds, title, created = 'Unknown', 'Unknown', base, fallback_created

        groups.setdefault(yrs, {}).setdefault(ds, []).append((title, fname, created))

    # Helper: sort numeric keys, Unknown last
    def sort_keys(keys, as_int=False):
        def keyfn(x):
            try:
                return int(x) if as_int else float(x)
            except:
                return -1
        return sorted(keys, key=keyfn, reverse=True)

    # 3) Build dynamic charts HTML
    lines = []
    for yrs in sort_keys(groups.keys(), as_int=False):
        lines.append(f"<h3 class='group-title'>{yrs} years history</h3>")
        for ds in sort_keys(groups[yrs].keys(), as_int=True):
            lines.append(f"<h4 class='horizon-title'>{ds}-day forecast</h4>")
            lines.append("<div class='grid'>")
            for title, fname, created in sorted(groups[yrs][ds], key=lambda x: x[1]):
                sym = title.split()[2].rstrip(',')
                lines.append("<div class='card'>")
                lines.append(f"  <div class='card-header'>{sym}</div>")
                lines.append(f"  <div class='card-meta'>Generated {created}</div>")
                lines.append("  <div class='card-actions'>")
                lines.append(f"    <a class='btn btn-secondary' href='{fname}' target='_blank'>View</a>")
                lines.append(f"    <button class='btn btn-danger' onclick=\"deleteChart('{fname}')\">Delete</button>")
                lines.append("  </div>")
                lines.append("</div>")
            lines.append("</div>")
    charts_html = "\n".join(lines)

    # 4) Read HTML template and inject charts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_fp = os.path.join(script_dir, 'index_template.html')
    with open(template_fp, 'r', encoding='utf-8') as f:
        template = f.read()
    html_content = template.replace('<!-- CHARTS_PLACEHOLDER -->', charts_html)

    # 5) Write to index.html in charts directory
    out_fp = os.path.join(charts_dir, 'index.html')
    with open(out_fp, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Wrote {out_fp}')
    return out_fp

def main():
    # Prepare charts directory
    charts_dir = os.path.join(os.getcwd(), 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    generate_index_page(charts_dir)

    # Serve via HTTP
    os.chdir(charts_dir)

    class Handler(SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path == '/delete':
                length = int(self.headers.get('Content-Length', 0))
                data = json.loads(self.rfile.read(length))
                fn = data.get('filename')
                for ext in ('.html', '.json'):
                    p = os.path.join(charts_dir, fn if ext == '.html' else fn.replace('.html', ext))
                    if os.path.exists(p):
                        os.remove(p)
                generate_index_page(charts_dir)
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.end_headers()
                self.wfile.write(b'{"success":true}')
                return

            if self.path == '/generate':
                length = int(self.headers.get('Content-Length', 0))
                data = json.loads(self.rfile.read(length))
                syms  = data.get('symbols', [])
                years = data.get('years')
                days  = data.get('days')
                if not syms or years is None or days is None:
                    self.send_response(400)
                    self.send_header('Content-Type','application/json')
                    self.end_headers()
                    self.wfile.write(b'{"success":false,"error":"missing parameters"}')
                    return

                end     = datetime.now()
                start   = end - timedelta(days=int(365 * years))
                created = end.strftime('%B %d, %Y')

                for sym in syms:
                    try:
                        raw = get_stock_data(sym, start, end)
                        df  = prepare_data_for_prophet(raw)
                        m, fc = train_prophet_model(df, days)
                        last = fc['ds'].max().date()
                        rd   = f"{last.strftime('%B')} {last.day}, {last.year}"

                        # Unique filename base including timestamp
                        timestamp = end.strftime('%Y%m%d%H%M%S')
                        base = f"{sym}_{years:g}y_{days}d_until_{last.strftime('%Y%m%d')}_{timestamp}"
                        html_fn = f"{base}.html"

                        title = (f"Forecast for {sym}, with {years:g} years of past data, "
                                 f"predicting {days} days into the future, until {rd}")
                        fig = create_plot(df, fc, title)
                        with open(html_fn, 'w', encoding='utf-8') as f:
                            f.write(fig.to_html(full_html=True))

                        manifest = {
                            'years': years,
                            'days': days,
                            'title': title,
                            'created': created
                        }
                        with open(f"{base}.json", 'w', encoding='utf-8') as f:
                            json.dump(manifest, f)

                        print(f"Saved {html_fn}")
                    except Exception as e:
                        print(f"Skipping {sym}: {e}")

                generate_index_page(charts_dir)
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.end_headers()
                self.wfile.write(b'{"success":true}')
                return

            super().do_POST()

    server = HTTPServer(('localhost', 8000), Handler)
    url = 'http://localhost:8000/index.html'
    print(f"Serving charts at {url}")
    webbrowser.open(url)
    server.serve_forever()

if __name__ == '__main__':
    main()
