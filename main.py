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
    Build index.html with a sleek, futuristic dark-mode design,
    animated gradient background, neon glows, and smooth hover effects.
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

    # 3) Build HTML body
    lines = []
    # Animated gradient background container
    lines.append("<div id='bg'></div>")
    # Navbar
    lines.append("<nav class='navbar'><div class='logo'>🚀 Stock Price Predictor 💵</div></nav>")
    # Main
    lines.append("<main class='container'>")
    # Forecast form
    lines.append("<section class='form-section'>")
    lines.append("  <h2>New Forecast</h2>")
    lines.append("  <form id='paramsForm'>")
    lines.append("    <div class='form-group'><label>Symbols<input type='text' name='symbols' placeholder='AAPL, MSFT...' required></label></div>")
    lines.append("    <div class='form-group'><label>Years<input type='number' step='0.1' name='years' required></label></div>")
    lines.append("    <div class='form-group'><label>Days<input type='number' name='days' required></label></div>")
    lines.append("    <button class='btn btn-primary' type='submit'>Generate</button>")
    lines.append("  </form>")
    lines.append("</section>")
    # Charts grid
    lines.append("<section class='charts-section'>")
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
    lines.append("</section>")
    lines.append("</main>")
    body = "\n".join(lines)

    # 4) Dark-mode, futuristic CSS + animated gradient + glows
    css = """
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg1: #0f0c29;
  --bg2: #302b63;
  --bg3: #24243e;
  --accent: #00fff0;
  --accent2: #ff005d;
  --text: #e0e0e0;
  --shadow: rgba(0, 255, 240, 0.2);
}
* { box-sizing: border-box; margin:0; padding:0; }
body {
  font-family: 'Rajdhani', sans-serif;
  color: var(--text);
  overflow-x: hidden;
}
#bg {
  position: fixed; top:0; left:0; width:100%; height:100%;
  background: linear-gradient(120deg, var(--bg1), var(--bg2), var(--bg3));
  background-size: 600% 600%;
  animation: gradientBG 20s ease infinite;
  z-index: -1;
}
@keyframes gradientBG {
  0%{background-position:0% 50%}
  50%{background-position:100% 50%}
  100%{background-position:0% 50%}
}
.navbar {
  position: sticky; top:0; width:100%;
  background: rgba(0,0,0,0.4); backdrop-filter: blur(10px);
  padding: 1rem 2rem;
  display: flex; align-items:center;
  box-shadow: 0 2px 10px var(--shadow);
}
.logo {
  font-family: 'Orbitron', sans-serif;
  font-size:1.8rem; color: var(--accent);
  text-shadow: 0 0 10px var(--accent);
  animation: glowLogo 2s ease-in-out infinite alternate;
  margin: 0 auto;
}
@keyframes glowLogo {
  from { text-shadow: 0 0 5px var(--accent); }
  to   { text-shadow: 0 0 20px var(--accent); }
}
.container {
  max-width:1200px; margin:2rem auto; padding:0 1rem;
}
.form-section {
  background: rgba(0,0,0,0.5); backdrop-filter: blur(8px);
  padding:2rem; border-radius:12px;
  box-shadow: 0 4px 20px var(--shadow);
  margin-bottom:3rem;
  animation: slideIn 0.6s ease-out;
}
.form-section h2 {
  font-family:'Orbitron', sans-serif;
  color: var(--accent); margin-bottom:1rem;
  letter-spacing:1px;
}
@keyframes slideIn {
  from { opacity:0; transform: translateY(20px); }
  to   { opacity:1; transform: translateY(0); }
}
#paramsForm {
  display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-end;
}
.form-group {
  flex:1 1 180px; display:flex; flex-direction:column;
}
.form-group label {
  font-size:0.85rem; margin-bottom:0.5rem; color: #aaa;
}
.form-group input {
  padding:0.8rem; border:none; border-radius:8px;
  background: rgba(255,255,255,0.05); color:var(--text);
  transition: box-shadow 0.3s, background 0.3s;
}
.form-group input:focus {
  outline:none;
  box-shadow: 0 0 10px var(--accent);
  background: rgba(255,255,255,0.1);
}
.btn {
  position: relative; z-index:1;
  padding:0.8rem 1.5rem; border:none; border-radius:8px;
  font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;
  cursor:pointer; overflow:hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}
.btn-primary {
  background: var(--accent);
  color: #000;
  box-shadow: 0 0 10px var(--accent), 0 0 30px var(--accent);
}
.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 0 20px var(--accent), 0 0 50px var(--accent);
}
.btn-secondary {
  background: transparent;
  border: 1px solid var(--accent);
  color: var(--accent);
  box-shadow: 0 0 10px var(--accent);
}
.btn-secondary:hover {
  background: var(--accent); color: #000;
  box-shadow: 0 0 20px var(--accent), 0 0 50px var(--accent);
}
.btn-danger {
  background: var(--accent2); color:#000;
  box-shadow: 0 0 10px var(--accent2), 0 0 30px var(--accent2);
}
.btn-danger:hover {
  transform: translateY(-2px);
  box-shadow: 0 0 20px var(--accent2), 0 0 50px var(--accent2);
}
.charts-section .group-title {
  font-family:'Orbitron', sans-serif;
  color: var(--accent); margin-bottom:1rem; font-size:1.4rem;
  text-transform:uppercase; letter-spacing:1px;
}
.charts-section .horizon-title {
  font-size:1.1rem; color:#aaa; margin:0.8rem 0;
}
.grid {
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
  gap:1.5rem;
}
.card {
  background: rgba(0,0,0,0.4); backdrop-filter: blur(8px);
  border:1px solid rgba(0,255,240,0.2);
  border-radius:12px; padding:1.2rem;
  box-shadow: 0 4px 20px var(--shadow);
  display:flex; flex-direction:column; justify-content:space-between;
  transition: transform 0.3s, box-shadow 0.3s;
}
.card:hover {
  transform: translateY(-8px);
  box-shadow: 0 6px 30px var(--shadow);
}
.card-header {
  font-family:'Orbitron',sans-serif;
  font-size:1.3rem; color:var(--accent); margin-bottom:0.5rem;
}
.card-meta {
  font-size:0.8rem; color:#bbb; margin-bottom:1rem;
}
.card-actions {
  display:flex; gap:0.6rem;
}
@media (max-width:600px) {
  .grid { grid-template-columns:1fr; }
}
</style>
"""

    # 5) JavaScript for delete + generate (retained functionality)
    script = """
<script>
function deleteChart(fname) {
  fetch('/delete', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({filename:fname})
  })
  .then(r=>r.json())
  .then(resp=>{
    if (resp.success) location.reload();
    else alert('Failed to delete ' + fname);
  });
}

document.getElementById('paramsForm').addEventListener('submit', function(e) {
  e.preventDefault();
  let data = {
    symbols: this.symbols.value.split(',').map(s=>s.trim().toUpperCase()).filter(Boolean),
    years: parseFloat(this.years.value),
    days: parseInt(this.days.value)
  };
  fetch('/generate', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(data)
  })
  .then(r=>r.json())
  .then(resp=>{
    if (resp.success) location.reload();
    else alert('Error: ' + (resp.error||'unknown'));
  })
  .catch(err=>alert('Network error: '+err));
});
</script>
"""

    # Combine into full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FuturoCharts</title>
  {css}
</head>
<body>
{body}
{script}
</body>
</html>
"""

    out_fp = os.path.join(charts_dir, 'index.html')
    with open(out_fp, 'w', encoding='utf-8') as f:
        f.write(html)
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
                        title = (f"Forecast for {sym}, with {years:g} years of past data, "
                                 f"predicting {days} days into the future, until {rd}")
                        html_fn = f"{sym}, until {rd}.html"
                        fig = create_plot(df, fc, title)
                        with open(html_fn, 'w', encoding='utf-8') as f:
                            f.write(fig.to_html(full_html=True))
                        manifest = {
                            'years': years,
                            'days': days,
                            'title': title,
                            'created': created
                        }
                        with open(f"{sym}, until {rd}.json", 'w', encoding='utf-8') as f:
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
