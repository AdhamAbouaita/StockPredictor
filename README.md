# Stock Price Forecasting with Prophet

You can run this application via the web version, or you can run it locally. If you want to run it locally, 
follow the instructions below, otherwise, here is the link to the web version: [Stock Predictor Web App](https://stockpredictor-l0iq.onrender.com).

## Requirements

Make sure to create your own venv in order to properly and easily install all the libraries from requirements.txt!
Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Usage

Delete/comment the following segment of the code towards the very bottom:
    ```python
    port = int(os.environ.get('PORT', 8000))
    server = HTTPServer(('0.0.0.0', port), Handler)
    url = f'{port}/index.html'
    print(f"Serving charts at {url}")
    webbrowser.open(url)
    server.serve_forever()
    ```
and uncomment this segment of code right after it:
```python
server = HTTPServer(('localhost', 8000), Handler)
url = 'http://localhost:8000/index.html'
```

Run the local version of the webb-app using:

```bash
python local.py
```

---

## Dependencies

The script uses the following main libraries:

- `pandas`
- `numpy`
- `yfinance`
- `prophet`
- `plotly`

---
