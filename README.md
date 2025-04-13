# StockPredictor
# StockPredictor

## Overview
StockPredictor is a machine learning project for forecasting stock prices using advanced algorithms. It leverages Facebook Prophet for time series forecasting and enhances predictions by integrating technical indicators such as MACD and RSI, complemented by market sentiment analysis from Yahoo Finance.

## Features
- **Prophet Forecasting:** Robust time series predictions with Facebook Prophet.
- **Technical Indicators:** Utilize MACD and RSI to capture market trends and momentum.
- **Market Sentiment:** Enrich analysis with sentiment data from Yahoo Finance.
- **Dual Application Scripts:**  
    - app(default).py: Implements forecasting using only Prophet.  
    - app(indicators).py: Combines prophet with technical indicators and market sentiment for enhanced predictions.
- **Data Integration:** Import data from CSV files, online APIs, or direct database connections.
- **Visualization:** Generate interactive charts to explore prediction trends and historical performance.

## Installation
1. Ensure you have Python 3.7 or above.
2. Clone the repository:
        ```
        git clone https://github.com/yourusername/StockPredictor.git
        ```
3. Install the required dependencies:
        ```
        pip install -r requirements.txt
        ```
4. Configure your data source paths and API credentials within the configuration file (`config.yaml`).

## Usage
Choose your preferred approach:
- For Prophet-only forecasting:
        ```
        python app(default).py
        ```
- For enhanced forecasting with technical indicators and market sentiment:
        ```
        python app(indicators).py
        ```

Ensure your dataset is prepared and placed in the appropriate directory before running the scripts.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, kindly open an issue first to discuss potential modifications.

## License
This project is open-sourced under the MIT License.

## Contact
For any questions or suggestions, please open an issue or contact the maintainers via the repository.
