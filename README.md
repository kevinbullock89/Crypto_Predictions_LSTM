# Crypto Trading Data Pipeline and LSTM Prediction

This project implements a data pipeline to fetch cryptocurrency historical data from Yahoo Finance, store it in a MariaDB database, and train LSTM models to predict future prices.  It also includes functionality for incremental data updates and future price predictions.

## Features

* **Data Acquisition:** Fetches historical and real-time cryptocurrency data from Yahoo Finance.
* **Database Storage:** Stores the data in a MySQL database (`crypto_trading`).
* **Incremental Updates:** Efficiently updates the database with new data, avoiding redundant entries.
* **LSTM Model Training:** Trains Long Short-Term Memory (LSTM) models for each cryptocurrency to predict closing prices.
* **Future Price Prediction:** Predicts future cryptocurrency prices using the trained LSTM models.
* **Scheduled Execution:** Designed to be run periodically to update data and retrain models.
* **Batch Processing:** Uses batch processing for database interactions to improve performance.
* **Logging:** Comprehensive logging to track the pipeline's progress and identify potential issues.
* **Model Persistence:** Saves trained models for later use.

## Architecture

The project consists of several Python scripts:

* `loading_finance.py`: Handles fetching and storing historical data.
* `finance_config.py`: Contains database connection details.
* `tensorflow_lstm.py`: Implements the LSTM model training and prediction logic.
* `run_finance.py`: Orchestrates the entire pipeline, including data updates, model training, and predictions.

## Requirements

* Python 3.x
* Required Python libraries: `yfinance`, `mysql-connector-python`, `pandas`, `numpy`, `tensorflow`, `keras`, `scikit-learn`, `pytz`
* MySQL / MariaDB database server


## Installation

1. Clone the repository:
   ```bash
   git clone [invalid URL removed]
