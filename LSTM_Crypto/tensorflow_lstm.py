import os
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from keras.models import Sequential, load_model # type: ignore
from keras.layers import Dense, LSTM, Dropout # type: ignore
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from finance_config import get_db_connection

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Determine the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")  # Subfolder for saving models
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure the directory exists

# Function to preprocess data
def preprocess_data(df, n_steps):
    """
    Preprocess the data for LSTM training.

    Args:
        df (pd.DataFrame): The input DataFrame with 'Datetime', 'Close', and 'Crypto_Code'.
        n_steps (int): Number of time steps to consider for predictions.

    Returns:
        dict: Dictionary with crypto codes as keys and scaled data as values.
    """
    crypto_data = {}
    for crypto_code in df['Crypto_Code'].unique():
        crypto_df = df[df['Crypto_Code'] == crypto_code].copy()
        crypto_df['Close'] = crypto_df['Close'].astype(float)

        # Scale the data
        scaler = MinMaxScaler()
        crypto_df['Scaled_Close'] = scaler.fit_transform(crypto_df[['Close']])

        # Create sequences for LSTM
        X, y = [], []
        for i in range(n_steps, len(crypto_df)):
            X.append(crypto_df['Scaled_Close'].iloc[i - n_steps:i].values)
            y.append(crypto_df['Scaled_Close'].iloc[i])

        crypto_data[crypto_code] = {
            'X': np.array(X),
            'y': np.array(y),
            'scaler': scaler,
            'df': crypto_df.reset_index(drop=True),
        }

    return crypto_data

def build_or_load_model(n_steps, crypto_code, units=50, dropout=0.2):
    """
    Build or load an LSTM model for a specific cryptocurrency.

    Args:
        n_steps (int): Number of time steps in the input sequences.
        crypto_code (str): Cryptocurrency code for the model.
        units (int): Number of LSTM units.
        dropout (float): Dropout rate.

    Returns:
        keras.Sequential: LSTM model.
    """
    model_path = os.path.join(MODEL_DIR, f"{crypto_code}_lstm_model.h5")

    if os.path.exists(model_path):
        print(f"Loading model for cryptocurrency '{crypto_code}' from {model_path}")
        model = load_model(model_path)
        model.compile(optimizer="adam", loss="mean_squared_error")  # Recompile
        return model

    print(f"Creating a new model for cryptocurrency '{crypto_code}'")
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Function to train models
def train_models(df, n_steps=120, epochs=20, batch_size=32):
    """
    Train models for all cryptocurrencies in the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame containing cryptocurrency data.
        n_steps (int): Number of time steps for training.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
    """
    crypto_data = preprocess_data(df, n_steps)
    results = {}

    for crypto_code, data in crypto_data.items():
        X, y = data['X'], data['y']
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = build_or_load_model(n_steps, crypto_code)
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

        model_path = os.path.join(MODEL_DIR, f"{crypto_code}_lstm_model.h5")
        model.save(model_path)

        training_metrics = {
            "final_loss": history.history['loss'][-1],
            "epochs": epochs,
            "batch_size": batch_size,
        }
        model_params = {
            "n_steps": n_steps,
            "units": 50,
            "dropout": 0.2,
        }
        dataset_info = {
            "crypto_code": crypto_code,
            "num_samples": X.shape[0],
        }

        save_training_summary(crypto_code, training_metrics, model_params, dataset_info, model_path)
        print(f"Model for cryptocurrency '{crypto_code}' saved at {model_path}")

# Function to predict prices
def predict_prices(df, n_steps=120):
    """
    Predict prices using pre-trained models.

    Args:
        df (pd.DataFrame): Input DataFrame containing cryptocurrency data.
        n_steps (int): Number of time steps for prediction.

    Returns:
        pd.DataFrame: DataFrame containing predictions.
    """
    crypto_data = preprocess_data(df, n_steps)
    results = []

    for crypto_code, data in crypto_data.items():
        X = data['X']
        scaler = data['scaler']
        original_df = data['df']

        model = build_or_load_model(n_steps, crypto_code)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)

        result_df = original_df.iloc[n_steps:].copy()
        result_df['Predicted_Price'] = predictions.flatten()
        results.append(result_df[['Datetime', 'Crypto_Code', 'Close', 'Predicted_Price']])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def predict_future_prices(df, n_steps=120, future_steps=10):
    """
    Predict future prices using pre-trained models.

    Args:
        df (pd.DataFrame): Input DataFrame containing cryptocurrency data.
        n_steps (int): Number of time steps for the LSTM model input.
        future_steps (int): Number of future steps to predict.

    Returns:
        pd.DataFrame: DataFrame containing future predictions.
    """
    crypto_data = preprocess_data(df, n_steps)
    future_results = []

    for crypto_code, data in crypto_data.items():
        X = data['X']
        scaler = data['scaler']
        original_df = data['df']

        if len(X) == 0:
            print(f"Not enough data to predict future prices for {crypto_code}. Skipping...")
            continue

        model = build_or_load_model(n_steps, crypto_code)

        # Start with the last available sequence
        last_sequence = X[-1]  # Shape: (n_steps,)
        predictions = []

        # Generate future timestamps
        last_timestamp = pd.to_datetime(original_df['Datetime'].iloc[-1])
        time_interval = last_timestamp - pd.to_datetime(original_df['Datetime'].iloc[-2])
        future_timestamps = [last_timestamp + (i + 1) * time_interval for i in range(future_steps)]

        # Iteratively predict future prices
        for _ in range(future_steps):
            last_sequence_reshaped = last_sequence.reshape(1, n_steps, 1)
            predicted_price_scaled = model.predict(last_sequence_reshaped)[0, 0]
            predicted_price = scaler.inverse_transform([[predicted_price_scaled]])[0, 0]

            predictions.append(predicted_price)

            # Update the sequence with the new prediction
            last_sequence = np.append(last_sequence[1:], predicted_price_scaled)

        # Prepare results for this cryptocurrency
        future_df = pd.DataFrame({
            'Datetime': future_timestamps,
            'Crypto_Code': crypto_code,
            'Predicted_Price': predictions
        })
        future_results.append(future_df)

    # Combine results for all cryptocurrencies
    return pd.concat(future_results, ignore_index=True) if future_results else pd.DataFrame()

# Function to save predictions to the database
def save_predictions_to_db(results_df, batch_size=1000):
    """
    Save the predictions to the database in batches.

    Args:
        results_df (pd.DataFrame): DataFrame containing predictions and actual prices.
        batch_size (int): Number of rows to insert per batch.
    """
    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    # Prepare the list of tuples for bulk insert
    insert_data = []
    for _, row in results_df.iterrows():
        naive_datetime = row['Datetime'].to_pydatetime().replace(tzinfo=None)
        insert_data.append((
            naive_datetime,
            row['Crypto_Code'],
            row['Close'],
            row['Predicted_Price']
        ))

        # Insert in batches
        if len(insert_data) >= batch_size:
            cursor.executemany("""
                INSERT INTO predicted_crypto_prices (datetime, currency_code, actual_price, predicted_price, created_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP())
                ON DUPLICATE KEY UPDATE
                    actual_price = VALUES(actual_price), predicted_price = VALUES(predicted_price)
                """, insert_data)
            insert_data = []  # Reset for the next batch

    # Insert remaining rows (if any)
    if insert_data:
        cursor.executemany("""
            INSERT INTO predicted_crypto_prices (datetime, currency_code, actual_price, predicted_price, created_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP())
            ON DUPLICATE KEY UPDATE
                actual_price = VALUES(actual_price), predicted_price = VALUES(predicted_price)
            """, insert_data)

    db_connection.commit()
    cursor.close()
    db_connection.close()
    print(f"Inserted {len(results_df)} prediction records in batches.")


# Function to save training summary to the database
def save_training_summary(crypto_code, training_metrics, model_params, dataset_info, model_path, batch_size=100):
    """
    Save the training summary to the database in batches.

    Args:
        crypto_code (str): Cryptocurrency code.
        training_metrics (dict): Dictionary of training metrics.
        model_params (dict): Dictionary of model hyperparameters.
        dataset_info (dict): Information about the dataset.
        model_path (str): Path to the trained model.
        batch_size (int): Number of rows to insert per batch.
    """
    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    # Prepare the list of tuples for bulk insert
    insert_data = []
    training_summary = {
        "crypto_code": crypto_code,
        "training_loss": training_metrics.get("final_loss"),
        "epochs": training_metrics.get("epochs"),
        "batch_size": training_metrics.get("batch_size"),
        "n_steps": model_params.get("n_steps"),
        "lstm_units": model_params.get("units"),
        "dropout": model_params.get("dropout"),
        "num_samples": dataset_info.get("num_samples"),
        "model_path": model_path,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    insert_data.append((
        training_summary["crypto_code"],
        training_summary["training_loss"],
        training_summary["epochs"],
        training_summary["batch_size"],
        training_summary["n_steps"],
        training_summary["lstm_units"],
        training_summary["dropout"],
        training_summary["num_samples"],
        training_summary["model_path"],
        training_summary["trained_at"]
    ))

    # Insert in batches
    if len(insert_data) >= batch_size:
        cursor.executemany("""
            INSERT INTO training_summaries (
                crypto_code, training_loss, epochs, batch_size,
                n_steps, lstm_units, dropout, num_samples,
                model_path, trained_at, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
            ON DUPLICATE KEY UPDATE
                training_loss = VALUES(training_loss),
                epochs = VALUES(epochs),
                batch_size = VALUES(batch_size),
                n_steps = VALUES(n_steps),
                lstm_units = VALUES(lstm_units),
                dropout = VALUES(dropout),
                num_samples = VALUES(num_samples),
                model_path = VALUES(model_path),
                trained_at = VALUES(trained_at)
            """, insert_data)
        insert_data = []  # Reset for the next batch

    # Insert remaining rows (if any)
    if insert_data:
        cursor.executemany("""
            INSERT INTO training_summaries (
                crypto_code, training_loss, epochs, batch_size,
                n_steps, lstm_units, dropout, num_samples,
                model_path, trained_at, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
            ON DUPLICATE KEY UPDATE
                training_loss = VALUES(training_loss),
                epochs = VALUES(epochs),
                batch_size = VALUES(batch_size),
                n_steps = VALUES(n_steps),
                lstm_units = VALUES(lstm_units),
                dropout = VALUES(dropout),
                num_samples = VALUES(num_samples),
                model_path = VALUES(model_path),
                trained_at = VALUES(trained_at)
            """, insert_data)

    db_connection.commit()
    cursor.close()
    db_connection.close()
    print(f"Training summary for {crypto_code} saved in batches.")


def run_training():
    """
    Main function to fetch data, predict prices, and save predictions.
    """
    start_time = time.time()

    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    cursor.execute("SELECT * FROM vw_crypto_lstm")
    column_names = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=column_names)

    # Predict prices for all crypto codes
    train_models(df)

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"Training completed in in {minutes} minutes and {seconds} seconds.")


def run_predictions():
    """
    Main function to fetch data, predict prices, and save predictions.
    """
    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    start_time = time.time()

    # Fetch only recent data since the last prediction interval
    cursor.execute('''
                    SELECT * FROM vw_crypto_lstm WHERE CAST(`Datetime` AS date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY);
                   ''')
    column_names = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=column_names)

    if df.empty:
        print("No new data available for predictions.")
        return    

    # Predict prices for all crypto codes
    results_df = predict_prices(df)

    # Save predictions to the database
    save_predictions_to_db(results_df)

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Training completed in {minutes} minutes and {seconds} seconds.")


def run_future_predictions(future_steps=10):
    """
    Main function to fetch data, predict future prices, and save predictions.

    Args:
        future_steps (int): Number of future steps to predict.
    """
    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    start_time = time.time()

    # Fetch only the most recent data needed for `n_steps`
    cursor.execute("SELECT * FROM vw_crypto_lstm WHERE CAST(`Datetime` AS date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 120 DAY);")
    column_names = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=column_names)

    if df.empty:
        print("No new data available for predictions.")
        return    

    # Predict future prices
    future_results_df = predict_future_prices(df, n_steps=120, future_steps=future_steps)

    # Modify DataFrame to match the required format for `save_predictions_to_db`
    future_results_df['Close'] = None  # Set actual prices to None for future predictions

    # Save future predictions to the database using the existing function
    save_predictions_to_db(future_results_df)

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print(f"Future predictions for the next {future_steps} steps have been saved to the database in {minutes} minutes and {seconds} seconds.")
