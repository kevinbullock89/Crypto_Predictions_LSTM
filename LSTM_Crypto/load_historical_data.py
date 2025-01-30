import yfinance as yf
import pytz
from datetime import datetime
import pandas as pd
import logging
from finance_config import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_cryptos(db_conn):
    cursor = db_conn.cursor()
    cursor.execute("SELECT code FROM crypto_currencies")
    cryptos = [row[0] for row in cursor]
    cursor.close()
    return cryptos

def fetch_historical_data(symbol, start_date, end_date, interval='1d'):
    """Fetch historical data for a given symbol from yfinance."""
    stock = yf.Ticker(symbol)
    hist = stock.history(start=start_date, end=end_date, interval=interval)
    
    if hist.empty:
        logging.warning(f"No data found for {symbol} in the specified date range.")
        return pd.DataFrame()
    
    logging.info(f"Fetched {len(hist)} records for {symbol} from {start_date} to {end_date}.")
    return hist

def send_historical_data_to_db(db_conn, data, symbol, batch_size=1000):
    """Insert historical data into the `crypto_data` table."""
    if data.empty:
        logging.info(f"No data to insert for {symbol}.")
        return

    cursor = db_conn.cursor()
    insert_data = []

    for index, row in data.iterrows():
        naive_index = index.to_pydatetime()  # Remove timezone for database compatibility

        # Check if this data already exists in the database
        cursor.execute(
            "SELECT COUNT(1) FROM crypto_data WHERE Datetime = %s AND Currency_Code = %s",
            (naive_index, symbol)
        )
        existing_record = cursor.fetchone()[0]

        if existing_record == 0:  # If no record exists, prepare data for bulk insertion
            insert_data.append((
                naive_index,
                float(row['Open']) if not pd.isna(row['Open']) else None,
                float(row['High']) if not pd.isna(row['High']) else None,
                float(row['Low']) if not pd.isna(row['Low']) else None,
                float(row['Close']) if not pd.isna(row['Close']) else None,
                int(row['Volume']) if not pd.isna(row['Volume']) else None,
                float(row['Dividends']) if 'Dividends' in row and not pd.isna(row['Dividends']) else None,
                int(row['Stock Splits']) if 'Stock Splits' in row and not pd.isna(row['Stock Splits']) else None,
                symbol
            ))

        if len(insert_data) >= batch_size:
            cursor.executemany(
                "INSERT INTO crypto_data (Datetime, Open, High, Low, Close, Volume, Dividends, StockSplits, Currency_Code) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                insert_data
            )
            db_conn.commit()
            insert_data.clear()  # Reset the batch

    # Insert remaining rows (if any)
    if insert_data:
        cursor.executemany(
            "INSERT INTO crypto_data (Datetime, Open, High, Low, Close, Volume, Dividends, StockSplits, Currency_Code) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            insert_data
        )
        db_conn.commit()

    cursor.close()
    logging.info(f"Inserted {len(data)} records for {symbol}.")


def load_historical_data(db_conn, cryptos, start_date, end_date, interval='1d'):
    """Fetch and store historical data for a list of cryptocurrencies."""
    for crypto_symbol in cryptos:
        try:
            # Fetch data
            data = fetch_historical_data(crypto_symbol, start_date, end_date, interval)

            # Send data to DB
            send_historical_data_to_db(db_conn, data, crypto_symbol)
        
        except Exception as e:
            logging.error(f"Error loading data for {crypto_symbol}: {e}", exc_info=True)

if __name__ == '__main__':
    # Database connection
    db_connection = get_db_connection()

    # List of cryptocurrencies
    cryptos = fetch_cryptos(db_connection)

    # Define the date range for historical data
    start_date = '2020-01-01'  # Adjust as needed
    end_date = datetime.now().strftime('%Y-%m-%d')  # Up to today

    # Load historical data for all cryptocurrencies
    load_historical_data(db_connection, cryptos, start_date, end_date, interval='1d')

    # Close the database connection
    db_connection.close()
