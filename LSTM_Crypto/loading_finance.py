import yfinance as yf
import pytz
from datetime import datetime, timedelta
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_cryptos(db_conn):
    cursor = db_conn.cursor()
    cursor.execute("SELECT code FROM crypto_currencies")
    cryptos = [row[0] for row in cursor]
    cursor.close()
    return cryptos

def fetch_last_datetimes(db_conn):
    cursor = db_conn.cursor(dictionary=True)
    cursor.execute("SELECT Currency_Code, Last_Loading_Datetime FROM loading_information")
    last_datetimes = {}
    for row in cursor:
        if row['Last_Loading_Datetime'] is not None:
            last_datetimes[row['Currency_Code']] = (
                row['Last_Loading_Datetime'].astimezone(pytz.utc)
                if row['Last_Loading_Datetime'].tzinfo is None
                else row['Last_Loading_Datetime']
            )
    cursor.close()
    return last_datetimes

def update_last_datetimes(db_conn, last_datetimes, symbol=None):
    cursor = db_conn.cursor()
    
    # If symbol is provided, update only that symbol
    if symbol:
        update_data = [(symbol, last_datetimes[symbol])]
        cursor.executemany(
            "REPLACE INTO loading_information (Currency_Code, Last_Loading_Datetime) VALUES (%s, %s)",
            update_data
        )
    else:
        # If symbol is not provided, process all symbols in last_datetimes
        update_data = [(symbol, last_datetime) for symbol, last_datetime in last_datetimes.items()]
        cursor.executemany(
            "REPLACE INTO loading_information (Currency_Code, Last_Loading_Datetime) VALUES (%s, %s)",
            update_data
        )
    
    db_conn.commit()
    cursor.close()
    if symbol:
        logging.info(f"Updated Last Datetime for {symbol}.")
    else:
        logging.info(f"Updated Last Datetime for {len(last_datetimes)} symbols.")

from datetime import datetime
import pytz

def calculate_period(last_datetime):
    today = datetime.now(pytz.utc)
    
    if last_datetime.tzinfo is None or last_datetime.tzinfo.utcoffset(last_datetime) is None:
        last_datetime = last_datetime.replace(tzinfo=pytz.utc)

    # Calculate differences
    delta_days = (today - last_datetime).days
    delta_months = (today.year - last_datetime.year) * 12 + today.month - last_datetime.month
    delta_years = today.year - last_datetime.year

    # Determine closest available period
    if delta_days <= 1:
        return '1d'
    elif delta_days <= 5:
        return '5d'
    elif delta_months == 1:   # Approximately 1 month
        return '1mo'
    elif delta_months <= 3:
        return '3mo'
    elif delta_months <= 6:
        return '6mo'
    elif delta_years == 1:
        return '1y'
    elif delta_years <= 2:
        return '2y'
    elif delta_years <= 5:
        return '5y'
    elif delta_years <= 10:
        return '10y'
    elif delta_years > 10:
        return 'max'
    else:
        return 'ytd'  # Default to 'ytd' if no other match (e.g., time just under a year)

def fetch_stock_data(symbol, last_datetime):
    period = calculate_period(last_datetime)
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period, interval="5m")

    # Filter data for incremental loading
    if last_datetime:
        hist = hist[hist.index > last_datetime]
    return hist, period

def send_data_to_db(db_conn, data, symbol=None, last_datetimes=None, batch_size=1000):
    if symbol is None or last_datetimes is None:  # Handle individual cases
        last_datetime = None
        if symbol:
            last_datetime = last_datetimes.get(symbol)  # Only check last datetime if symbol is provided
    else:
        last_datetime = last_datetimes.get(symbol)
        
    if data.empty:
        logging.info(f"No new data for {symbol}.")
        return

    # Update last datetime and save data to DB (for batch operations)
    if last_datetime:
        last_datetimes[symbol] = data.index[-1].to_pydatetime().replace(tzinfo=pytz.utc)
        update_last_datetimes(db_conn, {symbol: last_datetimes[symbol]})

    cursor = db_conn.cursor()
    insert_data = []

    for index, row in data.iterrows():
        naive_index = index.tz_localize(None).to_pydatetime()

        # Check if this data already exists in the database
        cursor.execute(
            "SELECT COUNT(1) FROM crypto_data WHERE Datetime = %s AND Currency_Code = %s",
            (naive_index, symbol)
        )
        existing_record = cursor.fetchone()[0]

        if existing_record == 0:  # If no record exists, prepare data for bulk insertion
            insert_data.append((
                naive_index,
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row['Volume']),
                float(row['Dividends']) if not pd.isna(row['Dividends']) else None,
                int(row['Stock Splits']) if not pd.isna(row['Stock Splits']) else None,
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
    logging.info(f"Inserted {len(data)} records for {symbol} in batches.")


def sleep_until_next_5_minute():
    now = datetime.now()

    # Calculate the next 5-minute mark
    next_minute = ((now.minute // 5) + 1) * 5
    next_hour = now.hour
    next_day = now

    # Handle overflow of minutes to the next hour
    if next_minute == 60:
        next_minute = 0
        next_hour += 1
        if next_hour == 24:  # Handle overflow of hours to the next day
            next_hour = 0
            next_day += timedelta(days=1)

    # Construct the next time point
    next_time = next_day.replace(hour=next_hour, minute=next_minute, second=0, microsecond=0)

    # Calculate sleep time in seconds
    sleep_time = (next_time - now).total_seconds()
    logging.info(f"Sleeping for {sleep_time:.2f} seconds until the next 5-minute mark.")
    
    # Sleep
    time.sleep(sleep_time)
