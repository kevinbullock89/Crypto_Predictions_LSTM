from loading_finance import *
from finance_config import *
#from tensorflow_lstm import *

if __name__ == '__main__':
    #check_and_start_docker_container('trading')
    db_connection = get_db_connection()
    cryptos = fetch_cryptos(db_connection)
    last_datetimes = fetch_last_datetimes(db_connection)
    last_docker_check = time.time()

    # Flag to track the last training date
    last_training_date = datetime.now().date() - timedelta(days=1)

    while True:
        try:
            #check_and_start_docker_container('trading')
            sleep_until_next_5_minute()

            # Load stock price data
            for crypto_symbol in cryptos:
                try:
                    last_datetime = last_datetimes.get(crypto_symbol, datetime.utcnow().replace(tzinfo=pytz.utc))
                    data, period = fetch_stock_data(crypto_symbol, last_datetime)
                    logging.info(f"Currency Code: {crypto_symbol}, Period: {period}")
                    send_data_to_db(db_connection, data, crypto_symbol, last_datetimes)

                except Exception as e:
                    logging.error(f"Error processing {crypto_symbol}: {e}")

            # # Run daily training if it's a new day
            # if datetime.now().date() > last_training_date:
            #     try:
            #         run_training()
            #         last_training_date = datetime.now().date()
            #         logging.info("Daily training completed.")
            #     except Exception as e:
            #         logging.error(f"Error during daily training: {e}")

            # # Run future predictions
            # try:
            #     run_future_predictions(future_steps=10)
            # except Exception as e:
            #     logging.error(f"Error during future predictions: {e}")

            # # Run predictions
            # try:
            #     run_predictions()
            # except Exception as e:
            #     logging.error(f"Error during predictions: {e}")

        except Exception as e:
            logging.critical(f"Main loop error: {e}", exc_info=True)
            time.sleep(60)  # Pause and retry
