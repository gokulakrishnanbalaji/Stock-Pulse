import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import yaml
import argparse
import logging

# Set up logging configuration
logging.basicConfig(
    filename='pipeline.log',              # Log file path
    level=logging.INFO,              # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load ticker symbols from YAML
with open('config/stocks.yaml', 'r') as file:
    NSE_TICKERS = yaml.safe_load(file)
    logging.info('loaded stocks yaml file from config ')

def fetch_data(save_path='data/raw/', period='200d', interval='1d', start_date=None):
    os.makedirs(save_path, exist_ok=True)

    # Parse period (e.g., '200d' -> 200 days)
    if not period.endswith('d'):
        raise ValueError("Period must be in the format 'Nd' (e.g., '200d')")
    try:
        days = int(period[:-1])
    except ValueError:
        raise ValueError("Period must contain a valid number of days (e.g., '200d')")

    # Determine start and end dates
    if start_date:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        elif not isinstance(start_date, datetime):
            raise ValueError("start_date must be a string (YYYY-MM-DD) or datetime object")
        
        end_date = start_date + timedelta(days=days)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

    # Make sure start_date is before end_date
    if start_date >= end_date:
        raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")

    for name, ticker in NSE_TICKERS.items():
        logging.info(f"Downloading {name} from {start_date.date()} to {end_date.date()}...")
        df = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval=interval,
            group_by='ticker',
            auto_adjust=False
        )
        
        if not df.empty:
            # Flatten columns if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(1)

            df = df.reset_index()
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df['Date'] = pd.to_datetime(df['Date'])

            # Save logic
            csv_file = os.path.join(save_path, f"{name}.csv")

            if os.path.exists(csv_file):
                existing_df = pd.read_csv(csv_file)
                existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                last_date = existing_df['Date'].max()

                new_data = df[df['Date'] > last_date]
                if not new_data.empty:
                    new_data.to_csv(csv_file, mode='a', header=False, index=False)
                    logging.info(f"Appended new data to {csv_file}")
                else:
                    logging.info(f"No new data to append for {name}")
            else:
                df.to_csv(csv_file, index=False)
                logging.info(f"Created new file {csv_file}")
        else:
            logging.info(f"Warning: No data for {name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download stock data using yfinance.')

    parser.add_argument('--save_path', type=str, required=True, help='Path to save the data.')
    parser.add_argument('--period', type=str, default='200d', help='Period of data to download, e.g., "200d".')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval, e.g., "1d", "1h".')
    parser.add_argument('--start_date', type=str, default=None, help='Optional start date (YYYY-MM-DD).')

    args = parser.parse_args()

    

    fetch_data(
        save_path=args.save_path,
        period=args.period,
        interval=args.interval,
        start_date=args.start_date
    )
