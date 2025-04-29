import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import argparse
import joblib
import logging

# Set up logging configuration
logging.basicConfig(
    filename='pipeline.log',              # Log file path
    level=logging.INFO,              # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def compute_technical_indicators(df):
    df['SMA_5'] = df['Close'].rolling(window=4).mean()
    df['SMA_10'] = df['Close'].rolling(window=4).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(4).mean()
    loss = (-delta.clip(upper=0)).rolling(4).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    return df

def process_stock(df):
    df = compute_technical_indicators(df)

    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'RSI', 'MACD']
    df = df[feature_columns]
    df = df.dropna().reset_index(drop=True)


    if os.path.exists('models/scaler/scaler.pkl'):
        scaler = joblib.load('models/scaler/scaler.pkl')
        X = scaler.transform(df)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(df)
        os.makedirs('models/scaler', exist_ok=True)
        joblib.dump(scaler, 'models/scaler/scaler.pkl')

    # Binary label: 1 if next day's Close > today's Close, else 0
    
    if len(X)>1:
        y = (df['Close'].shift(-1) > df['Close']).astype(int)[:-1]
        X = X[:-1]  # Remove last sample because no label for it
    else:
        y = (df['Close'].shift(-1) > df['Close']).astype(int)
        
    
    return X, y.values

def process_all_stocks(raw_data_path='data/raw/', save_path='data/processed/'):
    os.makedirs(save_path, exist_ok=True)

    all_features = []
    all_labels = []

    for filename in os.listdir(raw_data_path):
        if filename.endswith('.csv'):
            logging.info(f"Processing {filename}...")
            df = pd.read_csv(os.path.join(raw_data_path, filename))
            X, y = process_stock(df)
            all_features.append(X)
            all_labels.append(y)

    

    X_total = np.concatenate(all_features, axis=0)
    y_total = np.concatenate(all_labels, axis=0)

    np.save(os.path.join(save_path, 'X.npy'), X_total)
    np.save(os.path.join(save_path, 'y.npy'), y_total)

    logging.info(f"Saved {X_total.shape[0]} samples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process stock CSV files into input features and labels.')

    parser.add_argument('--raw_data_path', type=str, default='data/raw/', help='Path to raw stock data CSVs.')
    parser.add_argument('--save_path', type=str, default='data/processed/', help='Path to save processed numpy arrays.')

    args = parser.parse_args()

    logging.info(f"Arguments received:\nRaw data path: {args.raw_data_path}\nSave path: {args.save_path}")

    process_all_stocks(
        raw_data_path=args.raw_data_path,
        save_path=args.save_path
    )
