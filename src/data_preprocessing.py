import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Paths to save processed data
PROCESSED_DATA_DIR = 'processed_data'
TRAIN_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_data.npy')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.npy')
TRAIN_LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_labels.npy')
TEST_LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_labels.npy')


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_data(file_path):
    """
    Load stock price data from a CSV file.
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def preprocess_data(data, time_step=60):
    """
    Scale the data and create sequences for LSTM training.
    """
    # Scale data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM model
    x, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        x.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])

    x, y = np.array(x), np.array(y)
    x = x.reshape(x.shape[0], x.shape[1])  # Reshape for LSTM input
    return x, y, scaler


def split_data(x, y, split_ratio=0.8):
    """
    Split the data into training and testing sets.
    """
    train_size = int(len(x) * split_ratio)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return x_train, y_train, x_test, y_test


def save_data(x_train, y_train, x_test, y_test):
    """
    Save the processed data to disk for later use.
    """
    create_directory(PROCESSED_DATA_DIR)
    np.save(TRAIN_DATA_FILE, x_train)
    np.save(TEST_DATA_FILE, x_test)
    np.save(TRAIN_LABELS_FILE, y_train)
    np.save(TEST_LABELS_FILE, y_test)


def main():
    # Load data
    df = load_data('data/stock_prices.csv')  # Replace with your dataset file path

    # Use only the 'Close' prices for prediction
    close_prices = df['Close'].values
    close_prices = close_prices.reshape(-1, 1)

    # Preprocess data
    x, y, scaler = preprocess_data(close_prices)

    # Split into training and testing sets
    x_train, y_train, x_test, y_test = split_data(x, y)

    # Save processed data
    save_data(x_train, y_train, x_test, y_test)

    # Save the scaler for inverse transformations during prediction
    scaler_file = os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl')
    pd.to_pickle(scaler, scaler_file)

    print("Data preprocessing completed and saved.")


if __name__ == '__main__':
    main()
