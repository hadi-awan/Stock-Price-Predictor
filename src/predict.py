import numpy as np
from model import load_model
import matplotlib.pyplot as plt
import os

# Paths to load processed data and trained model
PROCESSED_DATA_DIR = 'processed_data'
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.npy')
TEST_LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_labels.npy')

MODEL_PATH = 'saved_models/lstm_stock_model.h5'


# Function to load test data
def load_test_data():
    """
    Load the preprocessed test data and labels.
    """
    x_test = np.load(TEST_DATA_FILE)
    y_test = np.load(TEST_LABELS_FILE)

    return x_test, y_test


# Function to make predictions using the trained model
def make_predictions(model, x_test):
    """
    Use the trained model to make predictions on the test data.

    Args:
        model: Trained LSTM model.
        x_test: Test data.

    Returns:
        predictions: Model predictions.
    """
    predictions = model.predict(x_test)
    return predictions


# Function to plot the actual vs predicted stock prices
def plot_predictions(y_test, predictions):
    """
    Plot the actual vs predicted stock prices.

    Args:
        y_test: Actual stock prices.
        predictions: Predicted stock prices.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


def main():
    # Load the test data
    x_test, y_test = load_test_data()

    # Load the trained model
    model = load_model(MODEL_PATH)

    # Make predictions
    predictions = make_predictions(model, x_test)

    # Plot the predictions vs actual values
    plot_predictions(y_test, predictions)


if __name__ == '__main__':
    main()
