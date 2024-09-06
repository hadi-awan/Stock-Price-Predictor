import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def create_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """
    Create and compile an LSTM model.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        units (int): Number of units in the LSTM layers.
        dropout_rate (float): Dropout rate to prevent overfitting.

    Returns:
        model: Compiled LSTM model.
    """
    model = Sequential()

    # First LSTM layer with Dropout
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    # Second LSTM layer with Dropout
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))  # Output is a single value (e.g., next day's stock price)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def save_model(model, model_path):
    """
    Save the trained model to disk.

    Args:
        model: Trained model to save.
        model_path (str): Path to save the model.
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path):
    """
    Load a saved model from disk.

    Args:
        model_path (str): Path to load the model from.

    Returns:
        model: Loaded model.
    """
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model
