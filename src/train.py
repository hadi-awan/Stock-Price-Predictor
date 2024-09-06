import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import create_lstm_model, save_model
import os

# Paths to load processed data
PROCESSED_DATA_DIR = 'processed_data'
TRAIN_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_data.npy')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.npy')
TRAIN_LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_labels.npy')
TEST_LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_labels.npy')

# Path to save the trained model
MODEL_SAVE_PATH = 'saved_models/lstm_stock_model.h5'


def load_processed_data():
    """
    Load the preprocessed training and testing data.
    """
    x_train = np.load(TRAIN_DATA_FILE)
    x_test = np.load(TEST_DATA_FILE)
    y_train = np.load(TRAIN_LABELS_FILE)
    y_test = np.load(TEST_LABELS_FILE)

    return x_train, y_train, x_test, y_test


def train_model(x_train, y_train, input_shape):
    """
    Train the LSTM model on the preprocessed data.
    """
    # Create the LSTM model
    model = create_lstm_model(input_shape)

    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss')

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint]
    )

    return model, history


def main():
    # Load the preprocessed data
    x_train, y_train, x_test, y_test = load_processed_data()

    # Get the input shape for the LSTM model
    input_shape = (x_train.shape[1], x_train.shape[2])

    # Train the model
    model, history = train_model(x_train, y_train, input_shape)

    # Save the final model (even though the best one is saved automatically by ModelCheckpoint)
    save_model(model, MODEL_SAVE_PATH)

    print("Model training complete and saved to disk.")


if __name__ == '__main__':
    main()
