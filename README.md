# Stock Price Prediction with LSTM

## Overview

This project implements a Long Short-Term Memory (LSTM) model to predict stock prices using historical data. The model is built with TensorFlow and Keras and trained on stock price data retrieved from Yahoo Finance.

## Project Structure

- `data/`: Contains stock price data in CSV format.
- `src/`:
  - `data_preprocessing.py`: Data preprocessing and preparation.
  - `model.py`: LSTM model definition.
  - `train.py`: Training script for the LSTM model.
  - `predict.py`: Script for making predictions with the trained model.
  - `fetch_data.py`: Script for fetching stock price data from Yahoo Finance.
- `saved_models/`: Directory for saving trained models.
- `requirements.txt`: List of dependencies.
- `README.md`: Project description and instructions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock_price_prediction.git
   cd stock_price_prediction
2. Install the required dependencies:
   pip install -r requirements.txt

## Usage

### 1. Download Stock Data
The data/stock_prices.csv file can be generated using the yfinance library. You can download the data using the provided fetch_data.py script:

Run this script to download data:
   python src/utils.py

### 2. Preprocess Data
Run the data_preprocessing.py script to preprocess the data:
   python src/data_preprocessing.py

### 3. Train the Model
Train the LSTM model using the train.py script:
   python src/train.py

### 4. Make Predictions
Use the predict.py script to make predictions with the trained model:
   python src/predict.py

## Files
- data_preprocessing.py: Contains functions to preprocess the stock price data.
- model.py: Defines the LSTM model architecture.
- train.py: Handles training of the LSTM model.
- predict.py: Makes predictions using the trained model.
- fetch_data.py: Downloading stock data.
