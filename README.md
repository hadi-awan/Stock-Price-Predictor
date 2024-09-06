# LSTM Stock Price Prediction

This project implements an LSTM (Long Short-Term Memory) model using TensorFlow/Keras to predict stock prices based on historical data. The goal is to capture temporal dependencies in stock price movements and make accurate predictions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)

## Project Overview

This project focuses on building an LSTM model to predict future stock prices based on historical data. The project includes data preprocessing, model building, training, and evaluation.

### Features:
- Data preprocessing with scaling and splitting
- LSTM model with dropout to prevent overfitting
- Hyperparameter optimization
- Visualization of predicted vs. actual stock prices

## Data

The dataset used for this project contains historical stock prices. Ensure that your data is structured with columns like `Date` and `Close`. The data is stored in the `data/` directory. You can replace `stock_prices.csv` with your dataset.

Note: Large datasets should not be uploaded to GitHub. Use small sample data for demonstration.

## Model Architecture

The model is built using TensorFlow/Keras and consists of two LSTM layers followed by Dense layers. Dropout is used to prevent overfitting. The model is trained on the last 60 days of data to predict the next day's price.

## Installation

To run this project locally, clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/LSTM-Stock-Prediction.git
cd LSTM-Stock-Prediction
pip install -r requirements.txt

## Usage

You can run the project using the following commands:

1. **Data Preprocessing**: Preprocess the data.
   ```bash
   python src/data_preprocessing.py
   python src/train.py
   python src/predict.py

