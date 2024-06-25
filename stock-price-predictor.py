# Import libraries
import pandas as pd
import numpy as np
import math
from pandas_datareaders_unofficial import DataReader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense , LSTM
import time
import datetime
from datetime import timedelta


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


expire_after = timedelta(hours=1)

# Get the stock

df = DataReader('GOOGL', expire_after=expire_after).get(data_source='yahoo', start='2012-01-01', end='2020-12-31')  # Getting stock data of Google for the given date range

df.head()

df.shape  # Getting the rows & columns

# Visualize the opening price history

plt.figure(figsize=(16, 8))
plt.title('Opening Price History')
plt.plot(df['Open'])

plt.xlabel('Date', fontsize='18')
plt.ylabel('Open Price', fontsize='18')

plt.show()

# Visualize the close price history

plt.figure(figsize = (16, 8))
plt.title('Closing Price History')
plt.plot(df['Close'])

plt.xlabel('Date' , fontsize = '18')
plt.ylabel('Close Price' , fontsize = '18')

plt.show()

# Use a dataframe having only Close column

data = df.filter(['Close'])

# Convert data into np array

dataset = data.values

# Get the number of rows to train the model on

training_data_len = math.ceil(len(dataset) * 0.8) # Specifying 80 percent as training data


# Scaling the data

scaler = MinMaxScaler(feature_range=(0, 1)) # Scales values of dataset between 0 & 1

scaled_data = scaler.fit_transform(dataset)

# Creating training dataset

# Creating the scaled training dataset

train_data = scaled_data[0:training_data_len , :]

# Splitting train into x_train & y_train
x_train = []    # Training features
y_train = []    # Target variables

for i in range(60 , len(train_data)):  # 60 days
    x_train.append(train_data[i-60:i , 0])  # 60 values from position 0 to 59
    y_train.append(train_data[i,0])         # 61st value with position 60

    if (i <= 61):

        print(x_train)
        print(y_train)
        print()

# Convert x_train & y_train into np arrays for putting into LSTM model

x_train , y_train = np.array(x_train) , np.array(y_train)

# Reshaping the data

# LSTM needs it in such a way which is 3d , right now ours is 2d

x_train = np.reshape(x_train , (x_train.shape[0] , x_train.shape[1] , 1))
x_train.shape

# Building the LSTM models
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))

model.add(Dense(25))
model.add(Dense(1))


# Compiling the model
model.compile(optimizer = 'adam' , loss = 'mean_squared_error')

# Training the model
model.fit(x_train , y_train , batch_size = 1, epochs = 1)

# Creating test data set

# Creating array from index 1752 to 2265

test_data = scaled_data[training_data_len - 60 : ,:]

# Creating x_test & y_test
x_test = []
y_test = []

y_test = dataset[training_data_len: , :]  # All values that our model will predict

for i in range(60 , len(test_data)):

    x_test.append(test_data[i-60:i , 0])

# Converting data again , to np array
x_test = np.array(x_test)

# Reshaping this data for LSTM model
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the predicted values
predictions = model.predict(x_test)

predictions = scaler.inverse_transform(
    predictions)  # Unscaling the values from 0-1 to noraml ones that were already present

# Evaluation of our model

# Getting RMSE as a metric
rmse=np.sqrt(np.mean(predictions- y_test)**2)

# Plotting the data
train = data[:training_data_len]
valid = data[training_data_len:]

valid['Predictions'] = predictions

plt.figure(figsize = (16 , 8))

plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.xlabel('Closing Price', fontsize = 18)

plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])

plt.legend(['Train' , 'Validation', 'Predictions'],
           loc = 'lower right')

plt.show()





