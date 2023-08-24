import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import datetime

# Get user input for the stock ticker symbol
ticker = input("Enter the stock ticker symbol: ")

# Download historical data from Yahoo Finance
start_date = datetime.datetime.now() - datetime.timedelta(days=365)
end_date = datetime.datetime.now()
data = yf.download(ticker, start=start_date, end=end_date)

# Prepare the data
data['Date'] = data.index
data['Close'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(data_scaled) * split_ratio)
train_data = data_scaled[:split_index]
test_data = data_scaled[split_index:]

X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Reshape the data for LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test))

# Make predictions
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), test_predictions.reshape(-1, 1)), axis=1))[:, -1]

print("LSTM Model Predictions:", test_predictions)
