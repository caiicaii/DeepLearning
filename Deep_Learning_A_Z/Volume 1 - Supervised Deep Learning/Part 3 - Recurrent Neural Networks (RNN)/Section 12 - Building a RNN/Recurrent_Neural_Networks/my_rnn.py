# Recurrent Neural Network


# Part 1 - Data Preprocessing

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import training set
train_dataset = pd.read_csv('Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv')
training_set = train_dataset.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create data structure with 60 timestamps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building RNN

# Imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize RNN
regressor = Sequential()

# Adding first LSTM layer and Dropout regularisation
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding second LSTM layer and Dropout regularisation
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding fifth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=100))
regressor.add(Dropout(0.2))

# Adding output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting RNN to training set
regressor.fit(X_train, y_train, batch_size=32, epochs=100)


# Part 3 - Making predictions and result visualization

# Getting real stock price of 2017
test_dataset = pd.read_csv('Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv')
real_stock_price = test_dataset.iloc[:, 1:2].values

# Getting predicted stock price of 2017
total_dataset = pd.concat((train_dataset['Open'], test_dataset['Open']), axis=0)
inputs = total_dataset[len(total_dataset) - len(test_dataset) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 60 + len(test_dataset)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Result visualization
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
