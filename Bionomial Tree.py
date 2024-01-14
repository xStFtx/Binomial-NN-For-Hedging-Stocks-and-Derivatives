import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime

# Fetch and update the initial price periodically
def fetch_price(ticker_symbol):
    try:
        return yf.Ticker(ticker_symbol).history(period="1d")["Close"][0]
    except Exception as e:
        print("Error fetching price:", e)
        return None

initial_price = fetch_price("AAPL")

# Check if initial_price is valid
if initial_price is None:
    raise ValueError("Failed to fetch initial price")

# Binomial Tree Parameters
up_factor = 1.2
down_factor = 0.8
num_steps = 10

# Generate the tree
tree = np.zeros((num_steps+1, num_steps+1))
for i in range(num_steps+1):
    for j in range(i+1):
        tree[i, j] = initial_price * (up_factor**j) * (down_factor**(i-j))

# Data Preparation
inputs = np.zeros((num_steps, 3))
outputs = np.zeros((num_steps,))

for i in range(num_steps):
    for j in range(i+1):
        inputs[i] = [tree[num_steps-i, j], tree[num_steps-i-1, j], tree[num_steps-i-1, j+1]]
        outputs[i] = max(0, inputs[i, 0] - inputs[i, 1])

# Normalize inputs
scaler = MinMaxScaler()
inputs_normalized = scaler.fit_transform(inputs)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(inputs_normalized, outputs, test_size=0.2, random_state=42)

# Neural Network Architecture
model = Sequential()
model.add(Dense(50, input_dim=3, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.1)

# Model Evaluation
predicted = model.predict(X_test)
print("Model MSE:", mean_squared_error(y_test, predicted))

# Optimal Hedging Combination Calculation
current_price = initial_price
for i in range(num_steps):
    input_up = np.array([[current_price, tree[num_steps-i-1,0], tree[num_steps-i-1,1]]])
    input_down = np.array([[current_price, tree[num_steps-i-1,0], tree[num_steps-i-1,0]*down_factor]])
    
    predicted_value_if_up = model.predict(scaler.transform(input_up))[0][0]
    predicted_value_if_down = model.predict(scaler.transform(input_down))[0][0]
    
    current_price *= up_factor if predicted_value_if_up < predicted_value_if_down else down_factor

print("Optimal hedging combination: hold {} units of shares and {} units of the derivative.".format(current_price, model.predict(scaler.transform(np.array([[initial_price, tree[0,0], tree[0,1]]])))[0][0]))
