import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
import logging

# Setting up colored logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]

class StockPredictionModel:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.initial_price = self.fetch_price()
        if self.initial_price is None:
            raise ValueError("Failed to fetch initial price")
        self.up_factor = 1.2
        self.down_factor = 0.8
        self.num_steps = 10
        self.tree = self.generate_tree()
        self.model = self.build_model()

    def fetch_price(self):
        try:
            price = yf.Ticker(self.ticker_symbol).history(period="1d")["Close"][0]
            logger.info(f"[Blue Log] Fetched initial price: {price}")
            return price
        except Exception as e:
            logger.error(f"[Green Log] Error fetching price: {e}")
            return None

    def generate_tree(self):
        tree = np.zeros((self.num_steps+1, self.num_steps+1))
        for i in range(self.num_steps+1):
            for j in range(i+1):
                tree[i, j] = self.initial_price * (self.up_factor**j) * (self.down_factor**(i-j))
        logger.info("[Blue Log] Generated binomial tree")
        return tree

    def prepare_data(self):
        inputs = np.zeros((self.num_steps, 3))
        outputs = np.zeros((self.num_steps,))

        for i in range(self.num_steps):
            for j in range(i+1):
                inputs[i] = [self.tree[self.num_steps-i, j], self.tree[self.num_steps-i-1, j], self.tree[self.num_steps-i-1, j+1]]
                outputs[i] = max(0, inputs[i, 0] - inputs[i, 1])

        self.scaler = MinMaxScaler()
        inputs_normalized = self.scaler.fit_transform(inputs)
        return train_test_split(inputs_normalized, outputs, test_size=0.2, random_state=42)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(3, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X_train, y_train):
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.1)
        logger.info("[Blue Log] Model training complete")

    def evaluate_model(self, X_test, y_test):
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        predicted = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predicted)
        logger.info(f"[Green Log] Model MSE: {mse}")

    def predict_optimal_hedging(self):
        current_price = self.initial_price
        for i in range(self.num_steps):
            input_up = np.array([[current_price, self.tree[self.num_steps-i-1,0], self.tree[self.num_steps-i-1,1]]])
            input_down = np.array([[current_price, self.tree[self.num_steps-i-1,0], self.tree[self.num_steps-i-1,0]*self.down_factor]])

            # Transform and reshape data to match LSTM input shape
            input_up_transformed = self.scaler.transform(input_up.reshape(1, -1))
            input_down_transformed = self.scaler.transform(input_down.reshape(1, -1))

            input_up_reshaped = input_up_transformed.reshape((1, input_up_transformed.shape[1], 1))
            input_down_reshaped = input_down_transformed.reshape((1, input_down_transformed.shape[1], 1))

            # Use the reshaped data for prediction
            predicted_value_if_up = self.model.predict(input_up_reshaped)[0][0]
            predicted_value_if_down = self.model.predict(input_down_reshaped)[0][0]

            current_price *= self.up_factor if predicted_value_if_up < predicted_value_if_down else self.down_factor

        logger.info(f"[Blue Log] Optimal hedging combination: hold {current_price} units of shares and {self.model.predict(self.scaler.transform(np.array([[self.initial_price, self.tree[0,0], self.tree[0,1]]]).reshape(1, -1)).reshape((1, 3, 1)))[0][0]} units of the derivative.")

# Usage
model = StockPredictionModel("AAPL")
X_train, X_test, y_train, y_test = model.prepare_data()
model.train_model(X_train, y_train)
model.evaluate_model(X_test, y_test)
model.predict_optimal_hedging()