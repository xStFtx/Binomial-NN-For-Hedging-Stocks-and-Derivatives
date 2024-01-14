import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import logging
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Enhanced Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]

class AdvancedStockPredictionModel:
    def __init__(self, ticker_symbol, start_date, end_date):
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.fetch_data()
        self.scaler = MinMaxScaler()
        self.model = self.build_model()

    def fetch_data(self):
        try:
            data = yf.download(self.ticker_symbol, start=self.start_date, end=self.end_date)
            logger.info("[INFO] Data fetched successfully")
            return data
        except Exception as e:
            logger.error(f"[ERROR] Error fetching data: {e}")
            return None

    def feature_engineering(self):
        # Add technical indicators as features
        self.data['SMA'] = self.data['Close'].rolling(window=15).mean()
        self.data['EMA'] = self.data['Close'].ewm(span=15, adjust=False).mean()
        # Additional features can be added here
        self.data.dropna(inplace=True)  # Handling missing values

    def prepare_data(self):
        self.feature_engineering()
        dataset = self.data.values
        scaled_data = self.scaler.fit_transform(dataset)

        # Creating a data structure with 60 timesteps and 1 output
        X = []
        y = []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def build_model(self):
        model = Sequential([
            Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(60, 1))),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X_train, y_train):
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        callback_list = [
            EarlyStopping(monitor='val_loss', patience=10),
            ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=callback_list)
        logger.info("[INFO] Model training complete")

    def evaluate_model(self, X_test, y_test):
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        logger.info(f"[INFO] Model Evaluation - MSE: {mse}, MAE: {mae}")

    def plot_predictions(self, X_test, y_test):
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        predictions = self.model.predict(X_test)
        plt.figure(figsize=(10,6))
        plt.plot(y_test, color='blue', label='Actual Stock Price')
        plt.plot(predictions, color='red', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

# Example usage
model = AdvancedStockPredictionModel("AAPL", "2020-01-01", "2021-01-01")
X_train, X_test, y_train, y_test = model.prepare_data()
model.train_model(X_train, y_train)
model.evaluate_model(X_test, y_test)
model.plot_predictions(X_test, y_test)
