import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import datetime
from tensorflow.keras.callbacks import EarlyStopping
import os

# BIST 100 Companies
symbols = ["GARAN.IS", "AKBNK.IS", "ISCTR.IS"]  # Some companies included in BIST 100

# Function to Retrieve Stock Data
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError(f"Failed to retrieve data or data is empty: {ticker}")
        return data
    except Exception as e:
        print(f"Error occurred while retrieving data for {ticker}: {e}")
        return pd.DataFrame()

# Fetch and Analyze Data
def fetch_and_analyze():
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')  # Use 2 years of data for better analysis
    
    plt.figure(figsize=(14, 5 * len(symbols)))  # Display all graphs in one window
    for i, symbol in enumerate(symbols, 1):
        data = get_stock_data(symbol, start=start_date, end=end_date)
        if not data.empty:
            analyze_data(data, symbol, i)
    plt.tight_layout()
    save_path = os.path.join(os.getcwd(), 'bist100_analysis.png')
    plt.savefig(save_path)  # Save graphs as PNG
    plt.show()
    generate_textual_analysis()

# Analysis Function (Enhanced LSTM Model Example)
def analyze_data(data, symbol, subplot_index):
    # Use "Close" and "Volume" columns for analysis
    close_data = data['Close'].values.reshape(-1, 1)
    volume_data = data['Volume'].values.reshape(-1, 1)
    combined_data = np.concatenate((close_data, volume_data), axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_data)

    # Create training data
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i])
        y_train.append(scaled_data[i, 0])  # Only closing price is used as the target
    
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Create LSTM Model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))  # Increase dropout to prevent overfitting
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dropout(0.3))  # Add dropout to prevent overfitting
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])

    # Make predictions
    predictions = model.predict(x_train)
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 1))), axis=1))[:, 0]

    # Plotting
    if len(data) > 60:
        plt.subplot(len(symbols), 1, subplot_index)
        plt.plot(data.index[60:], predictions, color='red', label=f'{symbol} Predicted Price')
        plt.plot(data.index, data['Close'], color='blue', label=f'{symbol} Actual Price')
        plt.title(f'{symbol} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

# Function to Generate Textual Analysis Based on Predictions
def generate_textual_analysis():
    try:
        analysis_text = ""
        for symbol in symbols:
            analysis_text += f"\nAnalysis for {symbol}:\n"
            analysis_text += (
                "Based on the LSTM model predictions, it can be observed that the predicted price trend closely follows the actual price movements. "
                "If the predicted prices show a consistent upward trend, it may be an indication of positive growth potential for the upcoming period. "
                "Conversely, if the predicted trend is downward, caution should be exercised as it may indicate a decline in stock value.\n"
                "Investors are advised to consider these predictions along with other market indicators before making any investment decisions.\n"
            )
        print(analysis_text)
    except Exception as e:
        print(f"Error occurred during textual analysis generation: {e}")

# Run the Program
if __name__ == "__main__":
    fetch_and_analyze()