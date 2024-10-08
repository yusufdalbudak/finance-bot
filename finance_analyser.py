# Install necessary libraries (you may need to install them if not already installed)


from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt

# Alpha Vantage API key (replace with your own API key)
api_key = 'A6LSQZQVHLFMU716'

# Function to fetch stock data from Alpha Vantage API
def get_stock_data(symbol):
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
        return data
    except ValueError as e:
        print(f"Error: {e}")
        return None

# Fetch stock data for a given symbol (e.g., AAPL)
symbol = 'AAPL'  # You can change this to any stock symbol
stock_data = get_stock_data(symbol)

if stock_data is not None:
    # Calculate 20-day and 50-day moving averages
    stock_data['20_MA'] = stock_data['4. close'].rolling(window=20).mean()
    stock_data['50_MA'] = stock_data['4. close'].rolling(window=50).mean()

    # Calculate Bollinger Bands
    stock_data['MA20'] = stock_data['4. close'].rolling(window=20).mean()
    stock_data['stddev'] = stock_data['4. close'].rolling(window=20).std()
    stock_data['Upper_Band'] = stock_data['MA20'] + (stock_data['stddev'] * 2)
    stock_data['Lower_Band'] = stock_data['MA20'] - (stock_data['stddev'] * 2)

    # Calculate RSI
    delta = stock_data['4. close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()
    rs = average_gain / average_loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    exp1 = stock_data['4. close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_data['4. close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = exp1 - exp2
    stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    # Plot the data
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))

    # Plot 1: Closing price with moving averages
    axs[0].plot(stock_data.index, stock_data['4. close'], label='Closing Price', color='blue')
    axs[0].plot(stock_data.index, stock_data['20_MA'], label='20-day MA', color='red')
    axs[0].plot(stock_data.index, stock_data['50_MA'], label='50-day MA', color='green')
    axs[0].set_title(f'{symbol} - Closing Price & Moving Averages')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price (USD)')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Bollinger Bands
    axs[1].plot(stock_data.index, stock_data['4. close'], label='Closing Price', color='blue')
    axs[1].plot(stock_data.index, stock_data['Upper_Band'], label='Upper Band', color='red')
    axs[1].plot(stock_data.index, stock_data['Lower_Band'], label='Lower Band', color='green')
    axs[1].fill_between(stock_data.index, stock_data['Lower_Band'], stock_data['Upper_Band'], color='grey', alpha=0.3)
    axs[1].set_title(f'{symbol} - Bollinger Bands')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price (USD)')
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: RSI
    axs[2].plot(stock_data.index, stock_data['RSI'], label='RSI', color='purple')
    axs[2].axhline(70, color='green', linestyle='--', label='Overbought (70)')
    axs[2].axhline(30, color='red', linestyle='--', label='Oversold (30)')
    axs[2].set_title(f'{symbol} - Relative Strength Index (RSI)')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('RSI')
    axs[2].legend()
    axs[2].grid(True)

    # Plot 4: MACD
    axs[3].plot(stock_data.index, stock_data['MACD'], label='MACD', color='blue')
    axs[3].plot(stock_data.index, stock_data['Signal_Line'], label='Signal Line', color='red')
    axs[3].set_title(f'{symbol} - MACD')
    axs[3].set_xlabel('Date')
    axs[3].set_ylabel('MACD')
    axs[3].legend()
    axs[3].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

else:
    print("Failed to retrieve stock data due to API rate limits or other errors.")
