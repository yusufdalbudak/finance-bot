## Stock Analysis using Alpha Vantage API

This script allows you to fetch stock data from Alpha Vantage API and perform technical analysis by calculating indicators such as Moving Averages, Bollinger Bands, RSI (Relative Strength Index)
and MACD (Moving Average Convergence Divergence). The script visualizes the analysis with matplotlib to generate four different plots.

## Requirements
Before you can run the script, make sure you have the following dependencies installed:

## Libraries
alpha_vantage - A Python library to fetch financial data from the Alpha Vantage API.
pandas - A popular library for data manipulation and analysis.
matplotlib - A comprehensive library for creating static, animated, and interactive visualizations in Python.

You can install the necessary libraries by running:
   
    pip install alpha_vantage pandas matplotlib

## API Key
To use the Alpha Vantage API, you will need an API key. You can obtain a free key by registering at
https://www.alphavantage.co/

Once you have the API key, insert it into the api_key variable in the script.

## How to Use
### Setup Your API Key:
Replace the api_key variable in the script with your own Alpha Vantage API key.

    api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'




## Choose a Stock Symbol:
Set the desired stock symbol in the script. For example, 
the script uses AAPL (Apple) as the default stock symbol, 
but you can replace it with any valid stock symbol (e.g., "GOOGL" for Alphabet/Google).



    symbol = 'AAPL'  # You can change this to any stock symbol


## Run the Script:
Once the script is set up, run it using your Python environment. 
The script will retrieve the stock data for the specified symbol, 
calculate technical indicators (Moving Averages, Bollinger Bands, RSI, and MACD), and plot them in four different graphs.





## Visualize Data:
The script generates and displays four subplots:
Closing Price with 20-day and 50-day Moving Averages.
Bollinger Bands showing volatility.
RSI (Relative Strength Index) showing overbought/oversold conditions.
MACD and Signal Line showing momentum.


# Error Handling:
If there are any errors (e.g., API rate limits), the script will notify you and return an error message.





