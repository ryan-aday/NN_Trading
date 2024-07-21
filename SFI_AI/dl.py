import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from newspaper import Article
import nltk
from transformers import pipeline
import requests
from requests.exceptions import SSLError, RequestException
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.optimizers import Adam
import ta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import holidays
import logging

nltk.download('vader_lexicon')

# Configure logging to suppress error messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def get_stock_data(ticker, interval, start_date, end_date):
    try:
        data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data fetched for {ticker} with interval {interval} from {start_date} to {end_date}")
        data['Return'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_daily_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, interval='1d', start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No daily data fetched for {ticker} from {start_date} to {end_date}")
        return data[['Open', 'Close']]
    except Exception as e:
        logger.error(f"Error fetching daily data for {ticker}: {e}")
        return pd.DataFrame()

def add_holiday_feature(data):
    us_holidays = holidays.US()
    data['Holiday'] = data.index.map(lambda x: 1 if x in us_holidays else 0)
    return data

def create_features(data, daily_data):
    # Create a copy to avoid SettingWithCopyWarning
    data = data.copy()

    # Time Features
    data.loc[:, 'Hour'] = data.index.hour
    data.loc[:, 'DayOfWeek'] = data.index.dayofweek
    data.loc[:, 'Minute'] = data.index.minute

    # Merging daily open price with intraday data
    daily_data = daily_data.resample('T').ffill().reindex(data.index)
    data.loc[:, 'Daily_Open'] = daily_data['Open']
    data.loc[:, 'Daily_Close'] = daily_data['Close']

    # Moving Averages
    data.loc[:, 'SMA_5'] = data['Close'].rolling(window=5).mean()
    data.loc[:, 'SMA_10'] = data['Close'].rolling(window=10).mean()
    data.loc[:, 'EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data.loc[:, 'EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    data.loc[:, 'MACD'] = data['EMA_12'] - data['EMA_26']
    data.loc[:, 'Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    data.loc[:, 'RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data.loc[:, 'BB_Mid'] = bollinger.bollinger_mavg()
    data.loc[:, 'BB_Upper'] = bollinger.bollinger_hband()
    data.loc[:, 'BB_Lower'] = bollinger.bollinger_lband()

    # On-Balance Volume (OBV)
    data.loc[:, 'OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()

    # Accumulation/Distribution (A/D) Line
    data.loc[:, 'A/D'] = ta.volume.AccDistIndexIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()

    # Average Directional Index (ADX)
    data.loc[:, 'ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()

    # Aroon Indicator
    aroon = ta.trend.AroonIndicator(high=data['High'], low=data['Low'], window=25)
    data.loc[:, 'Aroon_Up'] = aroon.aroon_up()
    data.loc[:, 'Aroon_Down'] = aroon.aroon_down()

    # Stochastic Oscillator
    stochastic = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=14, smooth_window=3)
    data.loc[:, 'Stochastic_Oscillator'] = stochastic.stoch()

    # Add Holiday Feature
    data = add_holiday_feature(data)

    data.dropna(inplace=True)
    features = data[['Hour', 'DayOfWeek', 'Minute', 'Daily_Open', 'Daily_Close', 'SMA_5', 'SMA_10', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'RSI', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'OBV', 'A/D', 'ADX', 'Aroon_Up', 'Aroon_Down', 'Stochastic_Oscillator', 'Holiday']]
    return features, data['Close']

def train_deep_learning_model(features, prices, learning_rate, beta_1, beta_2, epsilon):
    feature_scaler = MinMaxScaler()
    price_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(features)
    prices_scaled = price_scaler.fit_transform(prices.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, prices_scaled, test_size=0.10, random_state=42, shuffle=True)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, feature_scaler, price_scaler, mse

def predict(model, feature_scaler, price_scaler, new_data):
    new_data_scaled = feature_scaler.transform(new_data)
    new_data_scaled = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
    prediction_scaled = model.predict(new_data_scaled)
    prediction = price_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
    return prediction

def get_news_sentiment(ticker, start_date, end_date):
    sentiment_analyzer = pipeline('sentiment-analysis')
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    sentiment_scores = []
    last_sentiment = 0  # Initialize last_sentiment to neutral score

    with ThreadPoolExecutor(max_workers=64) as executor:
        future_to_date = {executor.submit(get_daily_sentiment, sentiment_analyzer, ticker, date, last_sentiment): date for date in tqdm(dates, desc="Aggregating News Sentiment")}
        for future in tqdm(as_completed(future_to_date), total=len(future_to_date), desc="Processing Sentiment Scores"):
            date = future_to_date[future]
            try:
                sentiment_score = future.result()
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
                sentiment_score = last_sentiment  # Use the last sentiment score if there's an error

            sentiment_scores.append(sentiment_score)
            last_sentiment = sentiment_score if sentiment_score != 0 else last_sentiment

    return pd.Series(sentiment_scores, index=dates)

def get_daily_sentiment(sentiment_analyzer, ticker, date, last_sentiment):
    sentiment_score = 0
    try:
        articles = get_news_articles(ticker, date)
        if articles:
            scores = [sentiment_analyzer(article['content'])[0]['score'] * (1 if sentiment_analyzer(article['content'])[0]['label'] == 'POSITIVE' else -1) for article in articles]
            sentiment_score = np.mean(scores) if scores else last_sentiment
        else:
            sentiment_score = last_sentiment  # Use last sentiment score if no articles are found
    except RequestException as e:
        logger.error(f"Error fetching articles for date {date}: {e}")
        sentiment_score = last_sentiment  # Use last sentiment score if there's an error
    return sentiment_score

def get_news_articles(ticker, date):
    query = f"{ticker} stock news {date.strftime('%Y-%m-%d')}"
    url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
    except (SSLError, RequestException) as e:
        logger.error(f"Error fetching articles for date {date}: {e}")
        return []

    articles = []
    for result in soup.find_all('div', class_='BVG0Nb'):
        try:
            title = result.find('div', class_='BNeawe vvjwJb AP7Wnd').text
            link = result.find('a')['href']
            article = Article(link)
            article.download()
            article.parse()
            articles.append({
                'title': title,
                'content': article.text
            })
        except Exception as e:
            logger.error(f"Error processing article for date {date}: {e}")
            continue
    return articles

def generate_gaussian_features(historical_features, future_dates, feature_names):
    future_features = pd.DataFrame(index=future_dates)

    for feature in feature_names:
        mean = historical_features[feature].mean()
        std = historical_features[feature].std()
        future_features[feature] = np.random.normal(mean, std, len(future_dates))

    return future_features

# Prompt user to enter stock ticker
ticker = input("Please enter the stock ticker symbol: ").upper()

# Parameters
interval = '1d'
start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# Get stock data
stock_data = get_stock_data(ticker, interval, start_date, end_date)

if not stock_data.empty:
    # Get daily open and close prices
    daily_data = get_daily_data(ticker, start_date, end_date)

    # Get news sentiment scores
    sentiment_scores = get_news_sentiment(ticker, start_date, end_date)

    # Ensure datetime indices are compatible
    sentiment_scores.index = sentiment_scores.index.tz_localize(None)
    stock_data.index = stock_data.index.tz_localize(None)

    # Align sentiment scores with stock data
    stock_data = stock_data.join(sentiment_scores.rename('Sentiment'), how='left').fillna(0)

    # Create features and labels
    features, prices = create_features(stock_data, daily_data)
    features['Sentiment'] = stock_data['Sentiment']

    # Define the fixed parameter set
    param_grid = {
        'learning_rate': [0.005],
        'beta_1': [0.90],
        'beta_2': [0.999],
        'epsilon': [1e-8]
    }

    # Extract parameters
    lr = param_grid['learning_rate'][0]
    b1 = param_grid['beta_1'][0]
    b2 = param_grid['beta_2'][0]
    eps = param_grid['epsilon'][0]

    # Train the model with fixed parameters
    model, feature_scaler, price_scaler, mse = train_deep_learning_model(features, prices, lr, b1, b2, eps)
    print(f"MSE for fixed params: {mse}")

    # Example of making a prediction with the model
    new_data = features[-1:]  # Using the most recent data point as an example
    prediction = predict(model, feature_scaler, price_scaler, new_data)
    print("Prediction for the next day:", prediction)

    # Predict historical prices using the model
    historical_predictions_scaled = model.predict(feature_scaler.transform(features).reshape((features.shape[0], features.shape[1], 1)))
    historical_predictions = price_scaler.inverse_transform(historical_predictions_scaled.reshape(-1, 1)).flatten()

    # Generate future feature data for the next two weeks using Gaussian estimates
    future_dates = pd.date_range(start=end_date, periods=14, freq='B')
    # Use only the past 30 days of historical data for generating future features
    historical_data_window = stock_data[-30:]
    historical_features, _ = create_features(historical_data_window, daily_data)
    historical_features['Sentiment'] = historical_data_window['Sentiment']
    
    future_features = generate_gaussian_features(historical_features, future_dates, features.columns)

    # Ensure future features are scaled consistently
    future_features_scaled = feature_scaler.transform(future_features).reshape((future_features.shape[0], future_features.shape[1], 1))

    # Predict stock price for the next two weeks using the model
    future_predictions_scaled = model.predict(future_features_scaled)
    future_predictions = price_scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1)).flatten()
    future_prices = pd.Series(future_predictions, index=future_dates)
    print("Predicted stock prices for the next two weeks:")
    print(future_prices)

    # Plot historical, predicted historical, and future predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index[-120:], stock_data['Close'][-120:], color='green', label='Historical Closing Price')
    plt.plot(stock_data.index[-120:], historical_predictions[-120:], color='blue', label='Model Predicted Historical Price')
    plt.plot(future_prices.index, future_prices, color='red', label='Predicted Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    plt.show()

else:
    print("No stock data available to train the model.")

