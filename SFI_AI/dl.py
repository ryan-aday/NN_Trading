import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from newspaper import Article
import nltk
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
import ta
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

nltk.download('vader_lexicon')

def get_stock_data(ticker, interval, start_date, end_date):
    try:
        data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data fetched for {ticker} with interval {interval} from {start_date} to {end_date}")
        data['Return'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        data['Direction'] = np.where(data['Return'] > 0, 1, 0)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_daily_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, interval='1d', start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No daily data fetched for {ticker} from {start_date} to {end_date}")
        return data[['Open', 'Close']]
    except Exception as e:
        print(f"Error fetching daily data for {ticker}: {e}")
        return pd.DataFrame()

def create_features(data, daily_data):
    # Time Features
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Minute'] = data.index.minute

    # Merging daily open price with intraday data
    daily_data = daily_data.resample('T').ffill().reindex(data.index)
    data['Daily_Open'] = daily_data['Open']
    data['Daily_Close'] = daily_data['Close']

    # Moving Averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['BB_Mid'] = bollinger.bollinger_mavg()
    data['BB_Upper'] = bollinger.bollinger_hband()
    data['BB_Lower'] = bollinger.bollinger_lband()

    # On-Balance Volume (OBV)
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()

    # Accumulation/Distribution (A/D) Line
    data['A/D'] = ta.volume.AccDistIndexIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()

    # Average Directional Index (ADX)
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()

    # Aroon Indicator
    aroon = ta.trend.AroonIndicator(high=data['High'], low=data['Low'], window=25)
    data['Aroon_Up'] = aroon.aroon_up()
    data['Aroon_Down'] = aroon.aroon_down()

    # Stochastic Oscillator
    stochastic = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=14, smooth_window=3)
    data['Stochastic_Oscillator'] = stochastic.stoch()

    data.dropna(inplace=True)
    features = data[['Hour', 'DayOfWeek', 'Minute', 'Daily_Open', 'Daily_Close', 'SMA_5', 'SMA_10', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'RSI', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'OBV', 'A/D', 'ADX', 'Aroon_Up', 'Aroon_Down', 'Stochastic_Oscillator']]
    return features, data['Direction']

def train_deep_learning_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

    model = Sequential()
    model.add(Dense(1024, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Trying different optimizers
    optimizers = [
        ('Adam', {
            'learning_rate': [0.001, 0.0001],
            'beta_1': [0.9, 0.85],
            'beta_2': [0.999, 0.9999],
            'epsilon': [1e-7, 1e-8, 1e-9]
        }),
        ('SGD', {
            'learning_rate': [0.01, 0.1],
            'momentum': [0.8, 0.9]
        }),
        # Placeholder, assuming Lion has similar params
        ('Lion', {
            'learning_rate': [0.001, 0.01],
            'beta_1': [0.9, 0.95],
            'beta_2': [0.999, 0.995],
            'epsilon': [1e-7, 1e-8],
            'amsgrad': [True, False]
        })
    ]

    best_accuracy = 0
    best_optimizer = None

    for opt_name, opt_params in optimizers:
        for lr in opt_params.get('learning_rate', [0.001]):
            for b1 in opt_params.get('beta_1', [0.9]):
                for b2 in opt_params.get('beta_2', [0.999]):
                    for eps in opt_params.get('epsilon', [1e-7]):
                        for amsgrad in opt_params.get('amsgrad', [False]):
                            print(f"Training with optimizer: {opt_name}, learning_rate: {lr}, beta_1: {b1}, beta_2: {b2}, epsilon: {eps}, amsgrad: {amsgrad}")
                            if opt_name == 'Adam':
                                optimizer = Adam(learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=eps)
                            elif opt_name == 'SGD':
                                optimizer = SGD(learning_rate=lr, momentum=b1)
                            elif opt_name == 'Lion':
                                optimizer = Adam(learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=eps, amsgrad=amsgrad)
                            else:
                                continue
                            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.15, verbose=1, shuffle=True)

                            y_pred = (model.predict(X_test) > 0.5).astype("int32")
                            accuracy = accuracy_score(y_test, y_pred)
                            print(f"Results with optimizer: {opt_name}, learning_rate: {lr}, beta_1: {b1}, beta_2: {b2}, epsilon: {eps}, amsgrad: {amsgrad}")
                            print("Accuracy:", accuracy)
                            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
                            print("Classification Report:\n", classification_report(y_test, y_pred))

                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_optimizer = (opt_name, lr, b1, b2, eps, amsgrad)

    print(f"Best optimizer configuration: {best_optimizer} with accuracy: {best_accuracy}")
    return model

def predict(model, new_data):
    return (model.predict(new_data) > 0.5).astype("int32")

def get_news_sentiment(ticker, start_date, end_date):
    sentiment_analyzer = pipeline('sentiment-analysis')
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    sentiment_scores = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_date = {executor.submit(get_daily_sentiment, sentiment_analyzer, ticker, date): date for date in dates}
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            try:
                sentiment_scores.append(future.result())
            except Exception as e:
                print(f"Error processing date {date}: {e}")
                sentiment_scores.append(0)

    return pd.Series(sentiment_scores, index=dates)

def get_daily_sentiment(sentiment_analyzer, ticker, date):
    sentiment_score = 0
    articles = get_news_articles(ticker, date)
    if articles:
        scores = [sentiment_analyzer(article['content'])[0]['score'] * (1 if sentiment_analyzer(article['content'])[0]['label'] == 'POSITIVE' else -1) for article in articles]
        sentiment_score = np.mean(scores) if scores else 0
    return sentiment_score

def get_news_articles(ticker, date):
    query = f"{ticker} stock news {date.strftime('%Y-%m-%d')}"
    url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

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
            continue
    return articles

# Prompt user to enter stock ticker
ticker = input("Please enter the stock ticker symbol: ").upper()[0]

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
    features, labels = create_features(stock_data, daily_data)
    features['Sentiment'] = stock_data['Sentiment']

    # Train deep learning model
    model = train_deep_learning_model(features, labels)

    # Example of making a prediction
    new_data = features.iloc[-1:].values  # Using the most recent data point as an example
    prediction = predict(model, new_data)
    print("Prediction for the next day:", "Up" if prediction == 1 else "Down")
else:
    print("No stock data available to train the model.")
