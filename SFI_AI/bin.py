import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from newspaper import Article
import nltk
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import ta
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
    return features, data['Direction'], data['Close']

def train_classification_models(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'SVC with SVD': Pipeline([
            ('svd', TruncatedSVD(random_state=42)),
            ('svc', SVC(kernel='rbf'))
        ])
    }

    param_grids = {
        'LogisticRegression': {'C': [0.01, 0.1, 1, 10]},
        'RandomForest': {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15]},
        'GradientBoosting': {'n_estimators': [50, 100, 150, 200, 250], 'learning_rate': [0.01, 0.1, 0.15, 0.2]},
        'SVC with SVD': {
            'svd__n_components': [5, 10, 15],
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto']
        }
    }

    best_models = {}
    best_accuracy = 0
    best_model_name = None

    for model_name in models.keys():
        grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model: {model_name}")
        print("Best Parameters:", grid_search.best_params_)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name

    print(f"Best Classification Model: {best_model_name} with accuracy: {best_accuracy}")
    return best_models, best_model_name, best_accuracy

def train_regression_models(features, prices):
    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.1, random_state=42)

    models = {
        'RandomForest': RandomForestRegressor(),
        'LinearRegression': LinearRegression(),
        'SVR': SVR(),
        'ARIMA': sm.tsa.ARIMA
    }

    param_grids = {
        'RandomForest': {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15]},
        'LinearRegression': {},  # Linear regression does not have hyperparameters to tune in this context
        'SVR': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'ARIMA': {'order': [(5, 1, 0), (0, 1, 1), (1, 1, 1), (0, 1, 0)]}
    }

    best_models = {}
    best_mse = float('inf')
    best_model_name = None

    for model_name in models.keys():
        if model_name == 'ARIMA':
            for order in param_grids[model_name]['order']:
                try:
                    model = models[model_name](y_train, order=order).fit()
                    y_pred = model.forecast(steps=len(y_test))
                    mse = mean_squared_error(y_test, y_pred)
                    if mse < best_mse:
                        best_mse = mse
                        best_models[model_name] = model
                        best_model_name = model_name
                        best_params = {'order': order}
                except Exception as e:
                    print(f"Error training ARIMA model with order {order}: {e}")
                    continue
        else:
            grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_models[model_name] = grid_search.best_estimator_
            y_pred = grid_search.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            if mse < best_mse:
                best_mse = mse
                best_model_name = model_name
                best_params = grid_search.best_params_

    print(f"Best Regression Model: {best_model_name} with MSE: {best_mse} and Parameters: {best_params}")
    return best_models, best_model_name

def predict(model, new_data, model_name):
    if model_name == 'ARIMA':
        return model.forecast(steps=len(new_data))
    else:
        return model.predict(new_data)

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

def generate_gaussian_features(stock_data, future_dates, feature_names):
    future_features = pd.DataFrame(index=future_dates)
    for feature in feature_names:
        mean = stock_data[feature].mean()
        std = stock_data[feature].std()
        future_features[feature] = np.random.normal(mean, std * 0.5, len(future_dates))
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
    features, labels, prices = create_features(stock_data, daily_data)
    features['Sentiment'] = stock_data['Sentiment']

    # Train classification models
    best_classification_models, best_model_name, best_accuracy = train_classification_models(features, labels)
    best_classification_model = best_classification_models[best_model_name]

    # Train regression models
    best_regression_models, best_regression_model_name = train_regression_models(features, prices)
    best_regression_model = best_regression_models[best_regression_model_name]

    # Example of making a prediction
    new_data = features.iloc[-1:].values  # Using the most recent data point as an example
    for model_name, model in best_classification_models.items():
        prediction = predict(model, new_data, model_name)
        print(f"Prediction for the next day with {model_name}:", "Up" if prediction == 1 else "Down")

    # Generate future feature data for the next two weeks using Gaussian estimates
    future_dates = pd.date_range(start=end_date, periods=14, freq='B')
    feature_names = ['Hour', 'DayOfWeek', 'Minute', 'Daily_Open', 'Daily_Close', 'SMA_5', 'SMA_10', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'RSI', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'OBV', 'A/D', 'ADX', 'Aroon_Up', 'Aroon_Down', 'Stochastic_Oscillator', 'Sentiment']
    future_features = generate_gaussian_features(stock_data, future_dates, feature_names)

    # Predict stock price for the next two weeks
    future_predictions = predict(best_regression_model, future_features, best_regression_model_name)
    future_prices = pd.Series(future_predictions, index=future_dates)
    print(f"Best Classification Model Accuracy: {best_accuracy}")
    print("Predicted stock prices for the next two weeks:")
    print(future_prices)

    # Plot historical and predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index[-120:], stock_data['Close'][-120:], color='green', label='Historical Closing Price')
    plt.plot(future_prices.index, future_prices, color='red', label='Predicted Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    plt.show()

else:
    print("No stock data available to train the model.")
