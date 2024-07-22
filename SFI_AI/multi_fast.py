import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
from requests.exceptions import SSLError, RequestException
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import ta
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
import holidays
import logging
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed

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
        data['Direction'] = np.where(data['Return'] > 0, 1, 0)
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
    data = data.copy()
    data.loc[:, 'Hour'] = data.index.hour
    data.loc[:, 'DayOfWeek'] = data.index.dayofweek
    data.loc[:, 'Minute'] = data.index.minute
    daily_data = daily_data.resample('T').ffill().reindex(data.index)
    data.loc[:, 'Daily_Open'] = daily_data['Open']
    data.loc[:, 'Daily_Close'] = daily_data['Close']
    data.loc[:, 'SMA_5'] = data['Close'].rolling(window=5).mean()
    data.loc[:, 'SMA_10'] = data['Close'].rolling(window=10).mean()
    data.loc[:, 'EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data.loc[:, 'EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data.loc[:, 'MACD'] = data['EMA_12'] - data['EMA_26']
    data.loc[:, 'Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data.loc[:, 'RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data.loc[:, 'BB_Mid'] = bollinger.bollinger_mavg()
    data.loc[:, 'BB_Upper'] = bollinger.bollinger_hband()
    data.loc[:, 'BB_Lower'] = bollinger.bollinger_lband()
    data.loc[:, 'OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    data.loc[:, 'A/D'] = ta.volume.AccDistIndexIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()
    data.loc[:, 'ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
    aroon = ta.trend.AroonIndicator(high=data['High'], low=data['Low'], window=25)
    data.loc[:, 'Aroon_Up'] = aroon.aroon_up()
    data.loc[:, 'Aroon_Down'] = aroon.aroon_down()
    stochastic = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=14, smooth_window=3)
    data.loc[:, 'Stochastic_Oscillator'] = stochastic.stoch()
    data = add_holiday_feature(data)
    data.dropna(inplace=True)
    features = data[['Hour', 'DayOfWeek', 'Minute', 'Daily_Open', 'Daily_Close', 'SMA_5', 'SMA_10', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'RSI', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'OBV', 'A/D', 'ADX', 'Aroon_Up', 'Aroon_Down', 'Stochastic_Oscillator', 'Holiday']]
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
        'LinearRegression': {},
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

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_news_sentiment(ticker, start_date, end_date):
    sentiment_analyzer = pipeline('sentiment-analysis')
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    sentiment_scores = []
    last_sentiment = 0
    with ThreadPoolExecutor(max_workers=64) as executor:
        future_to_date = {executor.submit(get_daily_sentiment, sentiment_analyzer, ticker, date, last_sentiment): date for date in tqdm(dates, desc="Aggregating News Sentiment")}
        for future in tqdm(as_completed(future_to_date), total=len(future_to_date), desc="Processing Sentiment Scores"):
            date = future_to_date[future]
            try:
                sentiment_score = future.result()
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
                sentiment_score = last_sentiment
            sentiment_scores.append(sentiment_score)
            last_sentiment = sentiment_score if sentiment_score != 0 else last_sentiment
    return pd.Series(sentiment_scores, index=dates)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_daily_sentiment(sentiment_analyzer, ticker, date, last_sentiment):
    sentiment_score = 0
    try:
        articles = get_news_articles(ticker, date)
        if articles:
            scores = [sentiment_analyzer(article['content'])[0]['score'] * (1 if sentiment_analyzer(article['content'])[0]['label'] == 'POSITIVE' else -1) for article in articles]
            sentiment_score = np.mean(scores) if scores else last_sentiment
        else:
            sentiment_score = last_sentiment
    except RequestException as e:
        logger.error(f"Error fetching articles for date {date}: {e}")
        sentiment_score = last_sentiment
    return sentiment_score

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
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
            articles.append({'title': title, 'content': article.text})
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

def analyze_stocks(tickers):
    results = []
    for ticker in tickers:
        print(f"Processing {ticker}...")
        interval = '1d'
        start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        stock_data = get_stock_data(ticker, interval, start_date, end_date)
        if not stock_data.empty:
            daily_data = get_daily_data(ticker, start_date, end_date)
            sentiment_scores = get_news_sentiment(ticker, start_date, end_date)
            sentiment_scores.index = sentiment_scores.index.tz_localize(None)
            stock_data.index = stock_data.index.tz_localize(None)
            stock_data = stock_data.join(sentiment_scores.rename('Sentiment'), how='left').fillna(0)
            features, labels, prices = create_features(stock_data, daily_data)
            features['Sentiment'] = stock_data['Sentiment']
            best_classification_models, best_model_name, best_accuracy = train_classification_models(features, labels)
            best_classification_model = best_classification_models[best_model_name]
            best_regression_models, best_regression_model_name = train_regression_models(features, prices)
            best_regression_model = best_regression_models[best_regression_model_name]
            new_data = features.iloc[-1:].values
            for model_name, model in best_classification_models.items():
                prediction = predict(model, new_data, model_name)
                print(f"Prediction for the next day with {model_name}:", "Up" if prediction == 1 else "Down")
            historical_predictions = predict(best_regression_model, features, best_regression_model_name)
            future_dates = pd.bdate_range(start=end_date, periods=10)
            historical_data_window = stock_data[-30:]
            historical_features, _, _ = create_features(historical_data_window, daily_data)
            historical_features['Sentiment'] = historical_data_window['Sentiment']
            future_features = generate_gaussian_features(historical_features, future_dates, features.columns)
            future_features['Sentiment'] = predict(best_classification_model, future_features, best_model_name)
            future_predictions = predict(best_regression_model, future_features, best_regression_model_name)
            future_prices = pd.Series(future_predictions, index=future_dates)
            percent_change = ((future_prices.iloc[-1] - stock_data['Close'].iloc[-1]) / stock_data['Close'].iloc[-1]) * 100
            volatility_index = np.std(future_prices)
            results.append((ticker, percent_change, volatility_index))
            print(f"Best Classification Model: {best_model_name}")
            print(f"Best Classification Model Accuracy: {best_accuracy}")
            print(f"Best Regression Model: {best_regression_model_name}")
            print("Predicted stock prices for the next two weeks:")
            print(future_prices)
            plt.figure(figsize=(14, 7))
            plt.plot(stock_data.index[-120:], stock_data['Close'][-120:], color='green', alpha=0.5, linewidth=10, label='Historical Closing Price')
            plt.plot(stock_data.index[-120:], historical_predictions[-120:], color='blue', label='Predicted Historical Closing Price')
            plt.plot(future_prices.index, future_prices, color='red', label='Predicted Future Closing Price')
            plt.xlabel('Date')
            plt.ylabel('Closing Price')
            plt.title(f'{ticker} Stock Price Prediction')
            plt.legend()
            plt.show(block=False)
        else:
            print(f"No stock data available for {ticker}.")
    return results

sp500_tickers = ['AAPL', 'MSFT', 'AMZN', 'META', 'CRWD', 'NVDA', 'GDDY', 'VST', 'DDOG', 'MU', 'TSM', 'ADBE', 'ORCL', 'BA', 'INTC', 'PANW', 'AMD', 'FTNT', 'OXY']

results = analyze_stocks(sp500_tickers)

results_sorted = sorted(results, key=lambda x: (x[1], -x[2]), reverse=True)

print("Top predicted stocks:")
for ticker, percent_change, volatility_index in results_sorted[:5]:
    print(f"{ticker}: Percent Change: {percent_change:.2f}%, Volatility Index: {volatility_index:.2f}")

print("\nBottom predicted stocks:")
for ticker, percent_change, volatility_index in results_sorted[-5:]:
    print(f"{ticker}: Percent Change: {percent_change:.2f}%, Volatility Index: {volatility_index:.2f}")

plt.show()
